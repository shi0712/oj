import os
import polars as pl
from submitter import LocalCodeSubmitter
import gc
import tempfile
import subprocess
import platform
import shutil
import json
from typing import Optional
from multiprocessing import Pool, cpu_count
import pickle
from functools import partial
import psutil  # 添加内存监控

dataset_path = "./dataset/ccplus_1x"
submitter = LocalCodeSubmitter()

# 输出目录
output_dir = "output_passed_incorrect"
checkpoint_file = "checkpoint.json"

# 创建输出目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"✓ Created output directory: {output_dir}")  # 断点续传文件

# 批量处理大小
BATCH_SIZE = 100

# 并行处理配置
USE_MULTIPROCESSING = False  # 是否启用多进程
NUM_WORKERS = 5

# 内存管理配置
MAX_PROCESSED_PROBLEMS_IN_MEMORY = 10000  # processed_problems达到此数量时写入磁盘并清空
MEMORY_WARNING_THRESHOLD_MB = 8000  # 内存使用超过此值时发出警告（8GB）

def get_memory_usage_mb():
    """获取当前进程的内存使用量（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def check_memory_usage():
    """检查内存使用，如果超过阈值则警告"""
    mem_mb = get_memory_usage_mb()
    if mem_mb > MEMORY_WARNING_THRESHOLD_MB:
        print(f"⚠ WARNING: Memory usage is high: {mem_mb:.1f} MB")
        print(f"  Consider reducing BATCH_SIZE or enabling checkpointing more frequently")
    return mem_mb

# Compilation parameters
GPP = "g++"
CXX_STANDARDS = ["c++23", "c++20", "c++17", "c++14", "c++11"]
SYSTEM = platform.system()

# 缓存已编译的checker，避免重复编译
checker_cache = {}  # {checker_code_hash: (is_compilable, exe_path or None)}
checker_cache_file = "checker_cache.pkl"  # checker缓存持久化文件

def load_checker_cache():
    """加载checker缓存"""
    global checker_cache
    if os.path.exists(checker_cache_file):
        try:
            with open(checker_cache_file, 'rb') as f:
                checker_cache = pickle.load(f)
            print(f"✓ Loaded checker cache: {len(checker_cache)} entries")
        except Exception as e:
            print(f"⚠ Failed to load checker cache: {e}")
            checker_cache = {}

def save_checker_cache():
    """保存checker缓存"""
    try:
        with open(checker_cache_file, 'wb') as f:
            pickle.dump(checker_cache, f)
        print(f"✓ Saved checker cache: {len(checker_cache)} entries")
    except Exception as e:
        print(f"⚠ Failed to save checker cache: {e}")

def get_code_hash(code: str) -> str:
    """获取代码的hash值用于缓存"""
    import hashlib
    return hashlib.md5(code.encode()).hexdigest() if code else ""

def load_checkpoint():
    """加载断点信息"""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            print(f"✓ Resuming from checkpoint: {checkpoint}")
            return checkpoint
        except Exception as e:
            print(f"⚠ Failed to load checkpoint: {e}")
    return {"processed_problems": [], "total_saved": 0}

def save_checkpoint(processed_problems: list, total_saved: int):
    """保存断点信息"""
    try:
        checkpoint = {
            "processed_problems": processed_problems,
            "total_saved": total_saved,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    except Exception as e:
        print(f"⚠ Failed to save checkpoint: {e}")

def get_compile_flags(std: str) -> list[str]:
    """Get compilation flags based on C++ standard."""
    base_flags = ["-I.", "-DONLINE_JUDGE", "-O2"]

    if SYSTEM == "Darwin":  # macOS
        flags = base_flags
    else:  # Linux or other Unix-like systems
        static_flags = ["-static"]
        flags = static_flags + base_flags


    return flags

def can_compile_code(code: str, prefix: str = "test_") -> bool:
    """Test if a C++ code can compile successfully."""
    if not code or code.strip() == "":
        return False

    temp_dir = tempfile.mkdtemp(prefix=prefix)
    try:
        src_path = os.path.join(temp_dir, "main.cpp")
        exe_path = os.path.join(temp_dir, "main")

        # Write code to file
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(code)

        # Try to compile with different C++ standards
        for std in CXX_STANDARDS:
            compile_flags = get_compile_flags(std)
            compile_command = [GPP, f"-std={std}", *compile_flags, src_path, "-o", exe_path]

            try:
                res = subprocess.run(
                    compile_command,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if res.returncode == 0 and os.path.exists(exe_path):
                    return True

            except (subprocess.TimeoutExpired, Exception):
                continue

        return False

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def can_compile_checker(checker_code: Optional[str]) -> bool:
    """
    Test if a checker (SPJ) code can compile successfully.
    Returns True if the checker is None (traditional problem) or compiles successfully.
    Returns False if compilation fails.
    优化：使用缓存避免重复编译相同的checker
    """
    if checker_code is None or checker_code == "":
        # Traditional problem without SPJ - always valid
        return True

    # 检查缓存
    checker_hash = get_code_hash(checker_code)
    if checker_hash in checker_cache:
        return checker_cache[checker_hash]

    # 编译并缓存结果
    result = can_compile_code(checker_code, prefix="checker_test_")
    checker_cache[checker_hash] = result
    return result

def get_first_correct_cpp_code(correct_submissions: list, problem_item: dict, submitter) -> Optional[str]:
    """
    从correct_submissions中获取第一个真正AC的C++代码
    逐个提交验证，找到第一个通过所有测试的代码后立即返回
    优化：先快速检查编译，编译失败的直接跳过

    Args:
        correct_submissions: 正确提交列表
        problem_item: 题目数据（包含test_cases等）
        submitter: LocalCodeSubmitter实例

    Returns:
        第一个验证通过的C++代码，如果没有则返回None
    """
    if not correct_submissions:
        return None

    cpp_count = 0
    # 逐个测试C++代码
    for idx, submission in enumerate(correct_submissions):
        if submission.get("language") != 'cpp':
            continue

        code = submission.get("code")
        if not code or len(code.strip()) == 0:
            continue

        cpp_count += 1

        if not can_compile_code(code, prefix="correct_quick_"):
            print(f"  ✗ Submission #{cpp_count}: compilation failed")
            continue

        print(f"  验证 correct submission #{cpp_count}...")

        # 单个提交验证
        result = submitter.batch_submit_code(
            problem_item.get("id"),
            [code],  # 只提交一个
            problem_item,
            all_judge=False,
            original_result="correct"
        )

        # 检查是否通过
        if "passed_submissions" in result and len(result["passed_submissions"]) > 0:
            print(f"  ✓ Found verified correct code ({len(code)} chars, tried {cpp_count} submissions)")
            return code
        else:
            print(f"  ✗ Failed tests")
            continue

    if cpp_count == 0:
        print(f"  ⚠ No C++ submissions found")
    else:
        print(f"  ⚠ No correct submission passed all tests (tried {cpp_count} submissions)")
    return None

def append_to_parquet(records: list, output_file: str):
    """
    追加记录到parquet文件
    优化：使用更快的写入方式，避免每次都读取整个文件

    Args:
        records: 要写入的记录列表
        output_file: 输出文件路径
    """
    if not records:
        return

    new_df = pl.DataFrame(records)

    # 优化：如果文件不存在，直接写入
    if not os.path.exists(output_file):
        new_df.write_parquet(output_file, compression="zstd")
        del new_df
        gc.collect()
        return

    # 优化：使用更快的concat方式
    try:
        existing_df = pl.read_parquet(output_file)
        combined_df = pl.concat([existing_df, new_df], how="vertical_relaxed")
        combined_df.write_parquet(output_file, compression="zstd")

        # 显式释放内存
        del existing_df
        del new_df
        del combined_df
        gc.collect()
    except Exception as e:
        print(f"Warning: Error appending to parquet: {e}")
        # 备份方案：写入临时文件
        temp_file = f"{output_file}.tmp"
        new_df.write_parquet(temp_file, compression="zstd")
        print(f"Written to temporary file: {temp_file}")
        del new_df
        gc.collect()

def process_single_problem(item: dict, submitter_instance=None) -> dict | None:
    """
    处理单个问题，返回包含所有通过测试的错误代码列表的单条记录
    用于并行处理

    Returns:
        None 或 一个字典，包含problem_id和所有通过的incorrect_codes列表
    """
    if submitter_instance is None:
        submitter_instance = LocalCodeSubmitter()

    problem_id = item.get("id")

    try:
        print(f"\n[Processing] {problem_id}")

        # 检查 checker
        checker_code = item.get("checker")
        if not can_compile_checker(checker_code):
            print(f"  ✗ {problem_id}: checker cannot compile")
            return None

        # 过滤 C++ 代码
        incorrect_submissions = item.get("incorrect_submissions", [])
        if not incorrect_submissions:
            print(f"  ✗ {problem_id}: no incorrect submissions")
            return None

        cpp_submissions = [
            submission['code']
            for submission in incorrect_submissions
            if submission.get("language") == 'cpp'
        ]

        if len(cpp_submissions) == 0:
            print(f"  ✗ {problem_id}: no C++ submissions")
            return None

        print(f"  {problem_id}: Found {len(cpp_submissions)} C++ incorrect submissions")

        # 获取题目描述
        description = item.get("description", "")

        # 构造hack_url
        contest_id, problem_idx = problem_id.split("_")[0], problem_id.split("_")[1]
        hack_url = f"https://codeforces.com/contest/{contest_id}/hacks?verdictName=CHALLENGE_SUCCESSFUL&chosenProblemIndex={problem_idx}"

        # 验证correct code
        print(f"  {problem_id}: Verifying correct submissions...")
        correct_submissions = item.get("correct_submissions", [])
        correct_code = get_first_correct_cpp_code(correct_submissions, item, submitter_instance)

        # 测试incorrect submissions
        print(f"  {problem_id}: Testing incorrect submissions...")
        incorrect_submissions_result = submitter_instance.batch_submit_code(
            problem_id,
            cpp_submissions,
            item,
            all_judge=False,
            original_result="incorrect"
        )

        # 收集所有通过的codes
        passed_codes = []
        if "passed_submissions" in incorrect_submissions_result:
            for passed_sub in incorrect_submissions_result["passed_submissions"]:
                passed_codes.append(passed_sub["code"])
            print(f"  ✓ {problem_id}: {len(passed_codes)} passed incorrect submission(s)")
        else:
            print(f"  ℹ {problem_id}: No incorrect submissions passed")
            return None

        # 如果有通过的codes，返回单条记录
        if passed_codes:
            return {
                "id": problem_id,
                "incorrect_codes": passed_codes,  # 列表形式
                "correct_code": correct_code,
                "description": description,
                "checker": item.get("checker"),
                "test_cases": item.get("test_cases"),
                "hack_url": hack_url
            }
        else:
            return None

    except Exception as e:
        print(f"  ✗ Error processing {problem_id}: {e}")
        return None

# 初始化
print("="*80)
print("Initializing...")
print("="*80)

# 显示系统内存信息
total_memory_gb = psutil.virtual_memory().total / (1024**3)
available_memory_gb = psutil.virtual_memory().available / (1024**3)
print(f"System Memory: {total_memory_gb:.1f} GB total, {available_memory_gb:.1f} GB available")
print(f"Initial process memory: {get_memory_usage_mb():.1f} MB")
print(f"Memory warning threshold: {MEMORY_WARNING_THRESHOLD_MB} MB")

# 清理临时文件（可能由之前中断的运行留下的）
def cleanup_temp_files():
    """清理当前目录和/tmp下的临时编译文件"""
    import glob
    cleaned = 0

    # 清理当前目录的临时文件
    patterns = [
        "checker_test_*.cpp",
        "checker_test_*",
        "correct_quick_*.cpp",
        "correct_quick_*",
        "*.out",
        "case_*.txt"
    ]

    for pattern in patterns:
        for file in glob.glob(pattern):
            try:
                if os.path.isfile(file):
                    os.remove(file)
                    cleaned += 1
            except Exception:
                pass

    if cleaned > 0:
        print(f"✓ Cleaned up {cleaned} temporary files")

cleanup_temp_files()

# 加载缓存和断点
load_checker_cache()
checkpoint = load_checkpoint()
processed_problems = set(checkpoint.get("processed_problems", []))
total_saved = checkpoint.get("total_saved", 0)

print(f"✓ Checkpoint loaded: {len(processed_problems)} problems already processed")

total_saved = 0

for dataset in os.listdir(dataset_path):
    path = os.path.join(dataset_path, dataset)
    if not path.endswith("parquet"): continue

    # 为每个输入parquet创建对应的输出文件
    dataset_name = os.path.splitext(dataset)[0]  # 去掉.parquet后缀
    output_file = os.path.join(output_dir, f"{dataset_name}_passed_incorrect.parquet")

    print(f"\n{'='*80}")
    print(f"Processing {dataset} -> {os.path.basename(output_file)}")
    print(f"{'='*80}")

    try:
        lazy_df = pl.scan_parquet(path)

        # 只选择需要的列，减少内存占用
        selected_columns = [
            "id",
            "test_cases",
            "incorrect_submissions",
            "correct_submissions",  # 添加正确提交
            "description",  # 添加题目描述
            "true_positive_rate",
            "true_negative_rate",
            "checker"
        ]
        lazy_df = lazy_df.select(selected_columns)

        # 优化：添加更多过滤条件，跳过不需要处理的数据
        lazy_df = lazy_df.filter(
            pl.col("id").str.contains("_") &
            (pl.col("test_cases").list.len() > 0) &
            (pl.col("true_negative_rate") != 1.0) &
            (pl.col("incorrect_submissions").list.len() > 0) &  # 优化：必须有incorrect_submissions
            (pl.col("correct_submissions").list.len() > 0)  # 优化：必须有correct_submissions
        )

        # 获取总行数，使用try-except处理可能的parquet错误
        try:
            total_rows = lazy_df.select(pl.len()).collect().item()
            print(f"Total rows to process: {total_rows}")
        except Exception as e:
            print(f"⚠ Error getting row count with lazy scan: {e}")
            print(f"  Falling back to eager loading...")
            # 回退方案：直接读取并计数
            try:
                df = pl.read_parquet(path, columns=selected_columns)
                df = df.filter(
                    pl.col("id").str.contains("_") &
                    (pl.col("test_cases").list.len() > 0) &
                    (pl.col("true_negative_rate") != 1.0) &
                    (pl.col("incorrect_submissions").list.len() > 0) &
                    (pl.col("correct_submissions").list.len() > 0)
                )
                total_rows = len(df)
                print(f"Total rows to process: {total_rows}")
                # 使用eager loaded df，不再使用lazy_df
                lazy_df = None
            except Exception as e2:
                print(f"✗ Error loading {dataset}: {e2}")
                print(f"  Skipping this dataset...")
                continue

    except Exception as e:
        print(f"✗ Error opening {dataset}: {e}")
        print(f"  Skipping this dataset...")
        continue

    for offset in range(0, total_rows, BATCH_SIZE):
        try:
            # 如果使用lazy_df
            if lazy_df is not None:
                batch_df = lazy_df.slice(offset, BATCH_SIZE).collect()
            else:
                # 如果使用eager loaded df
                batch_df = df.slice(offset, BATCH_SIZE)

        except Exception as e:
            print(f"✗ Error loading batch at offset {offset}: {e}")
            print(f"  Skipping this batch...")
            continue

        batch_num = offset//BATCH_SIZE + 1
        total_batches = (total_rows + BATCH_SIZE - 1)//BATCH_SIZE
        print(f"\n{'='*80}")
        print(f"Processing batch {batch_num}/{total_batches} (problems {offset+1}-{min(offset+BATCH_SIZE, total_rows)})")
        print(f"{'='*80}")

        # 收集需要处理的问题（跳过已处理的）
        items_to_process = []
        for item in batch_df.iter_rows(named=True):
            problem_id = item.get("id")
            if problem_id in processed_problems:
                print(f"  ⏭ Skipping {problem_id} (already processed)")
                continue
            items_to_process.append(item)

        if not items_to_process:
            print(f"  ℹ All problems in this batch already processed")
            del batch_df
            gc.collect()
            continue

        print(f"  Processing {len(items_to_process)} problems in this batch")

        # 并行或顺序处理
        if USE_MULTIPROCESSING and len(items_to_process) > 1:
            print(f"  Using parallel processing with {NUM_WORKERS} workers")
            with Pool(NUM_WORKERS) as pool:
                batch_results = pool.map(process_single_problem, items_to_process)
        else:
            print(f"  Using sequential processing")
            batch_results = [process_single_problem(item, submitter) for item in items_to_process]

        # 收集结果
        batch_records_to_save = []
        for result, item in zip(batch_results, items_to_process):
            problem_id = item.get("id")

            if result:  # result是字典或None
                num_codes = len(result.get("incorrect_codes", []))
                print(f"  ✓ {problem_id}: Found {num_codes} passed incorrect code(s)")
                batch_records_to_save.append(result)
            else:
                print(f"  ℹ {problem_id}: No incorrect submissions passed the tests")

            # 标记为已处理
            processed_problems.add(problem_id)

        # 批次完成后一次性写入所有结果
        if batch_records_to_save:
            append_to_parquet(batch_records_to_save, output_file)
            total_saved += len(batch_records_to_save)
            print(f"  💾 [Saved] {len(batch_records_to_save)} problems in this batch (Total: {total_saved} problems)")

        # 保存checkpoint
        save_checkpoint(list(processed_problems), total_saved)
        print(f"  💾 [Checkpoint] Saved progress: {len(processed_problems)} problems processed")

        # 内存管理：如果processed_problems太大，定期持久化并清空部分内存
        if len(processed_problems) > MAX_PROCESSED_PROBLEMS_IN_MEMORY:
            print(f"  ⚠ processed_problems size: {len(processed_problems)}, flushing to checkpoint...")
            save_checkpoint(list(processed_problems), total_saved)
            # 注意：不清空processed_problems，因为需要用来判断是否已处理

        # 检查内存使用
        mem_mb = check_memory_usage()
        print(f"  📊 Memory usage: {mem_mb:.1f} MB")

        del batch_df
        del batch_results  # 显式删除结果列表
        del items_to_process  # 显式删除待处理列表
        gc.collect()
        print(f"\n  Batch {batch_num} completed. Total saved so far: {total_saved}")

    # 清理DataFrame
    if lazy_df is not None:
        del lazy_df
    else:
        del df
    gc.collect()
    print(f"\n✓ Completed processing {dataset}\n")

# 保存最终状态
save_checker_cache()
save_checkpoint(list(processed_problems), total_saved)

# 清理临时文件
cleanup_temp_files()

# 最终内存报告
final_mem_mb = get_memory_usage_mb()
print(f"\n{'='*80}")
print(f"✅ Finished!")
print(f"{'='*80}")
print(f"Total saved: {total_saved} problems with passed incorrect submissions")
print(f"Total problems processed: {len(processed_problems)}")
print(f"Output directory: {output_dir}")

# 列出所有生成的输出文件
output_files = [f for f in os.listdir(output_dir) if f.endswith('.parquet')]
print(f"Generated {len(output_files)} output files:")
total_size = 0
for output_file in sorted(output_files):
    file_path = os.path.join(output_dir, output_file)
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        total_size += file_size
        print(f"  - {output_file}: {file_size:.2f} MB")
print(f"Total output size: {total_size:.2f} MB")

print(f"Checker cache entries: {len(checker_cache)}")
print(f"Final memory usage: {final_mem_mb:.1f} MB")
print(f"{'='*80}")

