"""
多版本测试脚本 - 使用RemoteOJSubmitter测试所有C++版本
对于每个代码，测试c++14/17/20/23所有版本，如果任意版本失败（排除CE）则认为代码错误
"""
import os
import polars as pl
from remote_submitter import RemoteOJSubmitter
import gc
import json
from typing import Optional
import psutil

dataset_path = "./dataset/ccplus_1x"

# 初始化远程提交器
submitter = RemoteOJSubmitter(
    base_url="http://localhost:8000",
    max_workers=8
)

# 输出目录
output_dir = "output_passed_incorrect_multiversion"
checkpoint_file = "checkpoint_multiversion.json"

# 创建输出目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"✓ Created output directory: {output_dir}")

# 批量处理大小
BATCH_SIZE = 100

# 内存管理配置
MAX_PROCESSED_PROBLEMS_IN_MEMORY = 10000
MEMORY_WARNING_THRESHOLD_MB = 8000

# 所有C++版本
CPP_VERSIONS = ["c++14", "c++17", "c++20", "c++23"]

def get_memory_usage_mb():
    """获取当前进程的内存使用量（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def check_memory_usage():
    """检查内存使用，如果超过阈值则警告"""
    mem_mb = get_memory_usage_mb()
    if mem_mb > MEMORY_WARNING_THRESHOLD_MB:
        print(f"⚠ WARNING: Memory usage is high: {mem_mb:.1f} MB")
    return mem_mb

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

def test_code_all_versions(problem_id: str, code: str, problem_info: dict) -> dict:
    """
    测试代码在所有C++版本上的表现
    从c++14到c++23顺序测试，一旦某个版本失败（非CE）就立即停止

    Returns:
        {
            "all_passed": bool,  # 所有非CE版本都通过
            "version_results": {
                "c++14": {"verdict": "...", "passed": bool},
                "c++17": {...},
                ...
            },
            "stopped_at": str  # 停止的版本（如果提前停止）
        }
    """
    version_results = {}
    stopped_at = None

    for version in CPP_VERSIONS:
        result = submitter.submit_code(
            problem_id=problem_id,
            code=code,
            problem_info=problem_info,
            language=version
        )

        verdict = result.get("verdict", "System Error")
        passed = result.get("passed", False)

        version_results[version] = {
            "verdict": verdict,
            "passed": passed,
            "time": result.get("time", 0),
            "memory": result.get("memory", 0),
            "failed_test": result.get("failed_test")
        }

        # 如果不是CE，检查是否通过
        if verdict != "Compile Error":
            if not passed:
                # 遇到失败，立即停止
                stopped_at = version
                return {
                    "all_passed": False,
                    "version_results": version_results,
                    "stopped_at": stopped_at
                }

    # 如果所有版本都是CE，认为通过（因为无法测试）
    # 如果所有非CE版本都通过，也认为通过
    all_passed = True

    return {
        "all_passed": all_passed,
        "version_results": version_results,
        "stopped_at": stopped_at
    }

def get_first_correct_cpp_code(correct_submissions: list, problem_item: dict, submitter_instance) -> Optional[str]:
    """
    从correct_submissions中获取第一个在所有版本上都AC的C++代码

    Args:
        correct_submissions: 正确提交列表
        problem_item: 题目数据
        submitter_instance: RemoteOJSubmitter实例

    Returns:
        第一个在所有版本上都验证通过的C++代码，如果没有则返回None
    """
    if not correct_submissions:
        return None

    cpp_count = 0
    problem_id = problem_item.get("id")
    problem_info = {
        "test_cases": problem_item.get("test_cases"),
        "checker": problem_item.get("checker"),
        "time_limit": problem_item.get("time_limit", 1000),
        "memory_limit": problem_item.get("memory_limit", 256)
    }

    for submission in correct_submissions:
        if submission.get("language") != 'cpp':
            continue

        code = submission.get("code")
        if not code or len(code.strip()) == 0:
            continue

        cpp_count += 1
        print(f"  验证 correct submission #{cpp_count}...")

        # 测试所有版本
        multi_result = test_code_all_versions(problem_id, code, problem_info)

        if multi_result["all_passed"]:
            print(f"  ✓ Found verified correct code (all versions passed, {len(code)} chars)")
            return code
        else:
            # 显示哪些版本失败了
            failed_versions = [
                v for v, r in multi_result["version_results"].items()
                if not r["passed"] and r["verdict"] != "Compile Error"
            ]
            print(f"  ✗ Failed on versions: {failed_versions}")
            continue

    if cpp_count == 0:
        print(f"  ⚠ No C++ submissions found")
    else:
        print(f"  ⚠ No correct submission passed all tests on all versions (tried {cpp_count} submissions)")
    return None

def append_to_parquet(records: list, output_file: str):
    """追加记录到parquet文件"""
    if not records:
        return

    new_df = pl.DataFrame(records)

    if not os.path.exists(output_file):
        new_df.write_parquet(output_file, compression="zstd")
        del new_df
        gc.collect()
        return

    try:
        existing_df = pl.read_parquet(output_file)
        combined_df = pl.concat([existing_df, new_df], how="vertical_relaxed")
        combined_df.write_parquet(output_file, compression="zstd")

        del existing_df
        del new_df
        del combined_df
        gc.collect()
    except Exception as e:
        print(f"Warning: Error appending to parquet: {e}")
        temp_file = f"{output_file}.tmp"
        new_df.write_parquet(temp_file, compression="zstd")
        print(f"Written to temporary file: {temp_file}")
        del new_df
        gc.collect()

def process_single_problem(item: dict, submitter_instance) -> dict | None:
    """
    处理单个问题，测试所有C++版本

    Returns:
        None 或包含所有通过测试的错误代码列表的字典
    """
    problem_id = item.get("id")

    try:
        print(f"\n[Processing] {problem_id}")

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

        # 构造problem_info
        problem_info = {
            "test_cases": item.get("test_cases"),
            "checker": item.get("checker"),
            "time_limit": item.get("time_limit", 1000),
            "memory_limit": item.get("memory_limit", 256)
        }

        # 测试incorrect submissions
        print(f"  {problem_id}: Testing incorrect submissions on all C++ versions...")
        passed_codes = []

        for idx, code in enumerate(cpp_submissions):
            # 测试所有版本（会在第一个失败版本停止）
            multi_result = test_code_all_versions(problem_id, code, problem_info)

            # 如果所有非CE版本都通过，则认为是"错误地通过"的代码
            if multi_result["all_passed"]:
                passed_codes.append({
                    "code": code,
                    "version_results": multi_result["version_results"]
                })
                print(f"    [{idx+1}/{len(cpp_submissions)}] ✓ Passed on all versions")
            else:
                # 显示在哪个版本失败
                stopped_at = multi_result.get("stopped_at")
                if stopped_at:
                    print(f"    [{idx+1}/{len(cpp_submissions)}] ✗ Failed at {stopped_at}")
                else:
                    print(f"    [{idx+1}/{len(cpp_submissions)}] ✗ Failed")

        if passed_codes:
            print(f"  ✓ {problem_id}: {len(passed_codes)} passed on all versions")
            return {
                "id": problem_id,
                "incorrect_codes": [item["code"] for item in passed_codes],
                "correct_code": correct_code,
                "description": description,
                "checker": item.get("checker"),
                "test_cases": item.get("test_cases"),
                "hack_url": hack_url
            }
        else:
            print(f"  ℹ {problem_id}: No incorrect submissions passed all versions")
            return None

    except Exception as e:
        print(f"  ✗ Error processing {problem_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

# 初始化
print("="*80)
print("Initializing Multi-Version Test...")
print("="*80)

# 显示系统内存信息
total_memory_gb = psutil.virtual_memory().total / (1024**3)
available_memory_gb = psutil.virtual_memory().available / (1024**3)
print(f"System Memory: {total_memory_gb:.1f} GB total, {available_memory_gb:.1f} GB available")
print(f"Initial process memory: {get_memory_usage_mb():.1f} MB")
print(f"Testing C++ versions: {CPP_VERSIONS}")

# 加载断点
checkpoint = load_checkpoint()
processed_problems = set(checkpoint.get("processed_problems", []))
total_saved = checkpoint.get("total_saved", 0)

print(f"✓ Checkpoint loaded: {len(processed_problems)} problems already processed")

for dataset in os.listdir(dataset_path):
    path = os.path.join(dataset_path, dataset)
    if not path.endswith("parquet"):
        continue

    # 为每个输入parquet创建对应的输出文件
    dataset_name = os.path.splitext(dataset)[0]
    output_file = os.path.join(output_dir, f"{dataset_name}_multiversion_passed.parquet")

    print(f"\n{'='*80}")
    print(f"Processing {dataset} -> {os.path.basename(output_file)}")
    print(f"{'='*80}")

    try:
        lazy_df = pl.scan_parquet(path)

        # 只选择需要的列
        selected_columns = [
            "id",
            "test_cases",
            "incorrect_submissions",
            "correct_submissions",
            "description",
            "true_positive_rate",
            "true_negative_rate",
            "checker",
            "time_limit",
            "memory_limit"
        ]
        lazy_df = lazy_df.select(selected_columns)

        # 过滤条件
        lazy_df = lazy_df.filter(
            pl.col("id").str.contains("_") &
            (pl.col("test_cases").list.len() > 0) &
            (pl.col("true_negative_rate") != 1.0) &
            (pl.col("incorrect_submissions").list.len() > 0) &
            (pl.col("correct_submissions").list.len() > 0)
        )

        # 获取总行数
        try:
            total_rows = lazy_df.select(pl.len()).collect().item()
            print(f"Total rows to process: {total_rows}")
        except Exception as e:
            print(f"⚠ Error getting row count: {e}")
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
            lazy_df = None

    except Exception as e:
        print(f"✗ Error opening {dataset}: {e}")
        continue

    for offset in range(0, total_rows, BATCH_SIZE):
        try:
            if lazy_df is not None:
                batch_df = lazy_df.slice(offset, BATCH_SIZE).collect()
            else:
                batch_df = df.slice(offset, BATCH_SIZE)

        except Exception as e:
            print(f"✗ Error loading batch at offset {offset}: {e}")
            continue

        batch_num = offset // BATCH_SIZE + 1
        total_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\n{'='*80}")
        print(f"Processing batch {batch_num}/{total_batches} (problems {offset+1}-{min(offset+BATCH_SIZE, total_rows)})")
        print(f"{'='*80}")

        # 收集需要处理的问题
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

        # 顺序处理（因为RemoteOJSubmitter内部已经有并发控制）
        batch_results = [process_single_problem(item, submitter) for item in items_to_process]

        # 收集结果
        batch_records_to_save = []
        for result, item in zip(batch_results, items_to_process):
            problem_id = item.get("id")

            if result:
                num_codes = len(result.get("incorrect_codes", []))
                print(f"  ✓ {problem_id}: Found {num_codes} code(s) passed all versions")
                batch_records_to_save.append(result)
            else:
                print(f"  ℹ {problem_id}: No incorrect submissions passed all versions")

            processed_problems.add(problem_id)

        # 批次完成后写入结果
        if batch_records_to_save:
            append_to_parquet(batch_records_to_save, output_file)
            total_saved += len(batch_records_to_save)
            print(f"  💾 [Saved] {len(batch_records_to_save)} problems in this batch (Total: {total_saved} problems)")

        # 保存checkpoint
        save_checkpoint(list(processed_problems), total_saved)
        print(f"  💾 [Checkpoint] Saved progress: {len(processed_problems)} problems processed")

        # 检查内存
        mem_mb = check_memory_usage()
        print(f"  📊 Memory usage: {mem_mb:.1f} MB")

        del batch_df
        del batch_results
        del items_to_process
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
save_checkpoint(list(processed_problems), total_saved)

# 最终报告
final_mem_mb = get_memory_usage_mb()
print(f"\n{'='*80}")
print(f"✅ Finished Multi-Version Test!")
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
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        total_size += file_size
        print(f"  - {output_file}: {file_size:.2f} MB")
print(f"Total output size: {total_size:.2f} MB")

print(f"Final memory usage: {final_mem_mb:.1f} MB")
print(f"{'='*80}")
