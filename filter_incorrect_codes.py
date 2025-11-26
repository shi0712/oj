"""
过滤多版本测试脚本
输入: *_passed_incorrect.parquet 文件
格式:
  - id: 题目ID (如 "755_A")
  - incorrect_codes: List[str] - 待过滤的代码列表
  - correct_code: str - 正确代码
  - description: str - 题目描述
  - checker: str - SPJ代码（可能为空）
  - test_cases: List[{input, output}] - 测试数据
  - hack_url: str

处理流程:
1. 对每个题目，先上传题目(test_cases, checker)到OJ
2. 对每个incorrect_code，测试c++14 -> c++17 -> c++20 -> c++23
3. 如果任意版本失败（非CE），从列表中移除该代码
4. 输出过滤后的parquet文件
"""
import os
import polars as pl
from remote_submitter import RemoteOJSubmitter
import aiohttp
import asyncio
import gc
import json
import zipfile
import tempfile
from typing import List
from tqdm import tqdm

# 输入目录（包含 *_passed_incorrect.parquet 文件）
input_dir = "../dataset"

# 输出目录
output_dir = "./output_filtered"

# 错误代码保存目录
failed_codes_dir = "./failed_codes"

# checkpoint文件
checkpoint_file = "checkpoint_filter.json"

# OJ服务器地址
OJ_BASE_URL = "http://localhost:8000"

# 创建输出目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"✓ Created output directory: {output_dir}")

# 创建错误代码保存目录
if not os.path.exists(failed_codes_dir):
    os.makedirs(failed_codes_dir)
    print(f"✓ Created failed codes directory: {failed_codes_dir}")

# 初始化远程提交器
submitter = RemoteOJSubmitter(
    base_url=OJ_BASE_URL,
    max_workers=8
)

# 所有C++版本（按顺序测试）
CPP_VERSIONS = ["c++17"]


def load_checkpoint():
    """加载断点信息"""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            print(f"✓ Resuming from checkpoint")
            return checkpoint
        except Exception as e:
            print(f"⚠ Failed to load checkpoint: {e}")
    return {"processed_problems": []}


def save_checkpoint(processed_problems: list):
    """保存断点信息"""
    try:
        checkpoint = {
            "processed_problems": processed_problems,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    except Exception as e:
        print(f"⚠ Failed to save checkpoint: {e}")


async def check_problem_exists(problem_id: str) -> bool:
    """检查题目是否已存在"""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{OJ_BASE_URL}/api/problems/{problem_id}") as resp:
                return resp.status == 200
        except Exception:
            return False


async def upload_problem(problem_id: str, test_cases: list, checker: str = None,
                         time_limit: int = 2000, memory_limit: int = 256):
    """
    上传题目到OJ服务器（如果题目已存在则跳过）

    Args:
        problem_id: 题目ID
        test_cases: 测试数据列表 [{input, output}, ...]
        checker: SPJ代码（可选）
        time_limit: 时间限制(ms)
        memory_limit: 内存限制(MB)

    Returns:
        True if successful or already exists, False otherwise
    """
    # 检查题目是否已存在
    if await check_problem_exists(problem_id):
        print(f"    ✓ Problem already exists, skipping upload")
        return True

    # 创建临时zip文件
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "testcases.zip")

        # 使用进度条显示打包进度
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for idx, tc in tqdm(enumerate(test_cases, 1), total=len(test_cases),
                               desc="    Packing testcases", leave=False):
                input_data = tc.get("input", "")
                output_data = tc.get("output", "")
                zf.writestr(f"{idx}.in", input_data)
                zf.writestr(f"{idx}.out", output_data)

        # 上传到OJ
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field('problem_id', problem_id)
            data.add_field('time_limit', str(time_limit))
            data.add_field('memory_limit', str(memory_limit))

            with open(zip_path, 'rb') as f:
                data.add_field('testcases', f, filename='testcases.zip',
                              content_type='application/zip')

                if checker and checker.strip():
                    # Send checker as file content, not string
                    data.add_field('checker', checker.encode('utf-8'),
                                  filename='checker.cpp',
                                  content_type='text/plain')

                print(f"    Uploading to OJ server...")
                async with session.post(f"{OJ_BASE_URL}/api/problems", data=data) as resp:
                    result = await resp.json()
                    if resp.status == 200:
                        return True
                    else:
                        print(f"    ⚠ Upload failed: {result}")
                        return False


async def test_code_all_versions_async(problem_id: str, code: str, max_retries: int = 2) -> dict:
    """
    异步测试代码在所有C++版本上的表现
    从c++14到c++23顺序测试，一旦某个版本失败（非CE）就立即停止
    System Error 会自动重试（TLE重试已在OJ内部处理）

    Returns:
        {
            "all_passed": bool,
            "stopped_at": str or None,
            "fail_verdict": str or None,
            "failed_test": int or None,
            "version_results": dict
        }
    """
    version_results = {}

    for version in CPP_VERSIONS:
        # System Error 自动重试
        for retry in range(max_retries):
            result = await submitter.submit_code_async(
                problem_id=problem_id,
                code=code,
                language=version
            )

            verdict = result.get("verdict", "System Error")
            passed = result.get("passed", False)
            failed_test = result.get("failed_test")

            # 只对 System Error 重试
            if verdict != "System Error":
                break

            if retry < max_retries - 1:
                await asyncio.sleep(0.5)

        version_results[version] = {
            "verdict": verdict,
            "passed": passed,
            "failed_test": failed_test
        }

        # 如果不是CE，检查是否通过
        if verdict != "Compile Error":
            if not passed:
                # 遇到失败，立即停止
                return {
                    "all_passed": False,
                    "stopped_at": version,
                    "fail_verdict": verdict,
                    "failed_test": failed_test,
                    "version_results": version_results
                }

    # 所有版本都通过（或CE）
    return {
        "all_passed": True,
        "stopped_at": None,
        "fail_verdict": None,
        "failed_test": None,
        "version_results": version_results
    }


def test_code_all_versions(problem_id: str, code: str) -> dict:
    """
    测试代码在所有C++版本上的表现（同步包装）
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(test_code_all_versions_async(problem_id, code))
    finally:
        loop.close()


def save_failed_code(problem_id: str, code: str, index: int, fail_info: dict):
    """
    保存失败的代码到文件

    Args:
        problem_id: 题目ID
        code: 代码内容
        index: 代码序号
        fail_info: 失败信息 {stopped_at, fail_verdict, failed_test}
    """
    # 创建题目目录
    problem_dir = os.path.join(failed_codes_dir, problem_id)
    if not os.path.exists(problem_dir):
        os.makedirs(problem_dir)

    # 保存代码文件
    code_file = os.path.join(problem_dir, f"{index}.cpp")
    with open(code_file, 'w', encoding='utf-8') as f:
        f.write(code)

    # 保存错误信息到同名的txt文件
    info_file = os.path.join(problem_dir, f"{index}.txt")
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f"Failed at: {fail_info['stopped_at']}\n")
        f.write(f"Verdict: {fail_info['fail_verdict']}\n")
        if fail_info['failed_test'] is not None:
            f.write(f"Failed test: {fail_info['failed_test']}\n")


async def filter_incorrect_codes_async(problem_id: str, incorrect_codes: List[str], max_concurrent: int = 2) -> List[str]:
    """
    异步并发过滤 incorrect_codes 列表
    失败的代码会保存到 failed_codes_dir/{problem_id}/ 目录

    Args:
        problem_id: 题目ID
        incorrect_codes: 待测试的代码列表
        max_concurrent: 最大并发数（默认2，避免CPU竞争）
    """
    filtered_codes = []
    failed_code_count = 0
    semaphore = asyncio.Semaphore(max_concurrent)

    async def test_single_code(idx: int, code: str):
        nonlocal failed_code_count
        async with semaphore:
            result = await test_code_all_versions_async(problem_id, code)

            if result["all_passed"]:
                filtered_codes.append(code)
                tqdm.write(f"    [{idx+1}/{len(incorrect_codes)}] ✓ Passed all versions")
                return True
            else:
                stopped_at = result.get("stopped_at")
                fail_verdict = result.get("fail_verdict")
                failed_test = result.get("failed_test")

                # 保存失败的代码
                failed_code_count += 1
                save_failed_code(
                    problem_id=problem_id,
                    code=code,
                    index=failed_code_count,
                    fail_info={
                        'stopped_at': stopped_at,
                        'fail_verdict': fail_verdict,
                        'failed_test': failed_test
                    }
                )

                # 输出详细错误信息
                error_info = f"{fail_verdict}"
                if failed_test is not None:
                    error_info += f" on test {failed_test}"

                tqdm.write(f"    [{idx+1}/{len(incorrect_codes)}] ✗ Failed at {stopped_at}: {error_info} -> Saved as {failed_code_count}.cpp")
                return False

    # 使用进度条显示过滤进度
    tasks = []
    with tqdm(total=len(incorrect_codes), desc="    Testing codes", leave=False) as pbar:
        for idx, code in enumerate(incorrect_codes):
            task = asyncio.create_task(test_single_code(idx, code))
            task.add_done_callback(lambda _: pbar.update(1))
            tasks.append(task)

        await asyncio.gather(*tasks)

    return filtered_codes


def filter_incorrect_codes(problem_id: str, incorrect_codes: List[str]) -> List[str]:
    """
    过滤 incorrect_codes 列表（同步包装）
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            filter_incorrect_codes_async(problem_id, incorrect_codes, max_concurrent=2)
        )
    finally:
        loop.close()


def is_code_failed(problem_id: str, code: str) -> bool:
    """检查某个代码是否在failed_codes目录中
    逐个文件读取并比较，避免一次性加载所有失败代码到内存

    Args:
        problem_id: 题目ID
        code: 待检查的代码内容

    Returns:
        True if code is in failed_codes, False otherwise
    """
    problem_dir = os.path.join(failed_codes_dir, problem_id)
    if not os.path.exists(problem_dir):
        return False

    for filename in os.listdir(problem_dir):
        if filename.endswith('.cpp'):
            filepath = os.path.join(problem_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    failed_code = f.read()
                    if failed_code == code:
                        return True
            except Exception as e:
                print(f"    ⚠ Failed to read {filepath}: {e}")

    return False


def process_parquet_file(input_file: str, output_file: str, processed_problems: set):
    """
    处理单个 parquet 文件
    """
    print(f"\n{'='*80}")
    print(f"Processing: {os.path.basename(input_file)}")
    print(f"{'='*80}")

    # 如果输出文件已存在，直接跳过整个文件
    if os.path.exists(output_file):
        print(f"  ⏭ Output file already exists, skipping entire file")
        return

    # 读取 parquet 文件（禁用内存映射避免 Windows 文件锁定问题）
    try:
        df = pl.read_parquet(input_file, use_pyarrow=False)
        print(f"Loaded {len(df)} rows")
        print(f"Columns: {df.columns}")
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return

    # 处理每一行
    filtered_records = []

    try:
        for row in df.iter_rows(named=True):
            problem_id = row.get("id")

            if problem_id in processed_problems:
                # 从failed_codes恢复filtered_codes（逐个比较，避免加载所有代码到内存）
                print(f"  ⏭ {problem_id} already processed, recovering from failed_codes...")
                incorrect_codes = row.get("incorrect_codes", [])
                if not incorrect_codes:
                    print(f"    ℹ No incorrect codes, skipping")
                    continue

                # 逐个检查代码是否失败，避免一次性加载所有失败代码
                filtered_codes = []
                for code in incorrect_codes:
                    if not is_code_failed(problem_id, code):
                        filtered_codes.append(code)
                    # 立即释放code，避免内存累积
                    del code

                if filtered_codes:
                    # 恢复记录到输出
                    new_record = dict(row)
                    new_record["incorrect_codes"] = filtered_codes
                    filtered_records.append(new_record)
                    print(f"    ✓ Recovered: {len(filtered_codes)}/{len(incorrect_codes)} codes (kept in dataset)")
                else:
                    print(f"    ℹ All codes filtered out (removed from dataset)")

                # 清理临时变量
                del incorrect_codes, filtered_codes
                continue

            incorrect_codes = row.get("incorrect_codes", [])
            if not incorrect_codes:
                print(f"  ℹ {problem_id}: No incorrect codes, skipping")
                processed_problems.add(problem_id)
                continue

            test_cases = row.get("test_cases", [])
            checker = row.get("checker")
            time_limit = row.get("time_limit")
            memory_limit = row.get("memory_limit")

            print(f"\n  [Processing] {problem_id}: {len(incorrect_codes)} codes, {len(test_cases)} test cases")

            # 1. 上传题目到OJ
            print(f"    Uploading problem to OJ...")
            upload_success = asyncio.run(upload_problem(
                problem_id=problem_id,
                test_cases=test_cases,
                checker=checker,
                time_limit=time_limit,
                memory_limit=memory_limit
            ))

            if not upload_success:
                print(f"    ⚠ Failed to upload problem, skipping")
                processed_problems.add(problem_id)
                # 清理大对象
                del test_cases, incorrect_codes
                continue

            print(f"    ✓ Problem uploaded")

            # 2. 过滤代码
            print(f"    Filtering codes...")
            original_count = len(incorrect_codes)
            filtered_codes = filter_incorrect_codes(problem_id, incorrect_codes)

            # 清理不再需要的大对象
            del test_cases, incorrect_codes
            gc.collect()

            # 标记已处理
            processed_problems.add(problem_id)

            if filtered_codes:
                # 创建新记录，保留原有字段，更新 incorrect_codes
                new_record = dict(row)
                new_record["incorrect_codes"] = filtered_codes
                filtered_records.append(new_record)
                print(f"  ✓ {problem_id}: {len(filtered_codes)}/{original_count} codes passed all versions")
            else:
                print(f"  ℹ {problem_id}: All codes filtered out")

            # 清理临时变量
            del filtered_codes

            # 定期保存checkpoint
            save_checkpoint(list(processed_problems))

    finally:
        # 清理DataFrame
        del df
        gc.collect()

    # 一次性写入所有记录
    if filtered_records:
        print(f"\n💾 Writing {len(filtered_records)} records to output file...")
        new_df = pl.DataFrame(filtered_records)
        new_df.write_parquet(output_file, compression="zstd")
        del new_df
        print(f"  ✓ Saved to {output_file}")
    else:
        print(f"\n  ℹ No records to save")

    gc.collect()
    print(f"\n✓ Finished processing {os.path.basename(input_file)}")


# 主程序
if __name__ == "__main__":
    import sys

    # 检查命令行参数
    if len(sys.argv) > 1:
        # 单文件模式：python filter_incorrect_codes.py <filename>
        target_filename = sys.argv[1]
        print("="*80)
        print(f"Filter Incorrect Codes - Single File Mode")
        print("="*80)
        print(f"Target file: {target_filename}")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"C++ versions: {CPP_VERSIONS}")

        # 加载断点
        checkpoint = load_checkpoint()
        processed_problems = set(checkpoint.get("processed_problems", []))

        # 处理单个文件
        input_file = os.path.join(input_dir, target_filename)
        if not os.path.exists(input_file):
            print(f"✗ Error: File not found: {input_file}")
            sys.exit(1)

        output_filename = target_filename.replace("_passed_incorrect.parquet", "_filtered.parquet")
        output_file = os.path.join(output_dir, output_filename)

        process_parquet_file(input_file, output_file, processed_problems)

        # 保存checkpoint
        save_checkpoint(list(processed_problems))

        print(f"\n{'='*80}")
        print(f"✅ Finished processing {target_filename}!")
        print(f"{'='*80}")

    else:
        # 批量模式：python filter_incorrect_codes.py
        print("="*80)
        print("Filter Incorrect Codes - Batch Mode")
        print("="*80)
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"C++ versions: {CPP_VERSIONS}")

        # 加载断点
        checkpoint = load_checkpoint()
        processed_problems = set(checkpoint.get("processed_problems", []))
        print(f"✓ Already processed: {len(processed_problems)} problems")

        # 遍历输入目录
        parquet_files = [f for f in os.listdir(input_dir) if f.endswith("_passed_incorrect.parquet")]
        print(f"Found {len(parquet_files)} parquet files")

        for filename in sorted(parquet_files):
            input_file = os.path.join(input_dir, filename)

            # 生成输出文件名
            output_filename = filename.replace("_passed_incorrect.parquet", "_filtered.parquet")
            output_file = os.path.join(output_dir, output_filename)

            process_parquet_file(input_file, output_file, processed_problems)

        # 保存最终状态
        save_checkpoint(list(processed_problems))

        # 最终报告
        print(f"\n{'='*80}")
        print(f"✅ Finished!")
        print(f"{'='*80}")
        print(f"Total problems processed: {len(processed_problems)}")
        print(f"Output directory: {output_dir}")

        # 列出输出文件
        if os.path.exists(output_dir):
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

        # 统计失败代码
        if os.path.exists(failed_codes_dir):
            problem_dirs = [d for d in os.listdir(failed_codes_dir)
                            if os.path.isdir(os.path.join(failed_codes_dir, d))]
            total_failed = 0
            for problem_dir in problem_dirs:
                failed_count = len([f for f in os.listdir(os.path.join(failed_codes_dir, problem_dir))
                                   if f.endswith('.cpp')])
                total_failed += failed_count
            print(f"\nFailed codes directory: {failed_codes_dir}")
            print(f"Total failed codes: {total_failed} codes across {len(problem_dirs)} problems")

        print(f"{'='*80}")
