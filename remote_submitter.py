"""
Remote OJ Submitter - 基于HTTP API的远程评测器
支持批量提交和多线程并发评测
"""
import asyncio
import aiohttp
import time
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class RemoteOJSubmitter:
    def __init__(self, base_url: str = "http://localhost:8000", max_workers: int = None):
        """
        Args:
            base_url: OJ服务器地址
            max_workers: 最大并发数，默认为None（使用CPU核心数）
        """
        self.base_url = base_url
        self.submit_url = f"{base_url}/api/submit"
        self.query_url = f"{base_url}/api/submissions"
        self.max_workers = max_workers
        self.poll_interval = 0.5  # 轮询间隔(秒)

    async def submit_code_async(
        self,
        problem_id: str,
        code: str,
        language: str = "c++17"
    ) -> Dict:
        """
        异步提交代码并等待评测结果

        Args:
            problem_id: 题目ID
            code: 源代码
            language: 编程语言 (c++14/c++17/c++20/c++23)

        Returns:
            评测结果字典
        """
        async with aiohttp.ClientSession() as session:
            # 提交代码
            data = aiohttp.FormData()
            data.add_field('problem_id', problem_id)
            data.add_field('code', code)
            data.add_field('language', language)

            try:
                async with session.post(self.submit_url, data=data) as response:
                    result = await response.json()
                    submission_id = result.get('submission_id')

                    if not submission_id:
                        return {
                            "success": False,
                            "verdict": "System Error",
                            "message": "Failed to get submission_id",
                            "time": 0,
                            "memory": 0,
                            "passed": False,
                            "failed_test": None,
                        }

            except Exception as e:
                return {
                    "success": False,
                    "verdict": "System Error",
                    "message": f"Submit failed: {e}",
                    "time": 0,
                    "memory": 0,
                    "passed": False,
                    "failed_test": None,
                }

            # 轮询获取结果
            start_time = time.time()
            while True:
                try:
                    async with session.get(f"{self.query_url}/{submission_id}") as response:
                        result = await response.json()
                        status = result.get('status')

                        # 如果还在评测中，继续等待
                        if status in ['Pending', 'Judging']:
                            await asyncio.sleep(self.poll_interval)
                            continue

                        # 评测完成，转换为统一格式
                        elapsed = time.time() - start_time

                        return {
                            "success": True,
                            "verdict": status,
                            "message": result.get('message', ''),
                            "time": result.get('time_used', 0),
                            "memory": result.get('memory_used', 0),
                            "score": result.get('score', 0),
                            "passed": status == "Accepted",
                            "failed_test": result.get('failed_case', None),
                            "submission_id": submission_id,
                            "total_time": elapsed,
                        }

                except Exception as e:
                    return {
                        "success": False,
                        "verdict": "System Error",
                        "message": f"Query failed: {e}",
                        "time": 0,
                        "memory": 0,
                        "passed": False,
                        "failed_test": None,
                    }

    def submit_code(
        self,
        problem_id: str,
        code: str,
        problem_info: dict,
        all_judge: bool = True,
        language: str = None
    ) -> Dict:
        """
        同步提交代码（兼容原LocalCodeSubmitter接口）

        Args:
            problem_id: 题目ID
            code: 源代码
            problem_info: 题目信息字典
            all_judge: 参数保留（OJ服务器端控制）
            language: 编程语言，优先使用此参数，否则从problem_info获取，默认c++17

        Returns:
            评测结果字典
        """
        if language is None:
            language = problem_info.get("language", "c++17")

        # 运行异步函数
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.submit_code_async(problem_id, code, language)
            )
            return result
        finally:
            loop.close()

    def batch_submit_code(
        self,
        problem_id: str,
        batch_code: List[str],
        problem_info: dict,
        original_result: str = "correct",
        all_judge: bool = True,
        use_multithreading: bool = True,
        language: str = None
    ) -> dict:
        """
        批量提交代码，支持多线程并行处理

        Args:
            problem_id: 题目ID
            batch_code: 代码列表
            problem_info: 题目信息
            original_result: "correct" 或 "incorrect"
            all_judge: 是否判断所有测试用例（保留参数）
            use_multithreading: 是否使用多线程（默认True）
            language: 编程语言，优先使用此参数，否则从problem_info获取，默认c++17

        Returns:
            包含TPR/TNR和通过的提交列表的字典
        """
        code_cnt = len(batch_code)
        passed_cnt = 0
        error_cnt = 0
        passed_submissions = []

        if use_multithreading and code_cnt > 1:
            # 多线程处理
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_idx = {
                    executor.submit(
                        self.submit_code,
                        problem_id,
                        code,
                        problem_info,
                        all_judge,
                        language
                    ): (idx, code)
                    for idx, code in enumerate(batch_code)
                }

                # 使用tqdm显示进度
                with tqdm(total=code_cnt, desc=f"Submitting {problem_id}") as pbar:
                    for future in as_completed(future_to_idx):
                        idx, code = future_to_idx[future]
                        try:
                            result = future.result()

                            if result.get("verdict") == "Accepted":
                                passed_submissions.append({
                                    "index": idx,
                                    "code": code,
                                    "result": result
                                })

                            if not result.get("success"):
                                error_cnt += 1
                            elif result.get("verdict") == "Accepted":
                                passed_cnt += 1

                        except Exception as e:
                            print(f"Error processing code {idx}: {e}")
                            error_cnt += 1

                        pbar.update(1)
        else:
            # 顺序处理
            for idx, code in enumerate(tqdm(batch_code, desc=f"Submitting {problem_id}")):
                result = self.submit_code(problem_id, code, problem_info, all_judge, language)

                if result.get("verdict") == "Accepted":
                    passed_submissions.append({
                        "index": idx,
                        "code": code,
                        "result": result
                    })

                if not result.get("success"):
                    error_cnt += 1
                elif result.get("verdict") == "Accepted":
                    passed_cnt += 1

        valid_cnt = code_cnt - error_cnt
        assert valid_cnt > 0, f"No valid submissions for problem {problem_id}"

        if original_result == "correct":
            return {
                "TPR": passed_cnt / valid_cnt,
                "FNR": 1 - passed_cnt / valid_cnt,
                "passed_submissions": passed_submissions
            }
        else:
            return {
                "TNR": 1 - passed_cnt / valid_cnt,
                "FPR": passed_cnt / valid_cnt,
                "passed_submissions": passed_submissions
            }


# 示例使用
if __name__ == "__main__":
    # 初始化提交器
    submitter = RemoteOJSubmitter(
        base_url="http://localhost:8000",
        max_workers=8  # 最大并发数
    )

    # 读取代码文件
    code_file = Path("test.cpp")
    if code_file.exists():
        code = code_file.read_text(encoding='utf-8')

        # 单次提交示例
        problem_info = {
            "language": "c++23"
        }

        result = submitter.submit_code(
            problem_id="55_A",
            code=code,
            problem_info=problem_info
        )

        print(f"Verdict: {result['verdict']}")
        print(f"Time: {result['time']}ms")
        print(f"Score: {result.get('score', 0)}")
        print(f"Failed Test: {result.get('failed_test', 'N/A')}")

        # 批量提交示例
        # batch_codes = [code] * 10  # 提交10次相同代码
        # batch_result = submitter.batch_submit_code(
        #     problem_id="55_A",
        #     batch_code=batch_codes,
        #     problem_info=problem_info,
        #     original_result="correct"
        # )
        # print(f"TPR: {batch_result['TPR']}")
        # print(f"Passed: {len(batch_result['passed_submissions'])}/{len(batch_codes)}")
    else:
        print(f"Error: {code_file} not found")
