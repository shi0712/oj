import asyncio
import subprocess
import os
import tempfile
import shutil
import hashlib
import gc
from pathlib import Path
from typing import Optional
import time

from config import COMPILERS, PROBLEMS_DIR, TESTLIB_DIR, DATA_DIR
from models import JudgeStatus

# 最大输出大小 (10MB)
MAX_OUTPUT_SIZE = 10 * 1024 * 1024

# 编译缓存目录
EXE_CACHE_DIR = DATA_DIR / "exe_cache"
EXE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 缓存清理计数器（每100次评测清理一次）
_judge_count = 0
_judge_count_lock = asyncio.Lock()

async def _cleanup_cache_if_needed():
    """定期清理exe缓存，避免累积过多"""
    global _judge_count
    async with _judge_count_lock:
        _judge_count += 1
        if _judge_count >= 100:
            _judge_count = 0
            try:
                # 只清理.tmp文件和超过1小时未使用的.exe文件
                import time as time_module
                now = time_module.time()
                cleaned_tmp = 0
                cleaned_old = 0

                for file in EXE_CACHE_DIR.iterdir():
                    if file.suffix == '.tmp':
                        try:
                            file.unlink()
                            cleaned_tmp += 1
                        except:
                            pass
                    elif file.suffix == '.exe':
                        # 删除超过1小时未访问的exe
                        try:
                            if now - file.stat().st_atime > 3600:
                                file.unlink()
                                cleaned_old += 1
                        except:
                            pass

                if cleaned_tmp > 0 or cleaned_old > 0:
                    print(f"[Cache] Cleaned {cleaned_tmp} tmp files, {cleaned_old} old exe files")
                    gc.collect()
            except Exception as e:
                print(f"[Cache] Cleanup failed: {e}")

class JudgeResult:
    def __init__(self, status: JudgeStatus, time_used: int = 0, memory_used: int = 0,
                 score: int = 0, message: str = "", failed_case: int = 0):
        self.status = status
        self.time_used = time_used
        self.memory_used = memory_used
        self.score = score
        self.message = message
        self.failed_case = failed_case

class Judge:
    def __init__(self, problem_id: str, code: str, language: str, submission_id: int = 0):
        self.problem_id = problem_id
        self.code = code
        self.language = language
        self.submission_id = submission_id
        self.problem_dir = PROBLEMS_DIR / problem_id
        self.work_dir: Optional[Path] = None

    async def run(self, time_limit: int, memory_limit: int, has_checker: bool) -> JudgeResult:
        # Periodic cache cleanup
        await _cleanup_cache_if_needed()

        # Create temp working directory
        self.work_dir = Path(tempfile.mkdtemp(prefix="oj_judge_"))
        print(f"[Judge #{self.submission_id}] Problem: {self.problem_id}, Language: {self.language}")

        try:
            # Compile source code
            print(f"[Judge #{self.submission_id}] Compiling...")
            compile_result = await self._compile()
            if compile_result.status != JudgeStatus.ACCEPTED:
                print(f"[Judge #{self.submission_id}] Compile Error: {compile_result.message[:200]}")
                return compile_result

            # Compile checker if needed
            if has_checker:
                checker_result = await self._compile_checker()
                if checker_result.status != JudgeStatus.ACCEPTED:
                    return JudgeResult(JudgeStatus.SYSTEM_ERROR, message="Checker compile error")

            # Run test cases
            return await self._run_tests(time_limit, memory_limit, has_checker)
        finally:
            # Cleanup
            if self.work_dir and self.work_dir.exists():
                shutil.rmtree(self.work_dir, ignore_errors=True)
            # 手动触发垃圾回收
            gc.collect()

    async def _compile(self) -> JudgeResult:
        if self.language not in COMPILERS:
            return JudgeResult(JudgeStatus.SYSTEM_ERROR, message=f"Unknown language: {self.language}")

        compiler = COMPILERS[self.language]

        # 计算代码哈希，用于缓存
        code_hash = hashlib.md5((self.code + self.language).encode()).hexdigest()
        cached_exe = EXE_CACHE_DIR / f"{code_hash}.exe"
        self.cached_exe_path = cached_exe  # 保存缓存路径，供评测时直接使用

        # 如果缓存存在，直接使用（不复制）
        if cached_exe.exists():
            print(f"[Judge #{self.submission_id}] Using cached exe: {code_hash[:8]}...")
            return JudgeResult(JudgeStatus.ACCEPTED)

        source_file = self.work_dir / "main.cpp"
        exe_file = self.work_dir / "main.exe"

        # Write source code
        source_file.write_text(self.code, encoding="utf-8")

        # Compile
        libs = compiler.get("libs", [])
        cmd = [compiler["path"]] + compiler["args"] + [str(source_file), "-o", str(exe_file)] + libs
        print(f"[Judge #{self.submission_id}] Compile command: {' '.join(cmd)}")

        # Set PATH to include compiler's bin directory for MSYS2 compatibility
        compiler_bin_dir = str(Path(compiler["path"]).parent)
        env = os.environ.copy()
        env["PATH"] = compiler_bin_dir + os.pathsep + env.get("PATH", "")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.work_dir),
                env=env
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)

            if process.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace")
                return JudgeResult(JudgeStatus.COMPILE_ERROR, message=error_msg[:2000])

            # 缓存编译好的 exe（使用原子操作避免竞争）
            # 先复制到临时文件，再重命名
            temp_exe = EXE_CACHE_DIR / f"{code_hash}_{self.submission_id}.tmp"
            try:
                shutil.copy(exe_file, temp_exe)
                # 原子重命名（如果目标已存在会覆盖，这是安全的）
                temp_exe.replace(cached_exe)
                print(f"[Judge #{self.submission_id}] Cached exe: {code_hash[:8]}...")
            except Exception as e:
                # 缓存失败不影响评测，继续使用本地exe
                print(f"[Judge #{self.submission_id}] Cache failed: {e}")
                self.cached_exe_path = exe_file
            finally:
                # 确保临时文件被删除（即使重命名失败）
                if temp_exe.exists():
                    try:
                        temp_exe.unlink()
                    except:
                        pass

            return JudgeResult(JudgeStatus.ACCEPTED)
        except asyncio.TimeoutError:
            return JudgeResult(JudgeStatus.COMPILE_ERROR, message="Compilation timeout")
        except Exception as e:
            return JudgeResult(JudgeStatus.SYSTEM_ERROR, message=str(e))

    async def _compile_checker(self) -> JudgeResult:
        checker_src = self.problem_dir / "checker.cpp"
        checker_exe = self.problem_dir / "checker.exe"  # Cache in problem dir

        if not checker_src.exists():
            return JudgeResult(JudgeStatus.SYSTEM_ERROR, message="Checker not found")

        # Check if cached exe exists and is newer than source
        if checker_exe.exists():
            src_mtime = checker_src.stat().st_mtime
            exe_mtime = checker_exe.stat().st_mtime
            if exe_mtime > src_mtime:
                print(f"[Judge #{self.submission_id}] Using cached checker")
                return JudgeResult(JudgeStatus.ACCEPTED)

        print(f"[Judge #{self.submission_id}] Compiling checker...")

        # Use C++20 for checker
        compiler = COMPILERS.get("c++20")
        libs = compiler.get("libs", [])
        cmd = [compiler["path"]] + compiler["args"] + [
            f"-I{TESTLIB_DIR}",
            str(checker_src),
            "-o", str(checker_exe)
        ]

        # Set PATH for MSYS2 compatibility
        compiler_bin_dir = str(Path(compiler["path"]).parent)
        env = os.environ.copy()
        env["PATH"] = compiler_bin_dir + os.pathsep + env.get("PATH", "")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=30)

            if process.returncode != 0:
                return JudgeResult(JudgeStatus.SYSTEM_ERROR,
                                   message=f"Checker compile error: {stderr.decode()[:500]}")

            return JudgeResult(JudgeStatus.ACCEPTED)
        except Exception as e:
            return JudgeResult(JudgeStatus.SYSTEM_ERROR, message=str(e))

    async def _run_tests(self, time_limit: int, memory_limit: int, has_checker: bool) -> JudgeResult:
        test_dir = self.problem_dir / "testcases"
        if not test_dir.exists():
            return JudgeResult(JudgeStatus.SYSTEM_ERROR, message="Test cases not found")

        # Get all test cases
        input_files = sorted(test_dir.glob("*.in"), key=lambda p: int(p.stem))
        if not input_files:
            return JudgeResult(JudgeStatus.SYSTEM_ERROR, message="No test cases")

        total_time = 0
        max_memory = 0
        passed = 0

        # 预热：运行第一个测试点预热程序（解决 C++17 冷启动问题）
        if len(input_files) > 0:
            print(f"[Judge #{self.submission_id}] Warming up (pre-run first test)...")
            first_input = input_files[0]
            await self._warmup_run(first_input)

        for idx, input_file in enumerate(input_files, 1):
            output_file = input_file.with_suffix(".out")
            if not output_file.exists():
                continue
            print(input_file, output_file)
            result = await self._run_single_test(
                input_file, output_file, time_limit, memory_limit, has_checker
            )

            # TLE/SE自动重试（TLE最多3次，SE最多10次）
            if result.status == JudgeStatus.TIME_LIMIT:
                for retry in range(2):
                    print(f"[Judge #{self.submission_id}] Test {idx} TLE, retry {retry+1}/3...")
                    await asyncio.sleep(1)
                    result = await self._run_single_test(
                        input_file, output_file, time_limit, memory_limit, has_checker
                    )
                    if result.status != JudgeStatus.TIME_LIMIT:
                        break
            elif result.status == JudgeStatus.SYSTEM_ERROR:
                for retry in range(2):
                    print(f"[Judge #{self.submission_id}] Test {idx} SE, retry {retry+1}/10...")
                    await asyncio.sleep(1)
                    result = await self._run_single_test(
                        input_file, output_file, time_limit, memory_limit, has_checker
                    )
                    if result.status != JudgeStatus.SYSTEM_ERROR:
                        break

            if result.status != JudgeStatus.ACCEPTED:
                result.score = int(passed * 100 / len(input_files))
                result.failed_case = idx
                return result

            total_time = max(total_time, result.time_used)
            max_memory = max(max_memory, result.memory_used)
            passed += 1

        return JudgeResult(
            JudgeStatus.ACCEPTED,
            time_used=total_time,
            memory_used=max_memory,
            score=100,
            message=f"Passed {passed}/{len(input_files)} test cases"
        )

    async def _warmup_run(self, input_file: Path):
        """预热运行：执行一次程序但不计入结果（解决冷启动）"""
        exe_file = getattr(self, 'cached_exe_path', self.work_dir / "main.exe")
        print(f"[Judge #{self.submission_id}] Warmup: running {exe_file}")

        # 重试最多10次（处理System Error等异常情况）
        for retry in range(2):
            start_time = time.perf_counter()

            try:
                with open(input_file, "rb") as fin:
                    process = await asyncio.create_subprocess_exec(
                        str(exe_file),
                        stdin=fin,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                        cwd=str(self.work_dir)
                    )
                    try:
                        await asyncio.wait_for(process.communicate(), timeout=5)
                        elapsed = (time.perf_counter() - start_time) * 1000
                        print(f"[Judge #{self.submission_id}] Warmup completed in {elapsed:.0f}ms")
                        return  # 成功，直接返回
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
                        print(f"[Judge #{self.submission_id}] Warmup timed out (5s)")
                        return  # 超时也算成功（程序能运行）
            except Exception as e:
                if retry < 9:
                    print(f"[Judge #{self.submission_id}] Warmup failed (retry {retry+1}/10): {e}")
                    await asyncio.sleep(1)  # 等待1秒后重试
                else:
                    print(f"[Judge #{self.submission_id}] Warmup failed after 10 retries: {e}")

    async def _run_single_test(self, input_file: Path, expected_output: Path,
                                time_limit: int, memory_limit: int, has_checker: bool) -> JudgeResult:
        exe_file = getattr(self, 'cached_exe_path', self.work_dir / "main.exe")
        user_output = self.work_dir / "user.out"

        # Debug logging for test case 33
        test_num = int(input_file.stem)
        # if test_num == 33:
        #     print(f"[Judge #{self.submission_id}] === Debug test 33 ===")
        #     print(f"  exe_file: {exe_file}, exists: {exe_file.exists()}")
        #     print(f"  input_file: {input_file}, exists: {input_file.exists()}, size: {input_file.stat().st_size if input_file.exists() else 0}")
        #     print(f"  expected_output: {expected_output}, exists: {expected_output.exists()}")
        #     print(f"  work_dir: {self.work_dir}, exists: {self.work_dir.exists()}")

        try:
            # Run program
            with open(input_file, "rb") as fin, open(user_output, "wb") as fout:
                start_time = time.perf_counter()

                process = await asyncio.create_subprocess_exec(
                    str(exe_file),
                    stdin=fin,
                    stdout=fout,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(self.work_dir)
                )

                try:
                    _, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=time_limit / 1000.0 + 0.5
                    )
                    # 清理 stderr 缓冲区
                    del stderr
                except asyncio.TimeoutError:
                    # 尝试终止进程，如果进程已经退出则忽略错误
                    try:
                        process.kill()
                        await process.wait()
                    except ProcessLookupError:
                        pass  # 进程已经退出
                    return JudgeResult(JudgeStatus.TIME_LIMIT, time_used=time_limit)

                elapsed_ms = int((time.perf_counter() - start_time) * 1000)

                if process.returncode != 0:
                    return JudgeResult(JudgeStatus.RUNTIME_ERROR,
                                       time_used=elapsed_ms,
                                       message=f"Exit code: {process.returncode}")

                if elapsed_ms > time_limit:
                    return JudgeResult(JudgeStatus.TIME_LIMIT, time_used=elapsed_ms)

            # Check output size
            if user_output.exists():
                output_size = user_output.stat().st_size
                if output_size > MAX_OUTPUT_SIZE:
                    return JudgeResult(JudgeStatus.RUNTIME_ERROR,
                                       time_used=elapsed_ms,
                                       message=f"Output too large: {output_size} bytes (limit: {MAX_OUTPUT_SIZE})")

            # Check output
            if has_checker:
                check_result = await self._run_checker(input_file, user_output, expected_output)
                check_result.time_used = elapsed_ms
                return check_result
            else:
                # Simple diff comparison
                if self._compare_output(user_output, expected_output):
                    return JudgeResult(JudgeStatus.ACCEPTED, time_used=elapsed_ms)
                else:
                    return JudgeResult(JudgeStatus.WRONG_ANSWER, time_used=elapsed_ms)

        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}"
            traceback_str = traceback.format_exc()
            print(f"[Judge #{self.submission_id}] SYSTEM_ERROR in _run_single_test:")
            print(traceback_str)
            return JudgeResult(JudgeStatus.SYSTEM_ERROR, message=error_msg)

    async def _run_checker(self, input_file: Path, user_output: Path, expected_output: Path) -> JudgeResult:
        checker_exe = self.problem_dir / "checker.exe"  # Use cached checker

        try:
            import time
            start_time = time.perf_counter()

            process = await asyncio.create_subprocess_exec(
                str(checker_exe),
                str(input_file),
                str(user_output),
                str(expected_output),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.work_dir)
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)

            elapsed = time.perf_counter() - start_time

            # 记录checker运行时间超过10秒的题目
            if elapsed > 10:
                self._record_slow_checker(input_file, elapsed)

            # testlib.h exit codes: 0=AC, 1=WA, 2=PE, 3=Fail
            if process.returncode == 0:
                return JudgeResult(JudgeStatus.ACCEPTED)
            else:
                msg = stderr.decode("utf-8", errors="replace") or stdout.decode("utf-8", errors="replace")
                result = JudgeResult(JudgeStatus.WRONG_ANSWER, message=msg[:500])
                # 清理大的输出缓冲区
                del stdout, stderr, msg
                return result

        except asyncio.TimeoutError:
            # Checker超时，记录问题
            self._record_slow_checker(input_file, timeout=True)

            print(f"[Judge #{self.submission_id}] Checker timeout after 10s on {input_file.stem}")
            try:
                process.kill()
                await process.wait()
            except:
                pass
            return JudgeResult(JudgeStatus.SYSTEM_ERROR, message=f"Checker timeout on test {input_file.stem}")
        except Exception as e:
            import traceback
            print(f"[Judge #{self.submission_id}] Checker exception:")
            print(traceback.format_exc())
            return JudgeResult(JudgeStatus.SYSTEM_ERROR, message=str(e))

    def _record_slow_checker(self, input_file: Path, elapsed: float = None, timeout: bool = False):
        """记录checker性能问题的题目"""
        import json
        from datetime import datetime

        record_file = DATA_DIR / "problematic_problems.json"

        # 读取现有记录
        if record_file.exists():
            try:
                with open(record_file, 'r', encoding='utf-8') as f:
                    records = json.load(f)
            except:
                records = {}
        else:
            records = {}

        problem_id = self.problem_id
        test_case = input_file.stem

        if problem_id not in records:
            records[problem_id] = {
                "reason": "slow_checker",
                "test_cases": {},
                "first_seen": datetime.now().isoformat()
            }

        if timeout:
            records[problem_id]["test_cases"][test_case] = {
                "status": "timeout",
                "time": ">10s",
                "last_seen": datetime.now().isoformat()
            }
        else:
            records[problem_id]["test_cases"][test_case] = {
                "status": "slow",
                "time": f"{elapsed:.2f}s",
                "last_seen": datetime.now().isoformat()
            }

        # 写入文件
        try:
            with open(record_file, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Judge #{self.submission_id}] Failed to record slow checker: {e}")

    def _compare_output(self, user_output: Path, expected_output: Path) -> bool:
        """Compare outputs ignoring trailing whitespace and blank lines"""
        try:
            with open(user_output, "r", encoding="utf-8", errors="replace") as f:
                user_lines = [line.rstrip() for line in f.readlines()]
            with open(expected_output, "r", encoding="utf-8", errors="replace") as f:
                expected_lines = [line.rstrip() for line in f.readlines()]

            # Remove trailing empty lines
            while user_lines and not user_lines[-1]:
                user_lines.pop()
            while expected_lines and not expected_lines[-1]:
                expected_lines.pop()

            result = user_lines == expected_lines

            # 清理大列表
            del user_lines, expected_lines

            return result
        except:
            return False

    async def hack(self, input_data: str, std_code: str, time_limit: int,
                   memory_limit: int, has_checker: bool) -> JudgeResult:
        """
        Hack: test code with custom input and std code.
        Returns user output and comparison result.
        """
        self.work_dir = Path(tempfile.mkdtemp(prefix="oj_hack_"))
        print(f"[Hack] Problem: {self.problem_id}, Language: {self.language}")

        try:
            # Compile user code
            print("[Hack] Compiling user code...")
            compile_result = await self._compile()
            if compile_result.status != JudgeStatus.ACCEPTED:
                return JudgeResult(JudgeStatus.COMPILE_ERROR, message=compile_result.message)

            user_exe = self.work_dir / "main.exe"

            # Compile std code if provided
            std_exe = None
            if std_code:
                print("[Hack] Compiling std code...")
                std_source = self.work_dir / "std.cpp"
                std_exe = self.work_dir / "std.exe"
                std_source.write_text(std_code, encoding="utf-8")

                # Always use C++23 for std code
                compiler = COMPILERS.get("c++23", COMPILERS.get("c++17", COMPILERS["c++14"]))
                libs = compiler.get("libs", [])
                cmd = [compiler["path"]] + compiler["args"] + [str(std_source), "-o", str(std_exe)] + libs

                compiler_bin_dir = str(Path(compiler["path"]).parent)
                env = os.environ.copy()
                env["PATH"] = compiler_bin_dir + os.pathsep + env.get("PATH", "")

                process = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    cwd=str(self.work_dir), env=env
                )
                _, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
                if process.returncode != 0:
                    return JudgeResult(JudgeStatus.SYSTEM_ERROR,
                                       message=f"Std compile error: {stderr.decode()[:500]}")

            # Compile checker if needed
            if has_checker:
                checker_result = await self._compile_checker()
                if checker_result.status != JudgeStatus.ACCEPTED:
                    return JudgeResult(JudgeStatus.SYSTEM_ERROR, message="Checker compile error")

            # Write input
            input_file = self.work_dir / "input.in"
            input_file.write_text(input_data, encoding="utf-8")

            # Run user code
            print("[Hack] Running user code...")
            user_output_file = self.work_dir / "user.out"
            user_result = await self._run_program(user_exe, input_file, user_output_file, time_limit)

            if user_result.status != JudgeStatus.ACCEPTED:
                user_result.message = f"User: {user_result.status.value}\n{user_result.message}"
                return user_result

            user_output = user_output_file.read_text(encoding="utf-8", errors="replace")

            # Run std code if provided
            expected_output_file = self.work_dir / "expected.out"
            if std_exe:
                print("[Hack] Running std code...")
                std_result = await self._run_program(std_exe, input_file, expected_output_file, time_limit)
                if std_result.status != JudgeStatus.ACCEPTED:
                    return JudgeResult(JudgeStatus.SYSTEM_ERROR,
                                       message=f"Std: {std_result.status.value}")

            # Compare results
            if has_checker and expected_output_file.exists():
                check_result = await self._run_checker(input_file, user_output_file, expected_output_file)
                check_result.message = f"User Output:\n{user_output[:2000]}\n\nChecker: {check_result.message}"
                check_result.time_used = user_result.time_used
                return check_result
            elif expected_output_file.exists():
                expected_output = expected_output_file.read_text(encoding="utf-8", errors="replace")
                if self._compare_output(user_output_file, expected_output_file):
                    return JudgeResult(JudgeStatus.ACCEPTED, time_used=user_result.time_used,
                                       message=f"User Output:\n{user_output[:2000]}")
                else:
                    return JudgeResult(JudgeStatus.WRONG_ANSWER, time_used=user_result.time_used,
                                       message=f"User Output:\n{user_output[:1000]}\n\nExpected:\n{expected_output[:1000]}")
            else:
                # No std, just return output
                return JudgeResult(JudgeStatus.ACCEPTED, time_used=user_result.time_used,
                                   message=f"Output:\n{user_output[:2000]}")

        finally:
            if self.work_dir and self.work_dir.exists():
                shutil.rmtree(self.work_dir, ignore_errors=True)
            # 手动触发垃圾回收
            gc.collect()

    async def _run_program(self, exe_file: Path, input_file: Path, output_file: Path,
                           time_limit: int) -> JudgeResult:
        """Run a program with input and capture output"""
        try:
            with open(input_file, "rb") as fin, open(output_file, "wb") as fout:
                start_time = time.perf_counter()
                process = await asyncio.create_subprocess_exec(
                    str(exe_file),
                    stdin=fin, stdout=fout, stderr=asyncio.subprocess.PIPE,
                    cwd=str(self.work_dir)
                )
                try:
                    _, stderr = await asyncio.wait_for(
                        process.communicate(), timeout=time_limit / 1000.0 + 0.5
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    return JudgeResult(JudgeStatus.TIME_LIMIT, time_used=time_limit)

                elapsed_ms = int((time.perf_counter() - start_time) * 1000)

                if process.returncode != 0:
                    return JudgeResult(JudgeStatus.RUNTIME_ERROR, time_used=elapsed_ms,
                                       message=f"Exit code: {process.returncode}")

                if elapsed_ms > time_limit:
                    return JudgeResult(JudgeStatus.TIME_LIMIT, time_used=elapsed_ms)

            # Check output size
            if output_file.exists():
                output_size = output_file.stat().st_size
                if output_size > MAX_OUTPUT_SIZE:
                    return JudgeResult(JudgeStatus.RUNTIME_ERROR, time_used=elapsed_ms,
                                       message=f"Output too large: {output_size} bytes")

                return JudgeResult(JudgeStatus.ACCEPTED, time_used=elapsed_ms)
        except Exception as e:
            return JudgeResult(JudgeStatus.SYSTEM_ERROR, message=str(e))
