import asyncio
import subprocess
import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional
import time

from config import COMPILERS, PROBLEMS_DIR, TESTLIB_DIR
from models import JudgeStatus

class JudgeResult:
    def __init__(self, status: JudgeStatus, time_used: int = 0, memory_used: int = 0,
                 score: int = 0, message: str = ""):
        self.status = status
        self.time_used = time_used
        self.memory_used = memory_used
        self.score = score
        self.message = message

class Judge:
    def __init__(self, problem_id: str, code: str, language: str, submission_id: int = 0):
        self.problem_id = problem_id
        self.code = code
        self.language = language
        self.submission_id = submission_id
        self.problem_dir = PROBLEMS_DIR / problem_id
        self.work_dir: Optional[Path] = None

    async def run(self, time_limit: int, memory_limit: int, has_checker: bool) -> JudgeResult:
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

    async def _compile(self) -> JudgeResult:
        if self.language not in COMPILERS:
            return JudgeResult(JudgeStatus.SYSTEM_ERROR, message=f"Unknown language: {self.language}")

        compiler = COMPILERS[self.language]
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

        # Use C++23 for checker
        compiler = COMPILERS.get("c++23", COMPILERS.get("c++17", COMPILERS["c++14"]))
        libs = compiler.get("libs", [])
        cmd = [compiler["path"]] + compiler["args"] + [
            f"-I{TESTLIB_DIR}",
            str(checker_src),
            "-o", str(checker_exe)
        ] + libs

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
        input_files = sorted(test_dir.glob("*.in"))
        if not input_files:
            return JudgeResult(JudgeStatus.SYSTEM_ERROR, message="No test cases")

        total_time = 0
        max_memory = 0
        passed = 0

        for input_file in input_files:
            output_file = input_file.with_suffix(".out")
            if not output_file.exists():
                continue

            result = await self._run_single_test(
                input_file, output_file, time_limit, memory_limit, has_checker
            )

            if result.status != JudgeStatus.ACCEPTED:
                result.score = int(passed * 100 / len(input_files))
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

    async def _run_single_test(self, input_file: Path, expected_output: Path,
                                time_limit: int, memory_limit: int, has_checker: bool) -> JudgeResult:
        exe_file = self.work_dir / "main.exe"
        user_output = self.work_dir / "user.out"

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
                except asyncio.TimeoutError:
                    process.kill()
                    return JudgeResult(JudgeStatus.TIME_LIMIT, time_used=time_limit)

                elapsed_ms = int((time.perf_counter() - start_time) * 1000)

                if process.returncode != 0:
                    return JudgeResult(JudgeStatus.RUNTIME_ERROR,
                                       time_used=elapsed_ms,
                                       message=f"Exit code: {process.returncode}")

                if elapsed_ms > time_limit:
                    return JudgeResult(JudgeStatus.TIME_LIMIT, time_used=elapsed_ms)

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
            return JudgeResult(JudgeStatus.SYSTEM_ERROR, message=str(e))

    async def _run_checker(self, input_file: Path, user_output: Path, expected_output: Path) -> JudgeResult:
        checker_exe = self.problem_dir / "checker.exe"  # Use cached checker

        try:
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

            # testlib.h exit codes: 0=AC, 1=WA, 2=PE, 3=Fail
            if process.returncode == 0:
                return JudgeResult(JudgeStatus.ACCEPTED)
            else:
                msg = stderr.decode("utf-8", errors="replace") or stdout.decode("utf-8", errors="replace")
                return JudgeResult(JudgeStatus.WRONG_ANSWER, message=msg[:500])

        except Exception as e:
            return JudgeResult(JudgeStatus.SYSTEM_ERROR, message=str(e))

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

            return user_lines == expected_lines
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

                return JudgeResult(JudgeStatus.ACCEPTED, time_used=elapsed_ms)
        except Exception as e:
            return JudgeResult(JudgeStatus.SYSTEM_ERROR, message=str(e))
