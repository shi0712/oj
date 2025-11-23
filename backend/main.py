import asyncio
from asyncio import Semaphore
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional
import shutil
import zipfile
import os
from pathlib import Path

from config import PROBLEMS_DIR, MAX_CONCURRENT_JUDGES, COMPILERS
from models import init_db, get_session, Problem, Submission, JudgeStatus
from judge import Judge

app = FastAPI(title="Online Judge")

# Semaphore for concurrent judge limit
judge_semaphore = Semaphore(MAX_CONCURRENT_JUDGES)

@app.on_event("startup")
async def startup():
    await init_db()

# ===== Problem APIs =====

@app.post("/api/problems")
async def create_problem(
    problem_id: str = Form(...),
    title: str = Form(""),
    time_limit: int = Form(1000),
    memory_limit: int = Form(256),
    testcases: UploadFile = File(...),
    checker: Optional[UploadFile] = File(None),
    session: AsyncSession = Depends(get_session)
):
    """Upload a problem with test cases and optional SPJ checker"""
    problem_dir = PROBLEMS_DIR / problem_id
    testcase_dir = problem_dir / "testcases"

    # Create directories
    problem_dir.mkdir(parents=True, exist_ok=True)
    testcase_dir.mkdir(exist_ok=True)

    # Extract test cases (zip file with .in and .out files)
    try:
        temp_zip = problem_dir / "temp.zip"
        with open(temp_zip, "wb") as f:
            shutil.copyfileobj(testcases.file, f)

        with zipfile.ZipFile(temp_zip, "r") as zf:
            for name in zf.namelist():
                if name.endswith(".in") or name.endswith(".out"):
                    # Extract to testcase directory with flat structure
                    basename = os.path.basename(name)
                    if basename:
                        with zf.open(name) as src, open(testcase_dir / basename, "wb") as dst:
                            dst.write(src.read())

        temp_zip.unlink()

        # Count test cases
        test_count = len(list(testcase_dir.glob("*.in")))

    except Exception as e:
        raise HTTPException(400, f"Failed to extract test cases: {e}")

    # Save checker if provided
    has_checker = False
    if checker:
        checker_path = problem_dir / "checker.cpp"
        with open(checker_path, "wb") as f:
            shutil.copyfileobj(checker.file, f)
        has_checker = True

    # Save to database
    problem = Problem(
        id=problem_id,
        title=title,
        time_limit=time_limit,
        memory_limit=memory_limit,
        has_checker=has_checker,
        test_case_count=test_count
    )

    # Upsert
    existing = await session.get(Problem, problem_id)
    if existing:
        existing.title = title
        existing.time_limit = time_limit
        existing.memory_limit = memory_limit
        existing.has_checker = has_checker
        existing.test_case_count = test_count
    else:
        session.add(problem)

    await session.commit()

    return {
        "success": True,
        "problem_id": problem_id,
        "test_case_count": test_count,
        "has_checker": has_checker
    }

@app.get("/api/problems")
async def list_problems(session: AsyncSession = Depends(get_session)):
    """List all problems"""
    result = await session.execute(select(Problem).order_by(Problem.id))
    problems = result.scalars().all()
    return [
        {
            "id": p.id,
            "title": p.title,
            "time_limit": p.time_limit,
            "memory_limit": p.memory_limit,
            "test_case_count": p.test_case_count,
            "has_checker": p.has_checker
        }
        for p in problems
    ]

@app.get("/api/problems/{problem_id}")
async def get_problem(problem_id: str, session: AsyncSession = Depends(get_session)):
    """Get problem details"""
    problem = await session.get(Problem, problem_id)
    if not problem:
        raise HTTPException(404, "Problem not found")

    return {
        "id": problem.id,
        "title": problem.title,
        "time_limit": problem.time_limit,
        "memory_limit": problem.memory_limit,
        "test_case_count": problem.test_case_count,
        "has_checker": problem.has_checker
    }

@app.patch("/api/problems/{problem_id}")
async def update_problem(
    problem_id: str,
    title: Optional[str] = Form(None),
    time_limit: Optional[int] = Form(None),
    memory_limit: Optional[int] = Form(None),
    testcases: Optional[UploadFile] = File(None),
    checker: Optional[UploadFile] = File(None),
    session: AsyncSession = Depends(get_session)
):
    """Update problem settings (title, time_limit, memory_limit, testcases, checker)"""
    problem = await session.get(Problem, problem_id)
    if not problem:
        raise HTTPException(404, "Problem not found")

    if title is not None:
        problem.title = title
    if time_limit is not None:
        problem.time_limit = time_limit
    if memory_limit is not None:
        problem.memory_limit = memory_limit

    problem_dir = PROBLEMS_DIR / problem_id
    testcase_dir = problem_dir / "testcases"

    # Update test cases if provided
    if testcases:
        if testcase_dir.exists():
            shutil.rmtree(testcase_dir)
        testcase_dir.mkdir(parents=True, exist_ok=True)

        try:
            temp_zip = problem_dir / "temp.zip"
            with open(temp_zip, "wb") as f:
                shutil.copyfileobj(testcases.file, f)

            with zipfile.ZipFile(temp_zip, "r") as zf:
                for name in zf.namelist():
                    if name.endswith(".in") or name.endswith(".out"):
                        basename = os.path.basename(name)
                        if basename:
                            with zf.open(name) as src, open(testcase_dir / basename, "wb") as dst:
                                dst.write(src.read())

            temp_zip.unlink()
            problem.test_case_count = len(list(testcase_dir.glob("*.in")))
        except Exception as e:
            raise HTTPException(400, f"Failed to extract test cases: {e}")

    # Update checker if provided
    if checker:
        checker_path = problem_dir / "checker.cpp"
        with open(checker_path, "wb") as f:
            shutil.copyfileobj(checker.file, f)
        problem.has_checker = True

    await session.commit()

    return {
        "success": True,
        "problem_id": problem_id,
        "title": problem.title,
        "time_limit": problem.time_limit,
        "memory_limit": problem.memory_limit,
        "test_case_count": problem.test_case_count,
        "has_checker": problem.has_checker
    }

@app.delete("/api/problems/{problem_id}")
async def delete_problem(problem_id: str, session: AsyncSession = Depends(get_session)):
    """Delete a problem"""
    problem = await session.get(Problem, problem_id)
    if problem:
        await session.delete(problem)
        await session.commit()

    # Remove files
    problem_dir = PROBLEMS_DIR / problem_id
    if problem_dir.exists():
        shutil.rmtree(problem_dir)

    return {"success": True}

# ===== Submission APIs =====

@app.post("/api/submit")
async def submit(
    problem_id: str = Form(...),
    code: str = Form(...),
    language: str = Form(...),
    background_tasks: BackgroundTasks = None,
    session: AsyncSession = Depends(get_session)
):
    """Submit code for judging"""
    # Validate problem
    problem = await session.get(Problem, problem_id)
    if not problem:
        raise HTTPException(404, "Problem not found")

    # Validate language
    if language not in COMPILERS:
        raise HTTPException(400, f"Unsupported language. Available: {list(COMPILERS.keys())}")

    # Create submission
    submission = Submission(
        problem_id=problem_id,
        code=code,
        language=language,
        status=JudgeStatus.PENDING.value
    )
    session.add(submission)
    await session.commit()
    await session.refresh(submission)

    # Start judging in background
    background_tasks.add_task(
        judge_submission,
        submission.id,
        problem_id,
        code,
        language,
        problem.time_limit,
        problem.memory_limit,
        problem.has_checker
    )

    return {"submission_id": submission.id, "status": "Pending"}

async def judge_submission(
    submission_id: int,
    problem_id: str,
    code: str,
    language: str,
    time_limit: int,
    memory_limit: int,
    has_checker: bool
):
    """Background task to judge a submission"""
    async with judge_semaphore:
        async with async_session() as session:
            # Update status to judging
            submission = await session.get(Submission, submission_id)
            submission.status = JudgeStatus.JUDGING.value
            await session.commit()

            # Run judge
            judge = Judge(problem_id, code, language, submission_id)
            result = await judge.run(time_limit, memory_limit, has_checker)

            # Update result
            submission.status = result.status.value
            submission.time_used = result.time_used
            submission.memory_used = result.memory_used
            submission.score = result.score
            submission.message = result.message
            await session.commit()
            print(f"[Judge #{submission_id}] Result: {result.status.value}, Time: {result.time_used}ms, Score: {result.score}")

# Import async_session for background task
from models import async_session

@app.get("/api/submissions/{submission_id}")
async def get_submission(submission_id: int, session: AsyncSession = Depends(get_session)):
    """Get submission status and result"""
    submission = await session.get(Submission, submission_id)
    if not submission:
        raise HTTPException(404, "Submission not found")

    return {
        "id": submission.id,
        "problem_id": submission.problem_id,
        "language": submission.language,
        "status": submission.status,
        "time_used": submission.time_used,
        "memory_used": submission.memory_used,
        "score": submission.score,
        "message": submission.message,
        "created_at": submission.created_at.isoformat()
    }

@app.get("/api/submissions")
async def list_submissions(
    problem_id: Optional[str] = None,
    limit: int = 50,
    session: AsyncSession = Depends(get_session)
):
    """List recent submissions"""
    query = select(Submission).order_by(Submission.id.desc()).limit(limit)
    if problem_id:
        query = query.where(Submission.problem_id == problem_id)

    result = await session.execute(query)
    submissions = result.scalars().all()

    return [
        {
            "id": s.id,
            "problem_id": s.problem_id,
            "language": s.language,
            "status": s.status,
            "time_used": s.time_used,
            "score": s.score,
            "created_at": s.created_at.isoformat()
        }
        for s in submissions
    ]

# ===== Config APIs =====

@app.get("/api/languages")
async def get_languages():
    """Get available languages and compilers"""
    return {
        lang: {"bits": cfg["bits"], "args": cfg["args"]}
        for lang, cfg in COMPILERS.items()
    }

@app.post("/api/config/compiler")
async def set_compiler_path(language: str = Form(...), path: str = Form(...)):
    """Set compiler path for a language"""
    if language not in COMPILERS:
        raise HTTPException(400, f"Unknown language: {language}")

    COMPILERS[language]["path"] = path
    return {"success": True, "language": language, "path": path}

# ===== Frontend =====

@app.get("/", response_class=HTMLResponse)
async def index():
    return open(Path(__file__).parent.parent / "frontend" / "index.html", encoding="utf-8").read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
