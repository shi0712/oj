from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, Enum as SQLEnum
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import enum

from config import DATABASE_URL

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

class JudgeStatus(str, enum.Enum):
    PENDING = "Pending"
    JUDGING = "Judging"
    ACCEPTED = "Accepted"
    WRONG_ANSWER = "Wrong Answer"
    TIME_LIMIT = "Time Limit Exceeded"
    MEMORY_LIMIT = "Memory Limit Exceeded"
    RUNTIME_ERROR = "Runtime Error"
    COMPILE_ERROR = "Compile Error"
    SYSTEM_ERROR = "System Error"

class Problem(Base):
    __tablename__ = "problems"

    id = Column(String(64), primary_key=True)
    title = Column(String(256), default="")
    time_limit = Column(Integer, default=1000)  # ms
    memory_limit = Column(Integer, default=256)  # MB
    has_checker = Column(Boolean, default=False)
    test_case_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

class Submission(Base):
    __tablename__ = "submissions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    problem_id = Column(String(64), nullable=False)
    code = Column(Text, nullable=False)
    language = Column(String(16), nullable=False)
    status = Column(String(32), default=JudgeStatus.PENDING.value)
    time_used = Column(Integer, default=0)  # ms
    memory_used = Column(Integer, default=0)  # KB
    score = Column(Integer, default=0)
    message = Column(Text, default="")
    failed_case = Column(Integer, default=0)  # Failed test case number (0 if AC)
    created_at = Column(DateTime, default=datetime.utcnow)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_session():
    async with async_session() as session:
        yield session
