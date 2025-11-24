import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PROBLEMS_DIR = DATA_DIR / "problems"
SUBMISSIONS_DIR = DATA_DIR / "submissions"
TESTLIB_DIR = BASE_DIR / "testlib"

# Create directories
PROBLEMS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
TESTLIB_DIR.mkdir(parents=True, exist_ok=True)

# Compiler configurations
COMPILERS = {
    "c++14": {
        "path": str(BASE_DIR / "compilers" / "c++14" / "bin" / "g++.exe"),
        "args": ["-static", "-DONLINE_JUDGE", "-Wl,--stack=268435456", "-O2", "-std=c++14"],
        "bits": 32
    },
    "c++17": {
        "path": str(BASE_DIR / "compilers" / "c++17" / "bin" / "g++.exe"),
        "args": ["-static", "-DONLINE_JUDGE", "-Wl,--stack=268435456", "-O2", "-std=c++17"],
        "bits": 32
    },
    "c++20": {
        "path": str(BASE_DIR / "compilers" / "c++20" / "bin" / "g++.exe"),
        "args": ["-Wall", "-Wextra", "-Wconversion", "-static", "-DONLINE_JUDGE", "-Wl,--stack=268435456", "-O2", "-std=c++20"],
        "bits": 64
    },
    "c++23": {
        "path": str(BASE_DIR / "compilers" / "c++23" / "ucrt64" / "bin" / "g++.exe"),
        "args": ["-Wall", "-Wextra", "-Wconversion", "-static", "-DONLINE_JUDGE", "-Wl,--stack=268435456", "-O2", "-std=c++23"],
        "libs": ["-lstdc++exp"],
        "bits": 64
    }
}

# Judge settings
MAX_CONCURRENT_JUDGES = 4  # 降低并发避免CPU竞争导致TLE
DEFAULT_TIME_LIMIT = 1000  # ms
DEFAULT_MEMORY_LIMIT = 256  # MB
MAX_INPUT_SIZE = 10 * 1024 * 1024  # 10MB max input data for hack

# Database
DATABASE_URL = f"sqlite+aiosqlite:///{DATA_DIR}/oj.db"
