"""Shared helpers for scripts: run metadata, JSON output, .env loading."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def run_meta(args) -> dict:
    return {"git_sha": git_sha(), "args": vars(args) if hasattr(args, "__dict__") else dict(args),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": os.uname().nodename}


def write_json(path: str | Path, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"wrote {path}")
