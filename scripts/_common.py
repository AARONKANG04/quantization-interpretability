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


# ---- shared evaluation and ledger helpers (used by every analysis script) ----

RAW_ROOT_DEFAULT = "/data/lm_eval_raw"


def eval_summaries(eval_dir: str | Path | None = None) -> dict:
    """All phase0_eval summaries keyed by condition name (file stem)."""
    eval_dir = Path(eval_dir or ROOT / "results/phase0/eval")
    return {p.stem: json.loads(p.read_text()) for p in sorted(eval_dir.glob("*.json"))}


def find_summary(summaries: dict, name: str, task: str):
    """The summary for `name` that holds `task`: the condition itself or one of its task-suffixed siblings (name_mmlu)."""
    for n, s in summaries.items():
        if (n == name or n.startswith(name + "_")) and task in s.get("results", {}):
            return s
    return None


def eval_items(summary: dict, task: str, raw_root: str | None = None) -> dict | None:
    """Per-item correctness from the raw lm-eval samples of one condition and task, restricted to the filter of the
    reported metric (exact_match,strict-match -> rows with filter == strict-match), keyed by (subtask, doc_id).
    raw_root remaps the recorded /data/lm_eval_raw prefix (for runs against a local archive)."""
    import glob
    r = summary["results"][task]
    mname, _, filt = r["metric"].partition(",")
    raw_dir = r["raw_dir"]
    if raw_root and not Path(raw_dir).exists():
        raw_dir = raw_dir.replace(RAW_ROOT_DEFAULT, str(Path(raw_root).expanduser()), 1)
    out = {}
    for sf in sorted(glob.glob(str(Path(raw_dir) / "**" / "samples_*.jsonl"), recursive=True)):
        subtask = Path(sf).name[len("samples_"):].rsplit("_", 1)[0]
        with open(sf) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if filt and filt != "none" and row.get("filter", filt) != filt:
                    continue
                if row.get(mname) is not None:
                    out[(subtask, row["doc_id"])] = bool(row[mname])
    return out or None


def paired_lists(items_a: dict, items_b: dict):
    """Correctness lists over the items both conditions have, in a fixed key order."""
    keys = sorted(set(items_a) & set(items_b))
    return [items_a[k] for k in keys], [items_b[k] for k in keys], keys


def bootstrap_module():
    """qi.metrics.bootstrap loaded without importing the qi package (so analyses run on a machine without torch)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("qi_bootstrap", ROOT / "qi" / "metrics" / "bootstrap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_ledger_block(header: str, lines: list):
    """Replace the ledger section that starts with `header` (up to the next level-2 heading), or append it."""
    path = ROOT / "results/LEDGER.md"
    text = path.read_text() if path.exists() else ""
    start = text.find(header)
    if start >= 0:
        nxt = text.find("\n## ", start + len(header))
        text = text[:start].rstrip("\n") + "\n" + ("\n" + text[nxt + 1:] if nxt >= 0 else "")
    path.write_text(text.rstrip("\n") + "\n" + "\n".join(lines) + "\n")


def set_ledger_row(placeholder: str, value: str, ci: str, control: str, source: str):
    """Fill one row of the placeholder table at the top of results/LEDGER.md (matched on the first cell)."""
    path = ROOT / "results/LEDGER.md"
    lines = path.read_text().split("\n")
    row = f"| {placeholder} | {value} | {ci} | {control} | {source} | {time.strftime('%Y-%m-%d')} |"
    for i, line in enumerate(lines):
        if line.startswith(f"| {placeholder} |"):
            lines[i] = row
            break
    else:
        raise KeyError(f"ledger has no row for {placeholder!r}")
    path.write_text("\n".join(lines))


def fmt_pts(x: float) -> str:
    return f"{100 * x:+.1f}"


def fmt_ci(lo: float, hi: float) -> str:
    return f"[{100 * lo:+.1f}, {100 * hi:+.1f}]"
