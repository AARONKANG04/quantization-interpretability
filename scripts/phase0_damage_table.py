"""Damage table and noise floor from the Phase 0 evals: paired bootstrap deltas and flips against bf16 for every
condition, written to results/phase0/damage_table.json and appended to results/LEDGER.md.

usage: python scripts/phase0_damage_table.py [--eval-dir results/phase0/eval] [--ref bf16]
"""
import argparse
import json
from pathlib import Path

from _common import ROOT, load_env, run_meta, write_json

load_env()
from qi.metrics.bootstrap import flips, paired_bootstrap_delta  # noqa: E402


def items(summary, task):
    """Per-item correctness from the raw lm-eval samples, restricted to the filter of the reported metric
    (e.g. exact_match,strict-match -> rows with filter == strict-match), keyed by (subtask, doc_id)."""
    import glob
    r = summary["results"][task]
    mname, _, filt = r["metric"].partition(",")
    out = {}
    for sf in sorted(glob.glob(str(Path(r["raw_dir"]) / "**" / "samples_*.jsonl"), recursive=True)):
        subtask = Path(sf).name[len("samples_"):].rsplit("_", 1)[0]
        with open(sf) as f:
            for line in f:
                row = json.loads(line)
                if filt and row.get("filter", filt) != filt:
                    continue
                if row.get(mname) is not None:
                    out[(subtask, row["doc_id"])] = bool(row[mname])
    return out or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=str(ROOT / "results/phase0/eval")); ap.add_argument("--ref", default="bf16")
    ap.add_argument("--out", default=str(ROOT / "results/phase0/damage_table.json"))
    args = ap.parse_args()
    eval_dir = Path(args.eval_dir)
    summaries = {p.stem: json.loads(p.read_text()) for p in eval_dir.glob("*.json")}
    table = {}
    for name, s in summaries.items():
        table[name] = {}
        for task, r in s["results"].items():
            entry = {"metric": r["metric"], "value": r["value"], "all": r.get("all_metrics", {})}
            if name != args.ref and args.ref in summaries:
                a, b = items(summaries[args.ref], task), items(s, task)
                if a and b:
                    keys = sorted(set(a) & set(b))
                    ca = [a[k] for k in keys]; cb = [b[k] for k in keys]
                    entry["paired"] = paired_bootstrap_delta(ca, cb)
                    entry["flips"] = flips(ca, cb)
            table[name][task] = entry
    write_json(args.out, {**run_meta(args), "ref": args.ref, "table": table})
    lines = ["", f"## Damage table (ref {args.ref}, paired bootstrap, {run_meta(args)['started']})", "",
             "| Condition | Task | Accuracy | Delta vs ref | 95% CI | Flips c->i / i->c |", "| --- | --- | --- | --- | --- | --- |"]
    for name, tasks in sorted(table.items()):
        for task, e in tasks.items():
            p, fl = e.get("paired"), e.get("flips")
            lines.append(f"| {name} | {task} | {e['value']:.4f} | " + (f"{p['delta']:+.4f} | [{p['lo']:+.4f}, {p['hi']:+.4f}] | {fl['correct_to_incorrect']:.3f} / {fl['incorrect_to_correct']:.3f} |" if p else "| | |"))
    with open(ROOT / "results/LEDGER.md", "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
