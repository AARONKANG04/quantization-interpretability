"""Gate for the full-size steering run: exit 0 when the steering arm beats the no-steering baseline on the subset
by at least --min-delta (accuracy points as a fraction), else exit 1 so the queue skips the expensive run.

usage: python scripts/phase2_steer_gate.py results/phase2/steer/q1_L17_none.json results/phase2/steer/q1_L17_gain.json [--min-delta 0.01]
"""
import argparse
import json
import sys


def acc(path):
    d = json.load(open(path))
    r = d.get("metrics") or d.get("results", d)
    for k in ("exact_match,strict-match", "strict", "exact_match"):
        if isinstance(r, dict) and isinstance(r.get(k), (int, float)):
            return float(r[k])
    items = d.get("items", [])
    vals = [i["strict"] for i in items if i.get("strict") is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline"); ap.add_argument("arm"); ap.add_argument("--min-delta", type=float, default=0.01)
    a = ap.parse_args()
    base, arm = acc(a.baseline), acc(a.arm)
    delta = arm - base
    print(f"baseline {base:.4f} arm {arm:.4f} delta {delta:+.4f} (gate {a.min_delta:+.4f})")
    sys.exit(0 if delta >= a.min_delta else 1)


if __name__ == "__main__":
    main()
