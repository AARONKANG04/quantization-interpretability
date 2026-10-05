"""Damage table and noise floor from the evals: paired bootstrap deltas and flips against the reference for every
condition that has raw samples, written to results/phase0/damage_table*.json and a ledger section per reference.

usage: python scripts/phase0_damage_table.py [--eval-dir results/phase0/eval] [--ref bf16] [--raw-root ~/qi_archive/boxA/lm_eval_raw]
       The boxS evals pair against bf16 (default); the boxA evals pair against bf16_rep: --ref bf16_rep --out results/phase0/damage_table_boxA.json
"""
import argparse

from _common import ROOT, bootstrap_module, eval_items, eval_summaries, load_env, paired_lists, run_meta, write_json, write_ledger_block

load_env()
bs = bootstrap_module()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=str(ROOT / "results/phase0/eval")); ap.add_argument("--ref", default="bf16")
    ap.add_argument("--raw-root", default=None)
    ap.add_argument("--out", default=None, help="default results/phase0/damage_table.json (ref bf16) or damage_table_<ref>.json")
    args = ap.parse_args()
    out_path = args.out or str(ROOT / ("results/phase0/damage_table.json" if args.ref == "bf16" else f"results/phase0/damage_table_{args.ref}.json"))
    summaries = eval_summaries(args.eval_dir)
    is_ref = lambda n: n == args.ref or n.startswith(args.ref + "_")  # noqa: E731  (bf16, bf16_arc_hs, bf16_rep_mmlu, ...)
    ref_for = {task: s for name, s in summaries.items() if is_ref(name) for task in s["results"]}
    ref_items = {task: eval_items(s, task, args.raw_root) for task, s in ref_for.items()}
    table = {}
    for name, s in summaries.items():
        table[name] = {}
        for task, r in s["results"].items():
            entry = {"metric": r["metric"], "value": r["value"], "all": r.get("all_metrics", {})}
            if not is_ref(name) and ref_items.get(task):
                b = eval_items(s, task, args.raw_root)
                if b:
                    ca, cb, _ = paired_lists(ref_items[task], b)
                    if ca:
                        entry["paired"] = bs.paired_bootstrap_delta(ca, cb)
                        entry["flips"] = bs.flips(ca, cb)
            table[name][task] = entry
    write_json(out_path, {**run_meta(args), "ref": args.ref, "table": table})
    header = f"## Damage table (ref {args.ref}"
    lines = ["", f"{header}, paired bootstrap, {run_meta(args)['started']})", "",
             "| Condition | Task | Accuracy | Delta vs ref | 95% CI | Flips c->i / i->c |", "| --- | --- | --- | --- | --- | --- |"]
    for name, tasks in sorted(table.items()):
        for task, e in tasks.items():
            p, fl = e.get("paired"), e.get("flips")
            if not p and not is_ref(name) and args.ref != "bf16":
                continue  # on a second reference only list what pairs with it
            lines.append(f"| {name} | {task} | {e['value']:.4f} | " + (f"{p['delta']:+.4f} | [{p['lo']:+.4f}, {p['hi']:+.4f}] | {fl['correct_to_incorrect']:.3f} / {fl['incorrect_to_correct']:.3f} |" if p else "| | |"))
    floor = [n for n in table if n.startswith("c1_")]
    if floor:
        lines += ["", "### Noise floor (C1: bf16 vs fp32)", "", "| Task | bf16 | fp32 | delta | paired 95% CI |", "| --- | --- | --- | --- | --- |"]
        for n in sorted(floor):
            for task, e in table[n].items():
                p = e.get("paired")
                if p:
                    lines.append(f"| {task} | {ref_for[task]['results'][task]['value']:.4f} | {e['value']:.4f} | {p['delta']:+.4f} | [{p['lo']:+.4f}, {p['hi']:+.4f}] |")
    write_ledger_block(header, lines)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
