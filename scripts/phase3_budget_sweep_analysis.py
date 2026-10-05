"""H4 numbers: recovery of the mixed-precision checkpoints per method and budget with paired bootstrap CIs, the
pairwise decision rule (feature-guided vs each baseline at two of three budgets), the whole-layer fallback curve
and the memory accounting. Writes results/phase3/budget_sweep.json, results/phase3/memory.json and the ledger.

usage: python scripts/phase3_budget_sweep_analysis.py [--ref bf16_rep] [--quant q1_rep] [--raw-root ~/qi_archive/boxA/lm_eval_raw]
"""
import argparse
import json
import re

from _common import (ROOT, bootstrap_module, eval_items, eval_summaries, find_summary, fmt_ci, fmt_pts, load_env,
                     paired_lists, run_meta, set_ledger_row, write_json, write_ledger_block)

load_env()
bs = bootstrap_module()
PAT = re.compile(r"^mix_([a-z_]+?)_(\d{4})$")
BYTES_PER_PROTECTED_WEIGHT = 1.4375  # bf16 (2 bytes) instead of NVFP4 (0.5625 bytes with scales)
METHOD_NAMES = {"fg_proj": "feature-guided (FG-proj)", "mag": "magnitude", "klg": "KL-gradient", "rand": "random channels"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="bf16_rep"); ap.add_argument("--quant", default="q1_rep")
    ap.add_argument("--raw-root", default=None); ap.add_argument("--tasks", default="gsm8k_cot,mmlu")
    ap.add_argument("--selections", default=str(ROOT / "results/phase3/selections.json"))
    ap.add_argument("--primary", default="fg_proj"); ap.add_argument("--target", type=float, default=0.8)
    ap.add_argument("--out", default=str(ROOT / "results/phase3/budget_sweep.json"))
    ap.add_argument("--memory-out", default=str(ROOT / "results/phase3/memory.json"))
    args = ap.parse_args()
    summaries = eval_summaries()
    sel = json.loads(open(args.selections).read())
    extra_bytes = {(s["method"], round(s["budget_frac"], 4)): s["extra_bytes"] for s in sel["selections"]}
    layer_bytes = sel["layer_weights"] * BYTES_PER_PROTECTED_WEIGHT
    conds = {}
    for name in summaries:
        m = PAT.match(name)
        if m:
            conds[name] = {"method": m.group(1), "budget": int(m.group(2)) / 1000}
    out = {**run_meta(args), "bytes_per_protected_weight": BYTES_PER_PROTECTED_WEIGHT, "layer_bytes": layer_bytes, "tasks": {}}
    for task in args.tasks.split(","):
        ref_s, q_s = find_summary(summaries, args.ref, task), find_summary(summaries, args.quant, task)
        if not ref_s or not q_s:
            print(f"{task}: reference pair missing, skipped"); continue
        ref_items, q_items = eval_items(ref_s, task, args.raw_root), eval_items(q_s, task, args.raw_root)
        if not ref_items or not q_items:
            print(f"{task}: raw samples missing, skipped"); continue
        a, b, _ = paired_lists(ref_items, q_items)
        loss = bs.paired_bootstrap_delta(a, b)
        L = -loss["delta"]
        T = {"n_items": loss["n"], "acc_ref": loss["acc_a"], "acc_quant": loss["acc_b"], "loss": L, "loss_ci": [-loss["hi"], -loss["lo"]],
             "conditions": {}, "pairwise": {}, "layer_fallback": {}}
        items = {}
        for name, c in conds.items():
            s = find_summary(summaries, name, task)
            it = eval_items(s, task, args.raw_root) if s else None
            if not it:
                continue
            items[name] = it
            qa, xb, _ = paired_lists(q_items, it)
            d = bs.paired_bootstrap_delta(qa, xb)
            eb = extra_bytes.get((c["method"], round(c["budget"], 4)))
            T["conditions"][name] = {**c, "acc": d["acc_b"], "recovered_pts": d["delta"], "recovered_ci": [d["lo"], d["hi"]],
                                     "recovery_frac": d["delta"] / L, "recovery_frac_ci": [d["lo"] / L, d["hi"] / L],
                                     "extra_bytes": eb, "extra_mb": eb / 1e6 if eb else None, "n": d["n"]}
        # pairwise: primary vs each baseline at each budget, same items
        for base in sorted({c["method"] for c in conds.values()} - {args.primary}):
            rows = {}
            for b in sorted({c["budget"] for c in conds.values()}):
                p = next((n for n, c in conds.items() if c["method"] == args.primary and c["budget"] == b and n in items), None)
                q = next((n for n, c in conds.items() if c["method"] == base and c["budget"] == b and n in items), None)
                if p and q:
                    xa, xb, _ = paired_lists(items[q], items[p])
                    d = bs.paired_bootstrap_delta(xa, xb)  # primary - baseline
                    rows[str(b)] = {"delta": d["delta"], "ci": [d["lo"], d["hi"]], "beats": d["lo"] > 0, "loses": d["hi"] < 0}
            n_beats = sum(r["beats"] for r in rows.values())
            T["pairwise"][base] = {"budgets": rows, "primary_beats_baseline": n_beats >= 2,
                                   "verdict": "beats" if n_beats >= 2 else ("loses" if sum(r["loses"] for r in rows.values()) >= 2 else "ties")}
        # whole-layer fallback curve from the ranked restore evals
        for name in summaries:
            m = re.match(r"^rc_ranked_top(\d+)$", name)
            s = find_summary(summaries, name, task) if m else None
            it = eval_items(s, task, args.raw_root) if s else None
            if it:
                qa, xb, _ = paired_lists(q_items, it)
                d = bs.paired_bootstrap_delta(qa, xb)
                n = int(m.group(1))
                T["layer_fallback"][str(n)] = {"n_layers": n, "acc": d["acc_b"], "recovered_pts": d["delta"], "recovered_ci": [d["lo"], d["hi"]],
                                               "recovery_frac": d["delta"] / L, "extra_bytes": n * layer_bytes, "extra_mb": n * layer_bytes / 1e6}
        out["tasks"][task] = T
    # average recovery (GSM8K + MMLU) per condition and the budget Z
    avg = {}
    for name in conds:
        vals = [out["tasks"][t]["conditions"][name]["recovery_frac"] for t in out["tasks"] if name in out["tasks"][t]["conditions"]]
        if vals:
            avg[name] = sum(vals) / len(vals)
    out["avg_recovery"] = avg
    prim = sorted(((conds[n]["budget"], n) for n in avg if conds[n]["method"] == args.primary))
    z = next((b for b, n in prim if avg[n] >= args.target), None)
    out["budget_Z"] = z; out["target_recovery"] = args.target
    best_b, best_n = max(prim, key=lambda bn: avg[bn[1]]) if prim else (None, None)
    out["best_primary"] = {"budget": best_b, "name": best_n, "avg_recovery": avg.get(best_n)}
    write_json(args.out, out)

    # memory: bytes of the primary method at each budget vs bytes of whole-layer fallback at the same GSM8K recovery
    mem = {**run_meta(args), "layer_bytes": layer_bytes, "rows": []}
    g = out["tasks"].get("gsm8k_cot", {})
    for b, n in prim:
        c = g.get("conditions", {}).get(n)
        if not c:
            continue
        reach = [lf for lf in g["layer_fallback"].values() if lf["recovery_frac"] >= c["recovery_frac"]]
        match = min(reach, key=lambda lf: lf["n_layers"]) if reach else None
        most = max(g["layer_fallback"].values(), key=lambda lf: lf["n_layers"]) if g["layer_fallback"] else None
        mem["rows"].append({"budget": b, "extra_mb": c["extra_mb"], "recovery_frac": c["recovery_frac"],
                            "layer_fallback_match": match, "memory_ratio": (c["extra_bytes"] / match["extra_bytes"]) if (match and c["extra_bytes"]) else None,
                            "layer_fallback_never_reaches": match is None, "largest_layer_fallback": most})
    write_json(args.memory_out, mem)

    lines = ["", f"## Phase 3: mixed precision budget sweep (ref {args.ref}, quant {args.quant}, paired bootstrap, {out['started']})", ""]
    for task, T in out["tasks"].items():
        lines += [f"**{task}**: loss {100 * T['loss']:.2f} {fmt_ci(*T['loss_ci'])} pts on {T['n_items']} items.", "",
                  "| Method | Budget | Extra MB | Acc | Recovered pts [95% CI] | Recovery |", "| --- | --- | --- | --- | --- | --- |"]
        for name, c in sorted(T["conditions"].items(), key=lambda kv: (kv[1]["method"], kv[1]["budget"])):
            lines.append(f"| {METHOD_NAMES.get(c['method'], c['method'])} | {100 * c['budget']:.0f}% | {c['extra_mb']:.0f} | {100 * c['acc']:.2f} | "
                         f"{fmt_pts(c['recovered_pts'])} {fmt_ci(*c['recovered_ci'])} | {100 * c['recovery_frac']:.0f}% |")
        for n, lf in sorted(T["layer_fallback"].items(), key=lambda kv: int(kv[0])):
            lines.append(f"| whole-layer fallback | {lf['n_layers']} layers | {lf['extra_mb']:.0f} | {100 * lf['acc']:.2f} | "
                         f"{fmt_pts(lf['recovered_pts'])} {fmt_ci(*lf['recovered_ci'])} | {100 * lf['recovery_frac']:.0f}% |")
        lines += ["", f"Decision rule ({METHOD_NAMES[args.primary]} vs baseline, paired, needs a win at two of three budgets):", ""]
        for base, pw in T["pairwise"].items():
            lines.append(f"- vs {METHOD_NAMES.get(base, base)}: **{pw['verdict']}** (" + "; ".join(
                f"{100 * float(b):.0f}%: {fmt_pts(r['delta'])} {fmt_ci(*r['ci'])}" for b, r in pw["budgets"].items()) + ")")
        lines.append("")
    write_ledger_block("## Phase 3: mixed precision budget sweep", lines)
    print("\n".join(lines))
    if best_n and g:
        c = g["conditions"][best_n]
        row = next((r for r in mem["rows"] if r["budget"] == best_b), None)
        ztxt = f"{100 * best_b:.0f}% (80% recovery not reached at any budget; best {METHOD_NAMES[args.primary]} budget)" if z is None else f"{100 * z:.0f}%"
        set_ledger_row("[Z]% of weights in bf16", ztxt, "", "RAND, MAG, KLG at the same budgets", "results/phase3/budget_sweep.json")
        set_ledger_row("recovering [X]% of FP4 loss", f"{100 * c['recovery_frac']:.0f}% of the GSM8K loss at {100 * best_b:.0f}% ({c['extra_mb']:.0f} MB); avg GSM8K+MMLU {100 * avg[best_n]:.0f}%",
                       fmt_ci(*c["recovery_frac_ci"]).replace("+", ""),
                       "; ".join(f"{METHOD_NAMES.get(b, b)}: {pw['verdict']}" for b, pw in g["pairwise"].items()), "results/phase3/budget_sweep.json")
        if row:
            if row["memory_ratio"] is not None:
                mtxt = f"{100 * row['memory_ratio']:.0f}% of the memory of whole-layer fallback at the same recovery ({row['layer_fallback_match']['n_layers']} layers, {row['layer_fallback_match']['extra_mb']:.0f} MB)"
            else:
                lf = row["largest_layer_fallback"]
                mtxt = f"whole-layer fallback never reaches this recovery within {lf['n_layers']} layers ({lf['extra_mb']:.0f} MB, {100 * lf['recovery_frac']:.0f}%); {c['extra_mb']:.0f} MB is {100 * c['extra_bytes'] / lf['extra_bytes']:.0f}% of that"
            set_ledger_row("[M]% of memory of full-layer fallback", mtxt, "", "LAYER curve (ranked restore evals)", "results/phase3/memory.json")


if __name__ == "__main__":
    main()
