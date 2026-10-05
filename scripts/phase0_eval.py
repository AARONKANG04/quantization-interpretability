"""Evaluate one checkpoint on the suites in configs/eval/suites.yaml with lm-eval over vLLM, one task per call,
and write a compact summary plus per-item correctness (needed later for paired bootstraps and flips).

usage: python scripts/phase0_eval.py --ckpt /data/ckpt/q1 --name q1 [--tasks gsm8k_cot,mmlu] [--dtype float32]
       [--limit 500] [--out results/phase0/eval]
"""
import argparse
import glob
import json
import os
import subprocess
from pathlib import Path

import yaml

from _common import ROOT, load_env, run_meta, write_json

load_env()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--suite", default=str(ROOT / "configs/eval/suites.yaml"))
    ap.add_argument("--tasks", default=None, help="comma subset of the suite's tasks")
    ap.add_argument("--dtype", default=None, help="override vLLM dtype, e.g. float32 for the C1 reference")
    ap.add_argument("--limit", type=int, default=None, help="items per task (sweeps use subsets)")
    ap.add_argument("--out", default=str(ROOT / "results/phase0/eval"))
    ap.add_argument("--raw-dir", default="/data/lm_eval_raw")
    ap.add_argument("--gpu-util", type=float, default=None, help="override vLLM gpu_memory_utilization")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "hf"], help="hf: transformers backend (fp32 reference)")
    ap.add_argument("--hf-batch", type=int, default=8)
    ap.add_argument("--reuse", action="store_true", help="summarise an existing raw lm-eval run instead of running it again")
    args = ap.parse_args()

    suite = yaml.safe_load(Path(args.suite).read_text())
    wanted = set(args.tasks.split(",")) if args.tasks else None
    vllm_args = suite["vllm_args"]
    if args.dtype:
        vllm_args = ",".join(a for a in vllm_args.split(",") if not a.startswith("dtype=")) + f",dtype={args.dtype}"
    if args.gpu_util is not None:
        vllm_args = ",".join(a for a in vllm_args.split(",") if not a.startswith("gpu_memory_utilization=")) + f",gpu_memory_utilization={args.gpu_util}"
    summary = {"meta": run_meta(args), "ckpt": args.ckpt, "results": {}}
    for t in suite["tasks"]:
        if wanted and t["task"] not in wanted:
            continue
        raw = Path(args.raw_dir) / args.name / t["task"]
        raw.mkdir(parents=True, exist_ok=True)
        if args.backend == "hf":
            margs = f"pretrained={args.ckpt},dtype={args.dtype or 'bfloat16'},max_length=4096"
            cmd = ["lm_eval", "--model", "hf", "--model_args", margs, "--batch_size", str(args.hf_batch)]
        else:
            cmd = ["lm_eval", "--model", "vllm", "--model_args", f"pretrained={args.ckpt},{vllm_args}", "--batch_size", "auto"]
        cmd += ["--tasks", t["task"], "--num_fewshot", str(t["num_fewshot"]),
               "--output_path", str(raw), "--log_samples", "--seed", "1234"]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        res_files = sorted(glob.glob(str(raw / "**" / "results_*.json"), recursive=True))
        if args.reuse and res_files:
            print("reusing", res_files[-1], flush=True)
        else:
            print(" ".join(cmd), flush=True)
            subprocess.run(cmd, check=True, env={**os.environ, "TOKENIZERS_PARALLELISM": "false"})
            res_files = sorted(glob.glob(str(raw / "**" / "results_*.json"), recursive=True))
        res = json.loads(Path(res_files[-1]).read_text())["results"]
        task_res = res.get(t["task"]) or next(iter(res.values()))
        metric_key = next((k for k in task_res if k.startswith(t["metric"]) and not k.endswith("stderr")), None)
        summary["results"][t["task"]] = {"metric": metric_key, "value": task_res.get(metric_key),
                                         "all_metrics": {k: v for k, v in task_res.items() if isinstance(v, (int, float))},
                                         "num_fewshot": t["num_fewshot"], "raw_dir": str(raw)}
        # per-item correctness for paired bootstraps and flips; group tasks (mmlu) write one samples file per subtask
        samples = sorted(glob.glob(str(raw / "**" / f"samples_{t['task']}*.jsonl"), recursive=True))
        items = []
        for sf in samples:
            subtask = Path(sf).name[len("samples_"):].rsplit("_", 1)[0]
            with open(sf) as f:
                for line in f:
                    r = json.loads(line)
                    val = r.get(t["metric"], r.get(metric_key))
                    items.append({"task": subtask, "doc_id": r.get("doc_id"), "correct": val})
        if items:
            Path(args.out, args.name).mkdir(parents=True, exist_ok=True)
            Path(args.out, args.name, f"{t['task']}_items.json").write_text(json.dumps(items))
    write_json(Path(args.out) / f"{args.name}.json", summary)
    print({k: v["value"] for k, v in summary["results"].items()})


if __name__ == "__main__":
    main()
