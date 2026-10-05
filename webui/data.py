"""Read-only loaders and joins for the trace browser.

Sources
  lm-eval raw samples  <raw_dir>/<condition>/<task>/<model_dir>/samples_<subtask>_<timestamp>.jsonl
  token KL dumps       <dump_dir>/<cond>_<set>.parquet, with <cond>_<set>_summary.json beside it

The raw directory may still be syncing, so nothing is hard-coded: conditions are discovered on request (cached for a
few seconds) and parsed files are cached by size and modification time, so new or growing files are picked up.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------------------------------------------
# Names. Extend these tables (or describe()) when new conditions, tasks or token sets appear.
# ---------------------------------------------------------------------------------------------------------------

GROUPS = ["Reference and quantized", "Layers kept in bf16, ranked", "Layers kept in bf16, random order",
          "Mixed precision", "Other"]
FIXED_NAMES = {  # condition -> (display name, short label)
    "bf16_rep": ("bf16", "bf16"), "bf16": ("bf16", "bf16"),
    "q1_rep": ("NVFP4", "NVFP4"), "q1": ("NVFP4", "NVFP4"),
    "q3": ("FP8", "FP8"),
    "c1_fp32": ("fp32", "fp32"),
    "c2": ("Gaussian noise matched to NVFP4 (C2)", "C2 noise"),
}
MIX_METHODS = {"fg_proj": ("feature-guided", "FG"), "mag": ("magnitude", "MAG"), "klg": ("KL-gradient", "KLG"),
               "rand": ("random channels", "RAND")}
TASK_LABELS = {"gsm8k_cot": "GSM8K", "gsm8k": "GSM8K", "mmlu": "MMLU", "arc_challenge": "ARC-C",
               "arc_easy": "ARC-E", "hellaswag": "HellaSwag"}
SET_LABELS = {"kl_2m": "held-out mix, 2M tokens", "kl_500k": "held-out mix, 500k tokens",
              "gsm8k_test": "GSM8K test", "pg19_long": "PG-19 long documents"}


def describe(cond: str) -> dict:
    """Display name, short label and group of a condition name (directory name without the `_mmlu` suffix)."""
    def out(name, short, group, *rank):
        return {"id": cond, "name": name, "short": short, "group": GROUPS[group], "_rank": (group, *rank)}

    if cond in FIXED_NAMES:
        return out(*FIXED_NAMES[cond], 0, list(FIXED_NAMES).index(cond))
    m = re.fullmatch(r"rc_ranked_top(\d+)", cond)
    if m:
        n = int(m[1])
        return out(f"NVFP4 + top-{n} layers in bf16 (ranked)", f"top-{n} ranked", 1, n)
    m = re.fullmatch(r"rc_random_s(\d+)_top(\d+)", cond)
    if m:
        s, n = int(m[1]), int(m[2])
        return out(f"NVFP4 + top-{n} layers in bf16 (random order, seed {s})", f"top-{n} random s{s}", 2, s, n)
    m = re.fullmatch(r"mix_(.+)_(\d{4})", cond)
    if m and m[1] in MIX_METHODS:
        method, short = MIX_METHODS[m[1]]
        pct = f"{int(m[2]) / 10:g}"  # 0010 -> 1 (% of weights)
        return out(f"Mixed precision, {method}, {pct}% of weights in bf16", f"{short} {pct}%", 3,
                   int(m[2]), list(MIX_METHODS).index(m[1]))
    return out(cond, cond, 4)


def public(desc: dict) -> dict:
    return {k: v for k, v in desc.items() if not k.startswith("_")}


def task_rank(task: str):
    order = ["gsm8k_cot", "mmlu"]
    return (order.index(task) if task in order else len(order), task)


# ---------------------------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------------------------

SAMPLES_RE = re.compile(r"^samples_(?P<sub>.+)_(?P<ts>\d{4}-\d\d-\d\dT[\d.\-]+)\.jsonl$")
RESULTS_RE = re.compile(r"^results_(?P<ts>\d{4}-\d\d-\d\dT[\d.\-]+)\.json$")
VARIANT_ORDER = ["strict-match", "flexible-extract", "acc", "acc_norm"]
LETTERS = "ABCDEFGHIJ"


def _subdirs(path: Path) -> list[Path]:
    try:
        return sorted(p for p in path.iterdir() if p.is_dir() and not p.name.startswith("."))
    except OSError:
        return []


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _stamp(path: Path, regex) -> tuple:
    """Sort key for lm-eval output files: the timestamp in the name, then the modification time."""
    m = regex.match(path.name)
    return (m["ts"] if m else "", _mtime(path))


def latest_samples(task_dir: Path) -> dict[str, Path]:
    """Latest samples file per subtask under a task directory (any depth)."""
    best: dict[str, Path] = {}
    try:
        found = list(task_dir.rglob("samples_*.jsonl"))
    except OSError:
        return best
    for p in found:
        m = SAMPLES_RE.match(p.name)
        sub = m["sub"] if m else p.stem[len("samples_"):]
        if sub not in best or _stamp(p, SAMPLES_RE) > _stamp(best[sub], SAMPLES_RE):
            best[sub] = p
    return best


def _signature(files: dict[str, Path]) -> tuple:
    sig = []
    for p in sorted(files.values()):
        try:
            st = p.stat()
            sig.append((str(p), st.st_size, st.st_mtime_ns))
        except OSError:
            sig.append((str(p), -1, -1))
    return tuple(sig)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def clean(obj):
    """Replace NaN and infinities (not valid JSON) with None, recursively."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    return obj


def _ordered(variants) -> list[str]:
    return sorted(variants, key=lambda v: (VARIANT_ORDER.index(v) if v in VARIANT_ORDER else 99, v))


def _short_sub(task: str, sub: str) -> str:
    return sub[len(task) + 1:] if sub.startswith(task + "_") else sub


def item_key(task: str, sub: str, doc_id) -> str:
    """Join key of a document: the doc id, prefixed by the subtask for group tasks (mmlu doc ids repeat)."""
    return str(doc_id) if sub == task else f"{_short_sub(task, sub)}:{doc_id}"


def _generation(row):
    try:
        x = row["resps"][0][0]
    except (KeyError, IndexError, TypeError):
        return None
    return x if isinstance(x, str) else None


def _first_str(x) -> str:
    while isinstance(x, list) and x:
        x = x[0]
    return "" if x is None or isinstance(x, list) else str(x)


def _loglik(x):
    """[[loglik, is_greedy]] or [loglik, is_greedy] (possibly as strings) -> float, or None."""
    while isinstance(x, list) and x and isinstance(x[0], list):
        x = x[0]
    if isinstance(x, list):
        x = x[0] if x else None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _argmax(values):
    best = None
    for i, v in enumerate(values):
        if v is not None and (best is None or v > values[best]):
            best = i
    return best


def _letter(i) -> str:
    return "" if i is None else (LETTERS[i] if i < len(LETTERS) else str(i))


def _correct(row, names, fallback=None):
    for name in names:
        v = row.get(name)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return bool(v >= 0.5)
        if isinstance(v, bool):
            return v
    return fallback


def _args(row) -> list:
    """The request arguments of a row as a list of dicts with arg_0 (context) and arg_1 (continuation or gen args)."""
    a = row.get("arguments")
    if isinstance(a, dict):
        keys = sorted(a, key=lambda k: int(re.sub(r"\D", "", k) or 0))
        return [a[k] if isinstance(a[k], dict) else {} for k in keys]
    if isinstance(a, list):
        return [{"arg_0": x[0], "arg_1": x[1] if len(x) > 1 else None} if isinstance(x, list) and x else {} for x in a]
    return []


def _prompt(row) -> str:
    args = _args(row)
    p = args[0].get("arg_0") if args else None
    return p if isinstance(p, str) else ""


def _question(doc: dict, prompt: str) -> str:
    for k in ("question", "query", "problem", "ctx", "goal", "sentence", "input", "text"):
        v = doc.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return prompt[-600:].strip()


def _choices(doc: dict, row) -> list[str]:
    c = doc.get("choices")
    if isinstance(c, list) and all(isinstance(x, str) for x in c):
        return c
    if isinstance(c, dict) and isinstance(c.get("text"), list):
        return [str(x) for x in c["text"]]
    if isinstance(doc.get("endings"), list):
        return [str(x) for x in doc["endings"]]
    return [str(a.get("arg_1", "")).strip() for a in _args(row)]


def _gold_index(row, doc: dict, choices: list[str]):
    for v in (row.get("target"), doc.get("answer"), doc.get("label"), doc.get("gold")):
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            s = v.strip()
            if s.isdigit():
                return int(s)
            if s in choices:
                return choices.index(s)
            if len(s) == 1 and s.upper() in LETTERS:
                return LETTERS.index(s.upper())
    return None


def split_prompt(prompt: str, question: str) -> tuple[str, str]:
    """(few-shot prefix, final query): split at the paragraph holding the last occurrence of the question."""
    i = prompt.rfind(question[:80]) if question else -1
    if i < 0:
        i = prompt.rfind("Q:")
    if i <= 0:
        return "", prompt
    j = prompt.rfind("\n\n", 0, i)
    cut = j + 2 if j >= 0 else i
    return prompt[:cut], prompt[cut:]


def _snippet(text: str, n: int = 160) -> str:
    line = " ".join(text.split())
    return line if len(line) <= n else line[: n - 1] + "…"


# ---------------------------------------------------------------------------------------------------------------
# lm-eval raw samples
# ---------------------------------------------------------------------------------------------------------------

class TaskData:
    """One condition on one task: every subtask file merged, one entry per document key."""

    def __init__(self, task: str, files: dict[str, Path]):
        self.task = task
        self.signature = _signature(files)
        self.kind = None                     # "generative" or "choice"
        self.docs: dict[str, dict] = {}      # key -> question, gold, subject, choices (same in every condition)
        self.gen: dict[str, str] = {}        # key -> generated text
        self.ll: dict[str, list] = {}        # key -> log-likelihood per choice
        self.scores: dict[str, dict] = {}    # filter (or acc / acc_norm) -> key -> (correct or None, answer)
        self.where: dict[str, tuple] = {}    # key -> (file, byte offset) of its first row, to re-read the prompt
        for sub, path in sorted(files.items()):
            self._read(sub, path)

    def _read(self, sub: str, path: Path):
        try:
            f = open(path, "rb")
        except OSError:
            return
        with f:
            offset = 0
            for line in f:
                start, offset = offset, offset + len(line)
                try:
                    row = json.loads(line)
                    if isinstance(row, dict) and "doc_id" in row:
                        self._add(sub, row, (path, start))
                except Exception:  # truncated (still syncing) or malformed line: skip it
                    continue

    def _add(self, sub: str, row: dict, where: tuple):
        doc = row.get("doc") if isinstance(row.get("doc"), dict) else {}
        key = item_key(self.task, sub, row["doc_id"])
        text = _generation(row)
        kind = "generative" if text is not None else "choice"
        self.kind = self.kind or kind
        if key not in self.docs:
            self.docs[key] = self._doc_fields(sub, row, doc, kind)
            self.where[key] = where
        if kind == "generative":
            self.gen.setdefault(key, text)
            names = tuple(m for m in row.get("metrics") or () if isinstance(m, str)) + ("exact_match", "acc")
            filt = str(row.get("filter") or "none")
            self.scores.setdefault(filt, {})[key] = (_correct(row, names), _first_str(row.get("filtered_resps")))
            return
        lls = [_loglik(x) for x in row.get("resps") or []]
        self.ll[key] = lls
        gold = self.docs[key].get("gold_index")
        chosen = _argmax(lls)
        self.scores.setdefault("acc", {})[key] = (_correct(row, ("acc",), chosen == gold), _letter(chosen))
        if "acc_norm" in row:  # lm-eval normalises by the character length of each choice (no leading delimiter)
            conts = [str(a.get("arg_1") or "").removeprefix(" ") for a in _args(row)]
            norm = [None if v is None else v / max(len(c), 1) for v, c in zip(lls, conts)] if len(conts) == len(lls) else lls
            cn = _argmax(norm)
            self.scores.setdefault("acc_norm", {})[key] = (_correct(row, ("acc_norm",), cn == gold), _letter(cn))

    def _doc_fields(self, sub: str, row: dict, doc: dict, kind: str) -> dict:
        question = _question(doc, _prompt(row))
        subject = doc.get("subject") if isinstance(doc.get("subject"), str) else None
        if subject is None and sub != self.task:
            subject = _short_sub(self.task, sub)
        fields = {"doc_id": row["doc_id"], "subject": subject, "question": question}
        if kind == "generative":
            reasoning = next((doc[k] for k in ("answer", "solution") if isinstance(doc.get(k), str)), None)
            target = row.get("target")
            if target is not None and not isinstance(target, (dict, list)):
                gold = str(target)
            else:
                gold = reasoning.split("####")[-1].strip() if reasoning and "####" in reasoning else ""
            fields.update(gold=gold, reasoning=reasoning)
        else:
            choices = _choices(doc, row)
            gi = _gold_index(row, doc, choices)
            fields.update(choices=choices, gold_index=gi, gold=_letter(gi) if gi is not None else str(row.get("target", "")))
        return fields

    def read_row(self, key: str):
        where = self.where.get(key)
        if not where:
            return None
        try:
            with open(where[0], "rb") as f:
                f.seek(where[1])
                return json.loads(f.readline())
        except (OSError, ValueError):
            return None


def _status(ref, other) -> str:
    if ref is None or other is None:
        return "unknown"
    if ref == other:
        return "both_correct" if ref else "both_wrong"
    return "flip_to_wrong" if ref else "flip_to_right"


def _b(x):
    return None if x is None else int(bool(x))


class RawStore:
    """lm-eval raw samples under one directory: discovery, per-condition parsing and joins across conditions."""

    def __init__(self, root: Path, results_dir: Path | None = None, ttl: float = 15.0, max_cached: int = 96):
        self.root, self.results_dir = Path(root), results_dir
        self.ttl, self.max_cached = ttl, max_cached
        self._layout: dict = {}
        self._scanned_at = -1e9
        self._cache: OrderedDict = OrderedDict()
        self._headlines: dict = {}
        self._scan_lock, self._load_lock = threading.Lock(), threading.Lock()

    # discovery ---------------------------------------------------------------------------------------------

    def layout(self, refresh: bool = False) -> dict:
        """condition -> task -> {subtask: latest samples file}. `<name>_mmlu` directories merge into `<name>`."""
        with self._scan_lock:
            if refresh or time.monotonic() - self._scanned_at > self.ttl:
                self._layout, self._scanned_at = self._scan(), time.monotonic()
            return self._layout

    def _scan(self) -> dict:
        found: dict = {}
        for cdir in _subdirs(self.root):
            cond = cdir.name[: -len("_mmlu")] if cdir.name.endswith("_mmlu") else cdir.name
            for tdir in _subdirs(cdir):
                files = latest_samples(tdir)
                if not files:
                    continue
                prev = found.setdefault(cond, {}).get(tdir.name)
                newest = lambda fs: max(_stamp(p, SAMPLES_RE) for p in fs.values())  # noqa: E731
                if prev is None or newest(files) > newest(prev):  # two runs of one task: keep the most recent
                    found[cond][tdir.name] = files
        return {c: t for c, t in found.items() if t}

    def conditions(self, refresh: bool = False) -> dict:
        layout = self.layout(refresh)
        conds = []
        for cond, tasks in layout.items():
            d = describe(cond)
            d["tasks"] = {t: self._headline(cond, t, files) for t, files in sorted(tasks.items(), key=lambda x: task_rank(x[0]))}
            conds.append(d)
        conds.sort(key=lambda d: (d["_rank"], d["id"]))
        tasks = sorted({t for d in conds for t in d["tasks"]}, key=task_rank)
        return {"raw_dir": str(self.root), "exists": self.root.is_dir(),
                "tasks": [{"id": t, "label": TASK_LABELS.get(t, t)} for t in tasks],
                "conditions": [public(d) for d in conds]}

    def _headline(self, cond: str, task: str, files: dict) -> dict:
        """Aggregate accuracy of a run, from lm-eval's results_*.json (or results/phase0/eval as a fallback)."""
        out = {"subtasks": len(files), "acc": None, "metric": None}
        dirs = {p.parent for p in files.values()}
        results = [r for d in dirs for r in d.glob("results_*.json")]
        summary = None
        if results:
            path = max(results, key=lambda p: _stamp(p, RESULTS_RE))
            ck = (str(path), _mtime(path))
            if ck not in self._headlines:
                data = _read_json(path) or {}
                self._headlines[ck] = (data.get("results") or {}).get(task)
            summary = self._headlines[ck]
        if isinstance(summary, dict):
            for metric in ("exact_match,strict-match", "acc,none", "acc_norm,none", "exact_match,flexible-extract"):
                if isinstance(summary.get(metric), (int, float)):
                    return {**out, "acc": summary[metric], "metric": metric}
        if self.results_dir:
            for name in (cond, cond + "_mmlu"):
                r = ((_read_json(Path(self.results_dir) / "phase0" / "eval" / f"{name}.json") or {}).get("results") or {}).get(task)
                if isinstance(r, dict) and isinstance(r.get("value"), (int, float)):
                    return {**out, "acc": r["value"], "metric": r.get("metric")}
        return out

    def task_data(self, cond: str, task: str) -> TaskData | None:
        files = self.layout().get(cond, {}).get(task)
        if not files:
            return None
        sig = _signature(files)
        with self._load_lock:
            td = self._cache.get((cond, task))
            if td is None or td.signature != sig:
                td = TaskData(task, files)
                self._cache[(cond, task)] = td
            self._cache.move_to_end((cond, task))
            while len(self._cache) > self.max_cached:
                self._cache.popitem(last=False)
            return td

    # joins -------------------------------------------------------------------------------------------------

    def _prepare(self, task: str, conds: list[str], filt):
        conds = [c for c in dict.fromkeys(conds) if c]
        datas = [self.task_data(c, task) for c in conds]
        avail = [i for i, d in enumerate(datas) if d is not None and d.docs]
        variants = _ordered(set().union(*(datas[i].scores for i in avail))) if avail else []
        if variants and filt not in variants:
            filt = variants[0]
        meta = [{**public(describe(c)), "available": i in avail} for i, c in enumerate(conds)]
        return conds, datas, avail, variants, filt, meta

    def items(self, task: str, conds: list[str], filt=None, q: str = "", status: str = "all", compare=None,
              subject=None, sort: str = "doc") -> dict:
        conds, datas, avail, variants, filt, meta = self._prepare(task, conds, filt)
        out = {"task": task, "label": TASK_LABELS.get(task, task), "kind": None, "filter": filt, "filters": variants,
               "conditions": meta, "reference": None, "compare": None, "n_total": 0, "n_unpaired": 0, "counts": {},
               "subjects": [], "sorts": ["doc"], "sort": "doc", "summary": {"n": 0, "accuracy": [None] * len(conds)},
               "items": []}
        if not avail:
            return out
        ref = avail[0]
        cmp = conds.index(compare) if compare in conds else None
        if cmp not in avail or cmp == ref:
            cmp = avail[1] if len(avail) > 1 else None
        kind = datas[ref].kind
        score = [datas[i].scores.get(filt, {}) if i in avail else {} for i in range(len(conds))]
        keys = set(score[ref])
        for i in avail[1:]:
            keys &= set(score[i])
        union = set().union(*(score[i] for i in avail))
        docs = datas[ref].docs

        def sort_key(k):
            d = docs[k]
            return (d.get("subject") or "", d["doc_id"] if isinstance(d["doc_id"], int) else 0, k)

        rows = []
        needle = q.strip().lower()
        for key in sorted(keys, key=sort_key):
            doc = docs[key]
            c = [_b(score[i][key][0]) if i in avail else None for i in range(len(conds))]
            a = [score[i][key][1] if i in avail else None for i in range(len(conds))]
            if needle:
                hay = [doc["question"], doc.get("gold") or "", *(x or "" for x in a)]
                if kind == "generative":
                    hay += [datas[i].gen.get(key, "") for i in avail]
                else:
                    hay += doc.get("choices") or []
                if needle not in "\n".join(hay).lower():
                    continue
            st = _status(c[ref], c[cmp]) if cmp is not None else ("unknown" if c[ref] is None else ("correct" if c[ref] else "wrong"))
            rows.append({"key": key, "doc_id": doc["doc_id"], "subject": doc.get("subject"), "q": _snippet(doc["question"]),
                         "gold": doc.get("gold", ""), "c": c, "a": a, "status": st,
                         "d": sum(1 for i in avail if c[i] != c[ref])})
        subjects: dict = {}
        for r in rows:
            if r["subject"]:
                subjects[r["subject"]] = subjects.get(r["subject"], 0) + 1
        if subject:
            rows = [r for r in rows if r["subject"] == subject]
        counts: dict = {}
        for r in rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        if status and status != "all":
            rows = [r for r in rows if r["status"] == status]

        sorts = ["doc", "disagree"] + (["length"] if kind == "generative" else ["gold_drop"])
        if sort == "disagree":
            rows.sort(key=lambda r: -r["d"])
        elif sort == "length" and kind == "generative":
            src = datas[cmp if cmp is not None else ref].gen
            rows.sort(key=lambda r: -len(src.get(r["key"], "")))
        elif sort == "gold_drop" and kind == "choice" and cmp is not None:
            def drop(r):
                g = docs[r["key"]].get("gold_index")
                la, lb = datas[ref].ll.get(r["key"], []), datas[cmp].ll.get(r["key"], [])
                if g is None or g >= min(len(la), len(lb)) or la[g] is None or lb[g] is None:
                    return math.inf
                return lb[g] - la[g]
            rows.sort(key=drop)
        else:
            sort = "doc"

        acc = []
        for i in range(len(conds)):
            vals = [r["c"][i] for r in rows if r["c"][i] is not None]
            acc.append(sum(vals) / len(vals) if vals else None)
        flips = {s: sum(1 for r in rows if r["status"] == s) for s in ("flip_to_wrong", "flip_to_right")}
        out.update(kind=kind, reference=conds[ref], compare=conds[cmp] if cmp is not None else None,
                   n_total=len(keys), n_unpaired=len(union) - len(keys), counts=counts,
                   subjects=[{"name": k, "n": v} for k, v in sorted(subjects.items())], sorts=sorts, sort=sort,
                   summary={"n": len(rows), "accuracy": acc, **flips}, items=rows)
        return out

    def item(self, task: str, key: str, conds: list[str], filt=None):
        conds, datas, avail, variants, filt, meta = self._prepare(task, conds, filt)
        have = [datas[i] for i in avail if key in datas[i].docs]
        if not have:
            return None
        base = have[0]
        doc = base.docs[key]
        row = base.read_row(key) or {}
        prompt = _prompt(row)
        prefix, query = split_prompt(prompt, doc["question"])
        out = {"task": task, "label": TASK_LABELS.get(task, task), "kind": base.kind, "key": key, **doc,
               "prompt_prefix": prefix, "prompt_query": query, "filter": filt, "filters": variants, "conditions": []}
        for i, c in enumerate(conds):
            d = datas[i]
            entry = {**meta[i], "present": bool(d is not None and key in d.docs)}
            if entry["present"]:
                variants_here = {v: {"correct": s[key][0], "answer": s[key][1]} for v, s in d.scores.items() if key in s}
                cur = variants_here.get(filt, {"correct": None, "answer": ""})
                entry.update(correct=cur["correct"], answer=cur["answer"], variants=variants_here)
                if base.kind == "generative":
                    entry["generation"] = d.gen.get(key, "")
                else:
                    entry["ll"] = d.ll.get(key, [])
            out["conditions"].append(entry)
        return out


# ---------------------------------------------------------------------------------------------------------------
# Tokenizer (Gemma 3), used only to turn token ids back into text
# ---------------------------------------------------------------------------------------------------------------

BYTE_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


class TokenDecoder:
    """Lazy Gemma 3 tokenizer with a per-id decode cache. When it cannot load, pieces() falls back to ids."""

    def __init__(self, env_file: Path, model: str = "google/gemma-3-4b-pt"):
        self.env_file, self.model = env_file, model
        self._tok = None
        self._special: set = set()
        self._cache: dict = {}
        self.error = None
        self._lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.get, daemon=True).start()

    def status(self) -> dict:
        return {"ok": self._tok is not None, "loading": self._tok is None and self.error is None, "error": self.error}

    def get(self):
        with self._lock:
            if self._tok is None and self.error is None:
                self._tok, self.error = self._load()
                if self._tok is not None:
                    self._special = set(self._tok.all_special_ids)
            return self._tok

    def _load(self):
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        try:
            from transformers import AutoTokenizer
            from transformers.utils import logging as hf_logging
            hf_logging.set_verbosity_error()
        except Exception as e:  # noqa: BLE001
            return None, f"transformers is not importable ({type(e).__name__})"
        token = os.environ.get("HF_TOKEN")
        if not token:
            try:
                from dotenv import dotenv_values
                token = dotenv_values(self.env_file).get("HF_TOKEN")
            except Exception:  # noqa: BLE001
                token = None
        try:  # the cached copy first, so the browser works offline once the tokenizer has been downloaded
            return AutoTokenizer.from_pretrained(self.model, token=token, local_files_only=True), None
        except Exception:  # noqa: BLE001
            pass
        try:
            return AutoTokenizer.from_pretrained(self.model, token=token), None
        except Exception as e:  # noqa: BLE001
            msg = (str(e).strip().splitlines() or [type(e).__name__])[0][:240]
            if token:
                msg = msg.replace(token, "***")
            return None, f"{type(e).__name__}: {msg}"

    def pieces(self, ids: list[int]) -> tuple[list[str], list[int]]:
        """Display text and a special-token flag per id. Runs of byte-fallback tokens are decoded together; the
        character goes to the first token of the run and the rest get an empty string."""
        tok = self.get()
        if tok is None:
            return [f"[{i}] " for i in ids], [0] * len(ids)
        text, special = [], []
        run: list[int] = []  # positions of the current byte-fallback run

        def flush():
            if run:
                raw = bytes(self._cache[ids[j]][1] for j in run).decode("utf-8", errors="replace")
                text[run[0]] = raw
                run.clear()

        for j, i in enumerate(ids):
            info = self._cache.get(i)
            if info is None:
                piece = tok.convert_ids_to_tokens(int(i)) or ""
                m = BYTE_RE.match(piece)
                if m:
                    info = ("byte", int(m[1], 16))
                elif i in self._special:
                    info = ("special", piece)
                else:
                    info = ("text", tok.decode([int(i)]))
                self._cache[i] = info
            if info[0] == "byte":
                run.append(j)
                text.append("")
                special.append(0)
                continue
            flush()
            text.append(info[1])
            special.append(1 if info[0] == "special" else 0)
        flush()
        return text, special


# ---------------------------------------------------------------------------------------------------------------
# Per-token KL dumps
# ---------------------------------------------------------------------------------------------------------------

PROFILE_BINS = 240
MAX_WINDOW = 16384


def _sig(values, digits: int = 3) -> list:
    """Floats rounded to a few significant digits (keeps the JSON small); non-finite -> None."""
    return [float(f"{v:.{digits}g}") if math.isfinite(v) else None for v in values]


def _codes(column):
    """(int codes, category names) for a string or dictionary column."""
    import pyarrow as pa
    import pyarrow.compute as pc
    arr = column.combine_chunks() if hasattr(column, "combine_chunks") else column
    if not pa.types.is_dictionary(arr.type):
        arr = pc.dictionary_encode(arr.cast(pa.string()))
    names = [str(x) for x in arr.dictionary.to_pylist()]
    codes = arr.indices.to_numpy(zero_copy_only=False).astype(np.int16)
    return codes, names


class Dump:
    """One parquet in memory. Documents are (source, doc_id) groups; a group can hold several packed sequences
    (positions restart at 1 in each), up to a few hundred thousand tokens for PG-19."""

    def __init__(self, path: Path):
        import pyarrow.parquet as pq
        t = pq.read_table(path).unify_dictionaries()
        num = lambda n: t.column(n).to_numpy()  # noqa: E731
        self.kl = num("kl").astype(np.float32)
        self.agree = t.column("top1_agree").to_numpy().astype(bool)
        self.ent = num("entropy_a").astype(np.float32)
        self.target = num("target").astype(np.int64)
        self.pos = num("position").astype(np.int64)
        src, doc = num("source").astype(np.int64), num("doc_id").astype(np.int64)
        self.cls, self.classes = _codes(t.column("token_class"))
        key = (src << 40) + doc
        starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]]) if len(key) else np.zeros(0, np.int64)
        if len(starts) != len(np.unique(key)):  # a document split over several runs: group its rows together
            order = np.argsort(key, kind="stable")
            for name in ("kl", "agree", "ent", "target", "pos", "cls"):
                setattr(self, name, getattr(self, name)[order])
            src, doc, key = src[order], doc[order], key[order]
            starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        self.starts = starts
        self.ends = np.r_[starts[1:], len(key)].astype(np.int64)
        self.n = self.ends - self.starts
        self.src, self.doc = src[starts], doc[starts]
        if len(starts):
            self.mean = np.add.reduceat(self.kl.astype(np.float64), starts) / self.n
            self.max = np.maximum.reduceat(self.kl, starts)
            self.flip = np.add.reduceat((~self.agree).astype(np.int64), starts) / self.n
            seq_start = np.r_[True, self.pos[1:] <= self.pos[:-1]]
            seq_start[starts] = True
            self.seq_start = seq_start
            self.n_seq = np.add.reduceat(seq_start.astype(np.int64), starts)
        else:
            self.mean = self.max = self.flip = np.zeros(0)
            self.seq_start = np.zeros(0, bool)
            self.n_seq = np.zeros(0, np.int64)
        self.index = {(int(s), int(d)): g for g, (s, d) in enumerate(zip(self.src, self.doc))}


def _split_dump_name(name: str, summary: dict) -> tuple[str, str]:
    args = (summary or {}).get("args") or {}
    cond, tokens = args.get("name"), args.get("tokens")
    if isinstance(cond, str) and isinstance(tokens, str) and name == f"{cond}_{Path(tokens).stem}":
        return cond, Path(tokens).stem
    cond, _, rest = name.partition("_")
    return cond, rest or name


def _source_names(kl_set_yaml: Path) -> dict:
    """token set -> source names, in the order that defines the `source` column (configs/data/kl_set.yaml)."""
    try:
        import yaml
        cfg = yaml.safe_load(kl_set_yaml.read_text())
        return {k: list((v or {}).get("sources") or {}) for k, v in (cfg.get("sets") or {}).items()}
    except Exception:  # noqa: BLE001
        return {}


class DumpStore:
    """All token dumps in one directory, each parquet loaded once on first use and kept in memory."""

    def __init__(self, root: Path, decoder: TokenDecoder, kl_set_yaml: Path | None = None):
        self.root, self.decoder = Path(root), decoder
        self.sources = _source_names(kl_set_yaml) if kl_set_yaml else {}
        self._dumps: dict = {}
        self._lock = threading.Lock()

    def _meta(self, name: str) -> dict:
        summary = _read_json(self.root / f"{name}_summary.json") or {}
        cond, set_name = _split_dump_name(name, summary)
        d = describe(cond)
        keep = ("n_tokens", "mean_kl", "median_kl", "p90_kl", "top1_agree", "by_class", "by_position")
        return {"name": name, "cond": cond, "set": set_name, "cond_name": d["name"],
                "set_name": SET_LABELS.get(set_name, set_name), "label": f"{d['name']} · {SET_LABELS.get(set_name, set_name)}",
                "summary": clean({k: summary.get(k) for k in keep if k in summary})}

    def list(self) -> dict:
        names = sorted(p.stem for p in self.root.glob("*.parquet")) if self.root.is_dir() else []
        dumps = [{**self._meta(n), "loaded": n in self._dumps} for n in names]
        dumps.sort(key=lambda d: (d["set"], describe(d["cond"])["_rank"], d["name"]))
        return {"dir": str(self.root), "dumps": dumps, "tokenizer": self.decoder.status()}

    def get(self, name: str) -> Dump | None:
        path = self.root / f"{name}.parquet"
        if "/" in name or "\\" in name or not path.is_file():
            return None
        with self._lock:  # reload when the file changed on disk
            mtime = _mtime(path)
            if name not in self._dumps or self._dumps[name][0] != mtime:
                self._dumps[name] = (mtime, Dump(path))
            return self._dumps[name][1]

    def _source_name(self, name: str, source: int) -> str:
        set_name = self._meta(name)["set"]
        names = self.sources.get(set_name) or []
        return names[source] if 0 <= source < len(names) else f"source {source}"

    def docs(self, name: str, sort: str = "mean_kl", limit: int = 200, offset: int = 0, source=None):
        d = self.get(name)
        if d is None:
            return None
        idx = np.arange(len(d.starts))
        if source is not None:
            idx = idx[d.src[idx] == source]
        keys = {"mean_kl": -d.mean, "max_kl": -d.max, "flip_rate": -d.flip, "length": -d.n.astype(np.float64),
                "doc": (d.src << 40) + d.doc}
        if sort not in keys:
            sort = "mean_kl"
        order = idx[np.argsort(keys[sort][idx], kind="stable")]
        limit, offset = max(0, int(limit)), max(0, int(offset))
        sel = order[offset: offset + limit]
        names = {int(s): self._source_name(name, int(s)) for s in np.unique(d.src)}
        return {"name": name, "sort": sort, "n_docs": int(len(idx)), "offset": offset,
                "max_mean_kl": float(d.mean.max()) if len(d.mean) else 0.0,
                "sources": [{"id": k, "name": v, "n": int((d.src == k).sum())} for k, v in sorted(names.items())],
                "docs": [{"doc_id": int(d.doc[g]), "source": int(d.src[g]), "source_name": names[int(d.src[g])],
                          "n": int(d.n[g]), "n_seq": int(d.n_seq[g]), "mean_kl": float(d.mean[g]),
                          "max_kl": float(d.max[g]), "flip_rate": float(d.flip[g])} for g in sel]}

    def doc(self, name: str, doc_id: int, source: int = 0, start: int = 0, limit: int = 2048):
        d = self.get(name)
        if d is None:
            return None
        g = d.index.get((int(source), int(doc_id)))
        if g is None:
            return None
        s, e = int(d.starts[g]), int(d.ends[g])
        n = e - s
        limit = int(min(max(int(limit), 1), MAX_WINDOW))
        start = int(min(max(int(start), 0), max(n - 1, 0)))
        a, b = s + start, min(s + start + limit, e)
        text, special = self.decoder.pieces(d.target[a:b].tolist())
        seq = np.cumsum(d.seq_start[s:e]) - 1
        kl_doc = d.kl[s:e]
        bins = int(min(PROFILE_BINS, n))
        edges = (np.arange(bins, dtype=np.int64) * n) // bins
        widths = np.diff(np.r_[edges, n])
        profile_mean = np.add.reduceat(kl_doc.astype(np.float64), edges) / widths
        profile_max = np.maximum.reduceat(kl_doc, edges)
        return {
            "name": name, "doc_id": int(doc_id), "source": int(source), "source_name": self._source_name(name, int(source)),
            "n": n, "n_seq": int(d.n_seq[g]), "start": start, "end": start + (b - a), "classes": d.classes,
            "seq_starts": np.flatnonzero(d.seq_start[s:e]).tolist(),
            "stats": {"mean_kl": float(d.mean[g]), "max_kl": float(d.max[g]), "flip_rate": float(d.flip[g]),
                      "argmax": int(np.argmax(kl_doc)) if n else 0},
            "tokenizer": self.decoder.status(),
            "tokens": {"text": text, "id": d.target[a:b].tolist(), "kl": _sig(d.kl[a:b].tolist()),
                       "ent": _sig(d.ent[a:b].tolist()), "agree": d.agree[a:b].astype(np.int8).tolist(),
                       "cls": d.cls[a:b].tolist(), "pos": d.pos[a:b].tolist(), "seq": seq[start: start + (b - a)].tolist(),
                       "special": special},
            "profile": {"mean": _sig(profile_mean.tolist()), "max": _sig(profile_max.astype(np.float64).tolist()),
                        "edges": edges.tolist()},
        }
