"""Auto-interp labels and fixed-taxonomy categorization for SAE features (plan E2.3).

explain_feature: top activating examples -> one-sentence explanation.
categorize: explanation -> one of CATEGORIES. Both take any LLMClient (Backboard in practice, a stub in tests).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

CATEGORIES = (
    "numeral_arithmetic",
    "named_entity_fact",
    "syntax_grammar",
    "code",
    "formatting_structure",
    "multilingual",
    "positional_attention_sink",
    "dialogue_instruction",
    "other",
)

_EXPLAIN_SYSTEM = (
    "You label features of a sparse autoencoder trained on a language model. You will see text snippets where one "
    "feature fires; the token where it fires most is marked with << >> and its activation strength is given. "
    "Reply with ONE sentence describing what the feature responds to. Be specific; do not hedge."
)

_CATEGORIZE_SYSTEM = (
    "Classify a one-sentence description of a language-model feature into exactly one category from this list: "
    + ", ".join(CATEGORIES)
    + ". Reply with the category name only."
)


@dataclass
class Example:
    tokens: list[str]
    activations: list[float]

    def render(self, window: int = 24) -> str:
        peak = max(range(len(self.activations)), key=lambda i: self.activations[i])
        lo, hi = max(0, peak - window), min(len(self.tokens), peak + window)
        parts = [f"<<{t}>>" if i == peak else t for i, t in enumerate(self.tokens[lo:hi], start=lo)]
        return f"(peak {self.activations[peak]:.2f}) " + "".join(parts).replace("\n", "\\n")


def explain_feature(client, examples: list[Example], max_examples: int = 20) -> str:
    body = "\n".join(f"{i + 1}. {ex.render()}" for i, ex in enumerate(examples[:max_examples]))
    return client.complete(f"Examples:\n{body}\n\nOne-sentence explanation:", system=_EXPLAIN_SYSTEM).strip()


def categorize(client, explanation: str) -> str:
    raw = client.complete(f"Description: {explanation}\n\nCategory:", system=_CATEGORIZE_SYSTEM).strip().lower()
    raw = re.sub(r"[^a-z_]", "", raw.replace(" ", "_").replace("-", "_"))
    for c in CATEGORIES:
        if raw == c or raw.startswith(c) or c in raw:
            return c
    return "other"


def enrichment_table(top_categories: list[str], control_categories: list[str]) -> dict:
    """Category share among top-shifted features over share among a random control, with Fisher's exact test."""
    from collections import Counter

    top, ctl = Counter(top_categories), Counter(control_categories)
    n_top, n_ctl = len(top_categories), len(control_categories)
    out = {}
    for c in CATEGORIES:
        a, b = top[c], n_top - top[c]
        cc, d = ctl[c], n_ctl - ctl[c]
        share_top, share_ctl = a / max(n_top, 1), cc / max(n_ctl, 1)
        try:
            from scipy.stats import fisher_exact
            p = float(fisher_exact([[a, b], [cc, d]], alternative="greater")[1])
        except ImportError:
            p = None
        out[c] = {"n_top": a, "n_control": cc, "share_top": share_top, "share_control": share_ctl,
                  "enrichment": (share_top / share_ctl) if share_ctl > 0 else None, "p_greater": p}
    return out


def dump(path, rows: list[dict]):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
