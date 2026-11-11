"""Shared, deterministic parsing and metrics.

One parser and one set of metrics are used for every model, so the only thing
that varies between runs is the model (and, in ablations, the prompt).
"""
from __future__ import annotations

import ast
import re
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Detection: binary label parsing
# --------------------------------------------------------------------------- #
POSITIVE = "Contradiction exists"
NEGATIVE = "Claims can coexist"


def parse_detection_conclusion(raw_text: str) -> str | None:
    """Extract the binary label from a model's ``<conclusion>`` block.

    Falls back to the whole response if no ``<conclusion>`` tag is present.
    Returns ``None`` when neither label can be recognised.
    """
    if not isinstance(raw_text, str):
        return None
    m = re.search(r"<conclusion>(.*?)</conclusion>", raw_text, re.DOTALL)
    text = (m.group(1) if m else raw_text).strip().lower()
    if "contradiction" in text and "coexist" not in text:
        return POSITIVE
    if "coexist" in text or "no contradiction" in text:
        return NEGATIVE
    return None


def binary_prf(gold: Iterable[str], pred: Iterable[str], positive: str = POSITIVE) -> dict:
    """Precision / recall / F1 / accuracy for the positive (contradiction) class."""
    gold = list(gold)
    pred = list(pred)
    tp = sum(1 for g, p in zip(gold, pred) if g == positive and p == positive)
    fp = sum(1 for g, p in zip(gold, pred) if g != positive and p == positive)
    fn = sum(1 for g, p in zip(gold, pred) if g == positive and p != positive)
    tn = sum(1 for g, p in zip(gold, pred) if g != positive and p != positive)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(gold) if gold else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_pos": sum(1 for g in gold if g == positive), "n": len(gold),
    }


def bootstrap_f1_ci(gold, pred, n_boot: int = 1000, seed: int = 0, positive: str = POSITIVE):
    """95% bootstrap confidence interval for the positive-class F1."""
    gold = np.array(list(gold), dtype=object)
    pred = np.array(list(pred), dtype=object)
    rng = np.random.default_rng(seed)
    n = len(gold)
    f1s = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        f1s.append(binary_prf(gold[idx], pred[idx], positive)["f1"])
    lo, hi = np.percentile(f1s, [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Explanation: multi-label contextual-factor parsing
# --------------------------------------------------------------------------- #
CANONICAL_FACTORS = [
    "species", "population", "dosage or exposure duration",
    "route or mode of administration", "combined drug effects",
    "evolving scientific evidence", "known controversy or self-qualified claims",
    "study design", "outcome measures", "ambiguous referent",
]

# Short codes used in the paper tables (Table C1).
FACTOR_CODES = {
    "outcome measures": "OM",
    "population": "POP",
    "dosage or exposure duration": "DOS",
    "study design": "SD",
    "ambiguous referent": "AMB",
    "combined drug effects": "CDE",
    "route or mode of administration": "RTE",
    "known controversy or self-qualified claims": "KC",
    "evolving scientific evidence": "ESE",
    "species": "SPE",
}

_ALIASES = {
    "dose or exposure duration": "dosage or exposure duration",
    "duration or exposure duration": "dosage or exposure duration",
    "dosage/exposure duration": "dosage or exposure duration",
    "combined drug effect": "combined drug effects",
    "known controversy": "known controversy or self-qualified claims",
    "self-qualified claims": "known controversy or self-qualified claims",
    "route or mode administration": "route or mode of administration",
    # Descriptive-prompt taxonomy variants -> gold canonical labels
    "ambiguous expression": "ambiguous referent",
    "physiological context": "population",  # closest gold factor; no gold "physiological" label
}


def normalize_factors(value) -> list[str]:
    """Normalize a gold list-string OR a raw model response into canonical labels.

    Accepts a Python list, a list literal string (``"['population', ...]"``), or
    a raw response containing ``contextual_factor: a, b`` inside a
    ``<conclusion>`` block.
    """
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    elif value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        # Isolate the <conclusion> block so the <explanation> body never bleeds
        # into the factor list.
        conc = re.search(r"<conclusion>(.*?)</conclusion>", text, flags=re.IGNORECASE | re.DOTALL)
        if conc:
            text = conc.group(1).strip()
        # Take the list after 'contextual_factor:' up to the first newline or tag.
        m = re.search(r"contextual_factor\s*:\s*([^\n<]*)", text, flags=re.IGNORECASE)
        if m:
            text = m.group(1).strip()
        if text.lower() in {"none", "[none]", ""}:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = re.split(r"\s*(?:\||;|,|\n)\s*", text.strip("[](){}"))
        items = parsed if isinstance(parsed, (list, tuple, set)) else [parsed]
    else:
        return []
    out = []
    for item in items:
        label = str(item).strip().strip("[](){}'\"`.- ").lower()
        label = re.sub(r"\s+", " ", label)
        label = _ALIASES.get(label, label)
        if label and label != "none":
            out.append(label)
    return out


def multilabel_metrics(gold_lists, pred_lists, factors: list[str] | None = None) -> pd.DataFrame:
    """Per-factor P/R/F1 plus ``MICRO`` and ``MACRO`` summary rows."""
    factors = factors or CANONICAL_FACTORS
    gold_lists = [normalize_factors(g) for g in gold_lists]
    pred_lists = [normalize_factors(p) for p in pred_lists]
    rows = []
    micro_tp = micro_fp = micro_fn = 0
    macro_f1s = []
    for f in factors:
        tp = sum(1 for g, p in zip(gold_lists, pred_lists) if f in g and f in p)
        fp = sum(1 for g, p in zip(gold_lists, pred_lists) if f not in g and f in p)
        fn = sum(1 for g, p in zip(gold_lists, pred_lists) if f in g and f not in p)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        support = sum(1 for g in gold_lists if f in g)
        rows.append({"factor": f, "precision": prec, "recall": rec, "f1": f1,
                     "support": support, "low_support": support < 40})
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        macro_f1s.append(f1)
    mp = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    mr = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    mf = 2 * mp * mr / (mp + mr) if (mp + mr) else 0.0
    rows.append({"factor": "MICRO", "precision": mp, "recall": mr, "f1": mf,
                 "support": sum(len(g) for g in gold_lists), "low_support": False})
    rows.append({"factor": "MACRO", "precision": np.nan, "recall": np.nan,
                 "f1": float(np.mean(macro_f1s)), "support": np.nan, "low_support": False})
    return pd.DataFrame(rows)


def gold_factor_distribution(gold_lists) -> Counter:
    c: Counter = Counter()
    for g in gold_lists:
        for f in normalize_factors(g):
            c[f] += 1
    return c


def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table (no extra deps)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in (r[c] for c in cols)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
