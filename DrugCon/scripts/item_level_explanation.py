"""Item-level (per-pair) explanation statistics for every explanation run.

score_all.py reports label-level micro/macro-F1. This script scores each gold
pair individually (gold set G, predicted set P) and averages over pairs:

  exact_match   P == G
  any_hit       |G & P| > 0
  jaccard       |G & P| / |G | P|
  item_f1       2|G & P| / (|G| + |P|)   (example-based F1)
  item_precision, item_recall, empty_pred, predicted_n

    python scripts/item_level_explanation.py                  # all prompts
    python scripts/item_level_explanation.py --prompt taxonomy_v1

Writes results/tables/item_level_explanation_summary.csv and, with --per-pair,
results/tables/item_level_explanation_perpair.csv.gz.
"""
from __future__ import annotations

import argparse

import pandas as pd

from drugcon import evaluation as ev
from drugcon import paths
from drugcon.runs import iter_run_files, load_eval_set, load_run


def per_pair(gold_lists, pred_lists) -> pd.DataFrame:
    rows = []
    for g_raw, p_raw in zip(gold_lists, pred_lists):
        g, p = set(ev.normalize_factors(g_raw)), set(ev.normalize_factors(p_raw))
        inter, union, denom = len(g & p), len(g | p), len(g) + len(p)
        rows.append({
            "gold_n": len(g), "predicted_n": len(p), "tp": inter,
            "fp": len(p - g), "fn": len(g - p),
            "exact_match": int(g == p),
            "any_hit": int(inter > 0),
            "jaccard": inter / union if union else 1.0,
            "item_f1": 2 * inter / denom if denom else 1.0,
            "item_precision": inter / len(p) if p else (1.0 if not g else 0.0),
            "item_recall": inter / len(g) if g else (1.0 if not p else 0.0),
            "empty_pred": int(not p),
            "gold_factors": sorted(g), "predicted_factors": sorted(p),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default="all", help="prompt version to include, or 'all'")
    ap.add_argument("--per-pair", action="store_true", help="also write the per-pair table")
    args = ap.parse_args()

    eval_df = load_eval_set("explanation")
    summaries, frames = [], []
    for path, meta in iter_run_files("explanation"):
        if args.prompt != "all" and meta["prompt"] != args.prompt:
            continue
        df = load_run(path, "explanation", eval_df)
        pairs = per_pair(df["human_labeled_contextual_factors"], df["raw_response"])
        summaries.append({
            "model": meta["model"], "prompt": meta["prompt"], "temperature": meta["temp"],
            "run": int(meta["run"]), "n_pairs": len(pairs),
            "exact_match_rate": pairs.exact_match.mean(),
            "any_hit_rate": pairs.any_hit.mean(),
            "mean_jaccard": pairs.jaccard.mean(),
            "mean_item_f1": pairs.item_f1.mean(),
            "mean_item_precision": pairs.item_precision.mean(),
            "mean_item_recall": pairs.item_recall.mean(),
            "empty_pred_rate": pairs.empty_pred.mean(),
            "mean_predicted_factors": pairs.predicted_n.mean(),
            "mean_gold_factors": pairs.gold_n.mean(),
        })
        if args.per_pair:
            pairs.insert(0, "id", df["id"].values)
            for i, k in enumerate(["model", "prompt", "temp", "run"]):
                pairs.insert(i, "temperature" if k == "temp" else k, meta[k])
            frames.append(pairs)

    if not summaries:
        print("No explanation run files found.")
        return
    summary = (pd.DataFrame(summaries)
               .sort_values(["prompt", "mean_item_f1"], ascending=[True, False])
               .reset_index(drop=True))
    paths.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = paths.TABLES_DIR / "item_level_explanation_summary.csv"
    summary.to_csv(out, index=False)
    print(ev.to_markdown(summary[["model", "prompt", "n_pairs", "exact_match_rate", "any_hit_rate",
                                  "mean_jaccard", "mean_item_f1", "empty_pred_rate",
                                  "mean_predicted_factors"]]))
    print(f"\nWrote {out}")
    if frames:
        pp = paths.TABLES_DIR / "item_level_explanation_perpair.csv.gz"
        pd.concat(frames, ignore_index=True).to_csv(pp, index=False)
        print(f"Wrote {pp}")


if __name__ == "__main__":
    main()
