"""Score every run in results/runs against the frozen eval sets.

    python scripts/score_all.py --task detection
    python scripts/score_all.py --task explanation
    python scripts/score_all.py --task all --predictions   # also dump per-row predictions

Writes to results/tables/:
  runs_<task>.csv               one row per run file (unaggregated metrics)
  comparison_<task>.csv         mean ± std over repeat runs, per (model, prompt, temperature)
  comparison_<task>_pooled.csv  mean ± std over all runs and temperatures, per (model, prompt)
  perfactor/<model>__<prompt>__<temp>.csv   per-factor P/R/F1 (explanation only)
  predictions_<task>.csv.gz     per-row parsed predictions (only with --predictions)
"""
from __future__ import annotations

import argparse

import pandas as pd

from drugcon import evaluation as ev
from drugcon import paths
from drugcon.runs import iter_run_files, load_eval_set, load_run


def score_detection(with_predictions: bool):
    eval_df = load_eval_set("detection")
    rows, preds = [], []
    for path, meta in iter_run_files("detection"):
        df = load_run(path, "detection", eval_df)
        parsed = df["raw_response"].map(ev.parse_detection_conclusion)
        unparsed = int(parsed.isna().sum())
        pred = parsed.fillna(ev.NEGATIVE)  # unparsed -> negative class
        m = ev.binary_prf(df["gold_label"], pred)
        rows.append({"model": meta["model"], "prompt": meta["prompt"], "temperature": meta["temp"],
                     "run": int(meta["run"]), **{k: m[k] for k in
                     ["precision", "recall", "f1", "accuracy", "tp", "fp", "fn", "tn", "n"]},
                     "unparsed": unparsed, "api_errors": int(df["api_error"].notna().sum())})
        if with_predictions:
            p = df[["pair_id", "gold_label"]].copy()
            p["prediction"] = pred.values
            p["correct"] = p["gold_label"] == p["prediction"]
            p["model"], p["prompt"], p["temperature"], p["run"] = (
                meta["model"], meta["prompt"], meta["temp"], int(meta["run"]))
            preds.append(p)
    return pd.DataFrame(rows), preds, ["precision", "recall", "f1", "accuracy"]


def score_explanation(with_predictions: bool):
    eval_df = load_eval_set("explanation")
    perfactor_dir = paths.TABLES_DIR / "perfactor"
    perfactor_dir.mkdir(parents=True, exist_ok=True)
    rows, preds, perfactor = [], [], {}
    for path, meta in iter_run_files("explanation"):
        df = load_run(path, "explanation", eval_df)
        table = ev.multilabel_metrics(df["human_labeled_contextual_factors"], df["raw_response"])
        micro = table.set_index("factor").loc["MICRO"]
        macro = table.set_index("factor").loc["MACRO"]
        rows.append({"model": meta["model"], "prompt": meta["prompt"], "temperature": meta["temp"],
                     "run": int(meta["run"]), "micro_precision": micro.precision,
                     "micro_recall": micro.recall, "micro_f1": micro.f1, "macro_f1": macro.f1,
                     "n": len(df), "api_errors": int(df["api_error"].notna().sum())})
        perfactor.setdefault((meta["model"], meta["prompt"], meta["temp"]), []).append(table)
        if with_predictions:
            p = df[["id"]].copy()
            p["gold_factors"] = [ev.normalize_factors(g) for g in df["human_labeled_contextual_factors"]]
            p["predicted_factors"] = [ev.normalize_factors(r) for r in df["raw_response"]]
            p["model"], p["prompt"], p["temperature"], p["run"] = (
                meta["model"], meta["prompt"], meta["temp"], int(meta["run"]))
            preds.append(p)

    for (model, prompt, temp), tables in perfactor.items():
        avg = (pd.concat(tables).groupby("factor", as_index=False, sort=False)
               .agg(precision=("precision", "mean"), recall=("recall", "mean"),
                    f1=("f1", "mean"), support=("support", "mean")))
        avg.to_csv(perfactor_dir / f"{model}__{prompt}__{temp}.csv", index=False)
    return pd.DataFrame(rows), preds, ["micro_precision", "micro_recall", "micro_f1", "macro_f1"]


def aggregate(runs: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    """Mean ± sample std across runs, with raw mean/std columns kept for reuse."""
    g = runs.groupby(keys)
    out = g[metrics].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out = out.reset_index()
    for m in metrics:
        std = out[f"{m}_std"].fillna(0.0)
        out[m] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(out[f"{m}_mean"], std)]
    out["n_runs"] = g.size().values
    return out[keys + metrics + ["n_runs"] + [f"{m}_{s}" for m in metrics for s in ("mean", "std")]]


def run_task(task: str, with_predictions: bool) -> None:
    runs, preds, metrics = (score_detection if task == "detection" else score_explanation)(with_predictions)
    if runs.empty:
        print(f"No {task} run files found in {paths.RUNS_DIR}.")
        return
    paths.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    runs = runs.sort_values(["model", "prompt", "temperature", "run"]).reset_index(drop=True)
    runs.to_csv(paths.TABLES_DIR / f"runs_{task}.csv", index=False)

    by_temp = aggregate(runs, ["model", "prompt", "temperature"], metrics)
    pooled = aggregate(runs, ["model", "prompt"], metrics)
    by_temp.to_csv(paths.TABLES_DIR / f"comparison_{task}.csv", index=False)
    pooled.to_csv(paths.TABLES_DIR / f"comparison_{task}_pooled.csv", index=False)

    print(f"\n=== {task}: mean ± std per (model, prompt, temperature) ===")
    print(ev.to_markdown(by_temp[["model", "prompt", "temperature"] + metrics + ["n_runs"]]))
    if with_predictions and preds:
        out = paths.TABLES_DIR / f"predictions_{task}.csv.gz"
        pd.concat(preds, ignore_index=True).to_csv(out, index=False)
        print(f"Wrote {out}")
    print(f"\nWrote tables to {paths.TABLES_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["detection", "explanation", "all"], required=True)
    ap.add_argument("--predictions", action="store_true", help="also write per-row predictions")
    args = ap.parse_args()
    for task in (["detection", "explanation"] if args.task == "all" else [args.task]):
        run_task(task, args.predictions)


if __name__ == "__main__":
    main()
