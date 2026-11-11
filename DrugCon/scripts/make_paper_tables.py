"""Regenerate the paper's result tables from results/tables (run score_all.py first).

    python scripts/score_all.py --task all
    python scripts/make_paper_tables.py            # paper lineup (11 backbones)
    python scripts/make_paper_tables.py --all-models

Writes to results/tables/paper/:
  table2_drugcon_detection.{md,tex}   DrugCon (GPT-4.1-nano, CF prompt), mean ± std over temperatures
  table3_ablation.{md,tex}            backbone x prompt ablation (detection CF/DC + explanation micro-F1)
  tableC1_perfactor.{md,tex}          per-factor explanation F1 by backbone
  perfactor_long.csv / perfactor_f1_matrix.csv

Which run backs each cell is recorded in table3_ablation_sources.csv so every
number can be traced to a file in results/runs.
"""
from __future__ import annotations

import argparse

import pandas as pd

from drugcon import evaluation as ev
from drugcon import paths

# (slug, display name, tier). Order follows the paper.
PAPER_MODELS = [
    ("ministral-3b", "Ministral-3B", "≤7B"),
    ("gemma-3-4b", "Gemma-3-4B", "≤7B"),
    ("llama-3.1-8b", "Llama-3.1-8B", "7–20B"),
    ("gemma-3-12b", "Gemma-3-12B", "7–20B"),
    ("gemma-3-27b", "Gemma-3-27B", ">20B"),
    ("qwen3-32b", "Qwen3-32B", ">20B"),
    ("llama-3.3-70b", "Llama-3.3-70B", ">20B"),
    ("claude-haiku-4.5", "Claude-Haiku-4.5", "Proprietary"),
    ("claude-sonnet-4.5", "Claude-Sonnet-4.5", "Proprietary"),
    ("gpt-5.5", "GPT-5.5", "Proprietary"),
    ("gpt-4.1-nano", "GPT-4.1-nano", "Proprietary"),
]
EXTRA_MODELS = [
    ("mistral-7b", "Mistral-7B", "≤7B"),
    ("gpt-oss-20b", "GPT-OSS-20B", "7–20B"),
]
MAIN_MODEL = "gpt-4.1-nano"

CF_PROMPTS = ["counterfactual_appendix_v2", "counterfactual_appendix_v1"]  # preference order
DC_PROMPT = "drug_disease_claim_only_v2"
EXPL_PROMPTS = ["taxonomy_v1", "drugcon_pipeline_v1"]  # preference order
FACTOR_ORDER = ["outcome measures", "population", "dosage or exposure duration", "study design",
                "ambiguous referent", "combined drug effects", "route or mode of administration",
                "known controversy or self-qualified claims", "evolving scientific evidence", "species"]


def _load(name: str) -> pd.DataFrame:
    path = paths.TABLES_DIR / name
    if not path.exists():
        raise SystemExit(f"{path} not found; run `python scripts/score_all.py --task all` first.")
    return pd.read_csv(path)


def _is_t0(temp: str) -> bool:
    return str(temp).startswith("t0.0")


def _pick(df: pd.DataFrame, model: str, prompts: list[str], pooled: bool):
    """First matching (prompt, temperature) row for ``model``; t=0 unless pooled."""
    for prompt in prompts:
        sub = df[(df.model == model) & (df.prompt == prompt)]
        if not pooled and "temperature" in sub:
            t0 = sub[sub.temperature.map(_is_t0)]
            sub = t0 if not t0.empty else sub[sub.temperature == "default"]
        if not sub.empty:
            return sub.iloc[0]
    return None


def _fmt(x) -> str:
    return "–" if x is None or pd.isna(x) else f"{x:.3f}"


def _write(stem: str, md: str, tex: str) -> None:
    out = paths.TABLES_DIR / "paper"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.md").write_text(md + "\n")
    (out / f"{stem}.tex").write_text(tex + "\n")
    print(f"\n## {stem}\n{md}")


def _latex(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(df.columns)
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{l" + "c" * (len(cols) - 1) + "}", r"\toprule",
             " & ".join(cols) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(str(r[c]) for c in cols).replace("≤", r"$\leq$")
                     .replace("±", r"$\pm$").replace("†", r"$^\dagger$") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table2(det_pooled: pd.DataFrame) -> None:
    rows = []
    for prompt in CF_PROMPTS:
        r = det_pooled[(det_pooled.model == MAIN_MODEL) & (det_pooled.prompt == prompt)]
        if not r.empty:
            r = r.iloc[0]
            rows.append({"Method": f"DrugCon ({MAIN_MODEL}, {prompt})", "P": r.precision,
                         "R": r.recall, "F1": r.f1, "runs": int(r.n_runs)})
    df = pd.DataFrame(rows)
    _write("table2_drugcon_detection", ev.to_markdown(df),
           _latex(df, "DrugCon detection on DrugCon-Bench, mean $\\pm$ std over decoding temperatures.",
                  "tab:detection"))


def table3(det: pd.DataFrame, det_pooled: pd.DataFrame, expl: pd.DataFrame, models) -> None:
    rows, sources = [], []
    for slug, name, tier in models:
        pooled = slug == MAIN_MODEL  # paper reports the temperature mean for the main backbone
        cf = _pick(det_pooled if pooled else det, slug, CF_PROMPTS, pooled)
        dc = _pick(det, slug, [DC_PROMPT], pooled=False)
        ex = _pick(expl, slug, EXPL_PROMPTS, pooled=False)
        get = (lambda r, m: None if r is None else r[f"{m}_mean"])
        mark = "†" if cf is not None and cf.prompt == "counterfactual_appendix_v1" else ""
        rows.append({"Tier": tier, "Model": name + mark,
                     "CF P": _fmt(get(cf, "precision")), "CF R": _fmt(get(cf, "recall")),
                     "CF F1": _fmt(get(cf, "f1")),
                     "DC P": _fmt(get(dc, "precision")), "DC R": _fmt(get(dc, "recall")),
                     "DC F1": _fmt(get(dc, "f1")), "Expl. micro-F1": _fmt(get(ex, "micro_f1"))})
        for col, r in [("CF", cf), ("DC", dc), ("Explanation", ex)]:
            sources.append({"model": slug, "column": col,
                            "prompt": None if r is None else r.prompt,
                            "temperature": "pooled" if (col == "CF" and pooled) or r is None
                            else r.get("temperature"),
                            "n_runs": None if r is None else int(r.n_runs)})
    df = pd.DataFrame(rows)
    note = ("\n\n† CF cell uses the strict counterfactual prompt (`counterfactual_appendix_v1`); "
            "unmarked CF cells use `counterfactual_appendix_v2`. See docs/prompts.md.")
    _write("table3_ablation", ev.to_markdown(df) + note,
           _latex(df, "Backbone and prompt ablation on DrugCon-Bench (temperature 0).", "tab:ablation"))
    pd.DataFrame(sources).to_csv(paths.TABLES_DIR / "paper" / "table3_ablation_sources.csv", index=False)


def tableC1(expl: pd.DataFrame, models) -> None:
    perfactor_dir = paths.TABLES_DIR / "perfactor"
    long_rows, out_rows = [], []
    support = {}
    for slug, name, _ in models:
        r = _pick(expl, slug, EXPL_PROMPTS, pooled=False)
        if r is None:
            continue
        pf = pd.read_csv(perfactor_dir / f"{slug}__{r.prompt}__{r.temperature}.csv").set_index("factor")
        row = {"Model": name}
        for f in FACTOR_ORDER:
            row[ev.FACTOR_CODES[f]] = pf.loc[f, "f1"]
            support[f] = int(pf.loc[f, "support"])
            long_rows.append({"model": slug, "prompt": r.prompt, "factor": f,
                              **pf.loc[f, ["precision", "recall", "f1", "support"]].to_dict()})
        row["Micro F1"] = pf.loc["MICRO", "f1"]
        out_rows.append(row)
    long = pd.DataFrame(long_rows)
    long.to_csv(paths.TABLES_DIR / "paper" / "perfactor_long.csv", index=False)
    long.pivot(index="factor", columns="model", values="f1").reindex(FACTOR_ORDER).to_csv(
        paths.TABLES_DIR / "paper" / "perfactor_f1_matrix.csv")

    df = pd.DataFrame(out_rows).sort_values("Micro F1", ascending=False)
    df = df.rename(columns={ev.FACTOR_CODES[f]: f"{ev.FACTOR_CODES[f]} (n={support[f]})"
                            for f in FACTOR_ORDER})
    shown = df.copy()
    for c in shown.columns[1:]:
        shown[c] = shown[c].map(_fmt)
    _write("tableC1_perfactor", ev.to_markdown(shown),
           _latex(shown, "Per-factor contradiction explanation F1 by backbone on DrugCon-Bench.",
                  "tab:perfactor_explanation"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all-models", action="store_true",
                    help="also include backbones evaluated but not reported in the paper")
    args = ap.parse_args()
    models = PAPER_MODELS + (EXTRA_MODELS if args.all_models else [])

    det = _load("comparison_detection.csv")
    det_pooled = _load("comparison_detection_pooled.csv")
    expl = _load("comparison_explanation.csv")
    (paths.TABLES_DIR / "paper").mkdir(parents=True, exist_ok=True)
    table2(det_pooled)
    table3(det, det_pooled, expl, models)
    tableC1(expl, models)


if __name__ == "__main__":
    main()
