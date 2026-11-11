"""Run-file naming, discovery, and loading.

A *run* is one model x prompt x temperature x repeat over a frozen eval set.
Each run is stored as::

    results/runs/<task>__<model>__<prompt_version>__<temperature>__run<id>.csv.gz
    results/runs/<task>__<model>__<prompt_version>__<temperature>__run<id>_meta.json

Run files hold only the join key plus model output (``raw_response`` and
telemetry). Gold labels are always taken from the current eval set at scoring
time, so re-labelling the benchmark never requires re-querying a model.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import pandas as pd

from . import paths

RUN_RE = re.compile(
    r"^(?P<task>detection|explanation)__(?P<model>.+?)__(?P<prompt>.+?)__"
    r"(?P<temp>.+?)__run(?P<run>\d+)\.csv(?:\.gz)?$"
)

# Join key between a run file and its eval set.
JOIN_KEY = {"detection": "pair_key", "explanation": "id"}

# Columns kept in released run files.
RUN_COLUMNS = ["raw_response", "api_error", "in_tokens", "out_tokens", "latency_s",
               "model", "prompt_version", "temperature", "run_id", "backend"]


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def run_stem(task: str, model: str, prompt_version: str, temp_tag: str, run_id: int) -> str:
    return f"{task}__{slug(model)}__{prompt_version}__{slug(temp_tag)}__run{run_id}"


def pair_key(claim_1: str, claim_2: str) -> str:
    """Order-insensitive key identifying a claim pair."""
    return " || ".join(sorted([str(claim_1), str(claim_2)]))


def iter_run_files(task: str, runs_dir: Path = paths.RUNS_DIR) -> Iterator[tuple[Path, dict]]:
    for path in sorted(runs_dir.glob(f"{task}__*__run*.csv*")):
        m = RUN_RE.match(path.name)
        if m:
            yield path, m.groupdict()


def load_eval_set(task: str) -> pd.DataFrame:
    path = paths.DETECTION_SET if task == "detection" else paths.EXPLANATION_SET
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `python scripts/build_eval_sets.py` first.")
    return pd.read_csv(path)


def load_run(path: Path, task: str, eval_df: pd.DataFrame) -> pd.DataFrame:
    """Load a run and align it to ``eval_df`` (inner join on the task's key).

    Rows in the run that are not in the eval set are dropped; eval rows that are
    missing from the run raise an error, because the metric would be wrong.
    """
    key = JOIN_KEY[task]
    run = pd.read_csv(path)
    keep = [key] + [c for c in RUN_COLUMNS if c in run.columns]
    run = run[keep]
    merged = eval_df.merge(run, on=key, how="left", validate="one_to_one", indicator=True)
    missing = int((merged["_merge"] == "left_only").sum())
    if missing:
        raise ValueError(f"{path.name}: {missing} eval rows have no model output")
    return merged.drop(columns="_merge")
