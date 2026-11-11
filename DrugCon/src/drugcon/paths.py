"""Canonical repository paths.

All scripts resolve data and results relative to the repository root so they
can be launched from any working directory. Set ``DRUGCON_ROOT`` to point at a
different checkout (e.g. when the package is installed non-editable).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(os.getenv("DRUGCON_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = REPO_ROOT / "data"
BENCH_DIR = DATA_DIR / "drugcon_bench"
EVAL_DIR = DATA_DIR / "eval_sets"

RESULTS_DIR = REPO_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
TABLES_DIR = RESULTS_DIR / "tables"

# Benchmark files
GOLD_PAIRS = BENCH_DIR / "contradictory_claim_pairs.csv"
CANDIDATE_PAIRS = BENCH_DIR / "candidate_pairs.csv"
ABSTRACTS = BENCH_DIR / "abstracts.csv"

# Frozen evaluation sets
DETECTION_SET = EVAL_DIR / "detection.csv"
EXPLANATION_SET = EVAL_DIR / "explanation.csv"
