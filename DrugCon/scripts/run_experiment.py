"""Run a grid of models x tasks x prompts x temperatures x repeats.

Shells out to run_model.py so every run is identical to a manual invocation and
produces its own output file.

Examples:
    # backbone ablation (paper Table 3): both detection prompts + explanation at t=0
    python scripts/run_experiment.py --backend bedrock \
        --models claude-haiku-4.5 qwen3-32b gemma-3-12b \
        --detection-prompts counterfactual direct --explanation-prompts taxonomy \
        --temperatures 0.0 --runs 1

    # temperature sweep for the main backbone (paper Table 2)
    python scripts/run_experiment.py --backend openai --models gpt-4.1-nano \
        --tasks detection --detection-prompts counterfactual --temperatures 0.0 0.5 1.0

    # smoke test
    python scripts/run_experiment.py --models gpt-oss-20b --limit 5
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from drugcon.prompts import DETECTION_PROMPTS, EXPLANATION_PROMPTS

HERE = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("\n>>> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=HERE.parent)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--backend", default="bedrock", choices=["bedrock", "azure", "openai"])
    ap.add_argument("--tasks", nargs="+", default=["detection", "explanation"],
                    choices=["detection", "explanation"])
    ap.add_argument("--detection-prompts", nargs="+", default=["counterfactual"],
                    choices=sorted(DETECTION_PROMPTS))
    ap.add_argument("--explanation-prompts", nargs="+", default=["taxonomy"],
                    choices=sorted(EXPLANATION_PROMPTS))
    ap.add_argument("--temperatures", nargs="+", type=float, default=[0.0])
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    for model in args.models:
        for temp in args.temperatures:
            for run_id in range(1, args.runs + 1):
                base = [sys.executable, str(HERE / "run_model.py"), "--backend", args.backend,
                        "--model", model, "--temperature", str(temp), "--run-id", str(run_id)]
                if args.limit:
                    base += ["--limit", str(args.limit)]
                if "detection" in args.tasks:
                    for prompt in args.detection_prompts:
                        run(base + ["--task", "detection", "--prompt", prompt])
                if "explanation" in args.tasks:
                    for prompt in args.explanation_prompts:
                        run(base + ["--task", "explanation", "--prompt", prompt])

    print("\nAll runs complete. Score with:")
    print("  python scripts/score_all.py --task detection")
    print("  python scripts/score_all.py --task explanation")


if __name__ == "__main__":
    main()
