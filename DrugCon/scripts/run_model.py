"""Run ONE model over a frozen eval set for detection or explanation.

Every model goes through the same code path (same prompts, same generation
settings), so the only variable is the model.

Examples:
    # counterfactual detection with GPT-4.1-nano on the OpenAI API
    python scripts/run_model.py --task detection --backend openai --model gpt-4.1-nano \
        --prompt counterfactual --temperature 0.0

    # explanation with Claude Sonnet 4.5 on Amazon Bedrock
    python scripts/run_model.py --task explanation --backend bedrock --model claude-sonnet-4.5 \
        --prompt taxonomy --temperature 0.0

    # smoke test on 5 rows
    python scripts/run_model.py --task detection --model gpt-oss-20b --limit 5

Writes results/runs/<task>__<model>__<prompt_version>__<temp>__run<id>.csv.gz
plus a matching _meta.json with token and latency telemetry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from drugcon import paths
from drugcon.backends import BACKENDS, BEDROCK_MODELS, is_reasoning_model, load_local_env
from drugcon.prompts import (DEFAULT_DETECTION_PROMPT, DEFAULT_EXPLANATION_PROMPT,
                             DETECTION_PROMPTS, EXPLANATION_PROMPTS, get_prompt)
from drugcon.runs import JOIN_KEY, RUN_COLUMNS, load_eval_set, run_stem


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["detection", "explanation"], required=True)
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="bedrock")
    ap.add_argument("--model", default=None,
                    help=f"model id or Bedrock alias ({', '.join(BEDROCK_MODELS)})")
    ap.add_argument("--prompt", default=None,
                    help=f"detection: {sorted(DETECTION_PROMPTS)} (default {DEFAULT_DETECTION_PROMPT}); "
                         f"explanation: {sorted(EXPLANATION_PROMPTS)} (default {DEFAULT_EXPLANATION_PROMPT})")
    ap.add_argument("--temperature", type=float, default=None, help="omit to use the model default")
    ap.add_argument("--run-id", type=int, default=1, help="repeat-run index")
    ap.add_argument("--limit", type=int, default=None, help="only the first N rows (smoke test)")
    ap.add_argument("--out-dir", default=str(paths.RUNS_DIR))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    load_local_env(paths.REPO_ROOT / ".env")

    if args.model is None:
        if args.backend == "bedrock":
            args.model = "gpt-oss-20b"
        elif args.backend == "openai":
            args.model = "gpt-4.1-nano"
        # azure falls back to AZURE_OPENAI_DEPLOYMENT_NAME inside the backend

    default = DEFAULT_DETECTION_PROMPT if args.task == "detection" else DEFAULT_EXPLANATION_PROMPT
    spec = get_prompt(args.task, args.prompt or default)

    df = load_eval_set(args.task)
    if args.limit:
        df = df.head(args.limit).copy()
    if args.task == "detection":
        prompts = [spec.builder(r.claim_1, r.claim_2) for r in df.itertuples()]
    else:
        prompts = [spec.builder(r.claim_1, r.claim_abs_1, r.claim_2, r.claim_abs_2)
                   for r in df.itertuples()]

    temp = args.temperature
    reasoning = args.backend == "bedrock" and is_reasoning_model(args.model or "")
    if temp is None:
        temp_tag = "default"
    elif reasoning:
        temp_tag = f"t{temp}(ignored)"  # reasoning models reject a custom temperature
    else:
        temp_tag = f"t{temp}"

    model_name = args.model or "azure-deployment"
    print(f"[{args.backend}] {model_name} | {args.task} [{spec.version}] | temp={temp_tag} | "
          f"run={args.run_id} | {len(df)} rows")

    results = BACKENDS[args.backend](prompts, args.model, temperature=temp)

    out = df[[JOIN_KEY[args.task]]].reset_index(drop=True).copy()
    for col in ["raw_response", "api_error", "in_tokens", "out_tokens", "latency_s"]:
        out[col] = [r[col] for r in results]
    out["model"] = model_name
    out["prompt_version"] = spec.version
    out["temperature"] = temp_tag
    out["run_id"] = args.run_id
    out["backend"] = args.backend
    out = out[[JOIN_KEY[args.task]] + RUN_COLUMNS]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = run_stem(args.task, model_name, spec.version, temp_tag, args.run_id)
    if args.limit:
        stem += f"_limit{args.limit}"  # keep smoke tests out of the scorer's glob
    out_path = out_dir / f"{stem}.csv.gz"
    out.to_csv(out_path, index=False)

    meta = {
        "task": args.task, "backend": args.backend, "model": model_name,
        "prompt_version": spec.version, "temperature": temp_tag, "run_id": args.run_id,
        "rows": len(out), "api_errors": int(out["api_error"].notna().sum()),
        "total_in_tokens": int(out["in_tokens"].dropna().sum()),
        "total_out_tokens": int(out["out_tokens"].dropna().sum()),
        "mean_latency_s": float(out["latency_s"].mean()),
    }
    (out_dir / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
