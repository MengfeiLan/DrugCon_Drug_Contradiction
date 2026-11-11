# DrugCon: Detecting Contradictory Evidence on Drug–Disease Associations at Scale

Code, benchmark, and model outputs for the paper

> Mengfei Lan, Joe Menke, Shiwei Liu, Zhiyong Lu, Halil Kilicoglu.
> **Detecting Contradictory Evidence on Drug–Disease Associations at Scale.** medRxiv, 2026.

DrugCon is an LLM-based framework that extracts conclusion claims from RCT
abstracts, links claims about the same drug–disease association, detects
contradictions with counterfactual reasoning, and explains them with a taxonomy
of ten study-level factors (population, dosage, outcome measures, ...).

Applying DrugCon to 224,756 RCT abstracts, we built **DrugCon-Bench**: 542
expert-annotated contradictory claim pairs covering every pairwise contradiction
among 819 abstracts.

| | P | R | F1 |
| --- | --- | --- | --- |
| Contradiction detection (GPT-4.1-nano, counterfactual prompt, mean over T ∈ {0, 0.5, 1}) | 0.628 | 0.898 | 0.739 |
| Contradiction explanation (GPT-4.1-nano, micro-averaged over factors) | 0.520 | 0.449 | 0.482 |

Numbers are recomputed from the released runs by `scripts/score_all.py`.

## Repository layout

```text
DrugCon/
├── data/
│   ├── drugcon_bench/          # DrugCon-Bench release (see data/README.md)
│   └── eval_sets/              # frozen detection / explanation sets
├── src/drugcon/                # installable package
│   ├── prompts.py              # all prompt templates (versioned)
│   ├── backends.py             # Bedrock / Azure OpenAI / OpenAI clients
│   ├── evaluation.py           # shared output parser + metrics
│   ├── runs.py                 # run-file naming and loading
│   └── paths.py
├── scripts/
│   ├── build_eval_sets.py      # freeze eval sets from the benchmark
│   ├── run_model.py            # one model x task x prompt x temperature
│   ├── run_experiment.py       # grid of runs
│   ├── score_all.py            # score every run, aggregate over runs
│   ├── item_level_explanation.py
│   ├── make_paper_tables.py    # regenerate Tables 2, 3, C1
│   └── export_prompts.py       # regenerate docs/prompts.md
├── results/
│   ├── runs/                   # raw model outputs for every reported run (.csv.gz + _meta.json)
│   └── tables/                 # scored metrics; paper/ holds the paper tables
└── docs/prompts.md             # every prompt, and which result it backs
```

## Installation

```bash
git clone https://github.com/<org>/DrugCon.git
cd DrugCon
python -m venv .venv && source .venv/bin/activate
pip install -e .            # scoring only (numpy, pandas)
pip install -e ".[all]"     # + openai and boto3 for querying models
```

Python 3.10 or newer. `requirements.txt` pins the environment used for the
released results.

## Reproduce the paper tables (no API calls)

All model outputs used in the paper are in `results/runs`, so every number can be
recomputed offline:

```bash
python scripts/build_eval_sets.py          # data/eval_sets/{detection,explanation}.csv
python scripts/score_all.py --task all     # results/tables/comparison_*.csv
python scripts/make_paper_tables.py        # results/tables/paper/
```

`results/tables/paper/table3_ablation_sources.csv` maps each table cell to the
run (model, prompt version, temperature) it came from. See
[`docs/prompts.md`](docs/prompts.md) for the prompt behind each column.

## Run models

Copy `.env.example` to `.env` and fill in the credentials for the backend you use.

| Backend | `--backend` | Credentials |
| --- | --- | --- |
| Amazon Bedrock (Converse API) | `bedrock` | standard AWS credential chain; `AWS_PROFILE`, `AWS_REGION` |
| Azure OpenAI | `azure` | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_DEPLOYMENT_NAME` |
| OpenAI | `openai` | `OPENAI_API_KEY` |

```bash
# smoke test on 5 pairs (written as *_limit5, ignored by the scorer)
python scripts/run_model.py --task detection --backend openai --model gpt-4.1-nano --limit 5

# DrugCon detection: counterfactual prompt over 3 temperatures
python scripts/run_experiment.py --backend openai --models gpt-4.1-nano \
    --tasks detection --detection-prompts counterfactual --temperatures 0.0 0.5 1.0

# backbone ablation on Bedrock
python scripts/run_experiment.py --backend bedrock \
    --models claude-sonnet-4.5 qwen3-32b gemma-3-12b \
    --detection-prompts counterfactual direct --explanation-prompts taxonomy

python scripts/score_all.py --task all
```

Bedrock model aliases (`ministral-3b`, `gemma-3-4b`, `llama-3.1-8b`,
`gemma-3-12b`, `gemma-3-27b`, `qwen3-32b`, `llama-3.3-70b`, `claude-haiku-4.5`,
`claude-sonnet-4.5`, `gpt-5.5`, plus `mistral-7b` and `gpt-oss-20b`) are defined
in `src/drugcon/backends.py`. Any full Bedrock model ID also works.

Prompts are selected by alias:

| Task | Alias | Version id |
| --- | --- | --- |
| detection | `counterfactual` (default) | `counterfactual_appendix_v2` |
| detection | `direct` | `drug_disease_claim_only_v2` |
| detection | `counterfactual_strict` | `counterfactual_appendix_v1` |
| explanation | `taxonomy` (default) | `taxonomy_v1` |
| explanation | `descriptive` | `appendix_v1` |

Each run writes
`results/runs/<task>__<model>__<prompt_version>__<temperature>__run<id>.csv.gz`,
holding the eval-set key and the raw response, plus a `_meta.json` with token
counts and latency. Gold labels are joined at scoring time, so runs stay valid
if the benchmark labels are revised.

## Evaluation

- **Detection** reports precision, recall, F1, and accuracy for the
  `Contradiction exists` class on the 1,731-pair closed-world set. The label is
  read from the `<conclusion>` block; unparseable responses count as negative
  and are reported in the `unparsed` column.
- **Explanation** is multi-label over the ten factors on the 542 gold pairs.
  `score_all.py` reports per-factor and micro/macro P/R/F1;
  `item_level_explanation.py` reports per-pair exact match, Jaccard, and
  example-based F1.

## Scope of this release

Included: DrugCon-Bench, detection and explanation prompts, model-querying
harness, evaluation code, and all model outputs behind the paper's ablation.

Not included:
- Upstream pipeline code for claim extraction (rhetorical sentence classifier)
  and claim pairing (PubTator 3 entity linking + BioHiCL-Base dense filtering).
  Their outputs for the 819 benchmark abstracts are in `data/drugcon_bench/`.
- Baseline implementations (SemRep-based, DeBERTa-Base-Con, MNLI-MedNLI).
- ManConCorpus, which must be obtained from its authors (see `data/README.md`).

## Citation

```bibtex
@article{lan2026drugcon,
  title   = {Detecting Contradictory Evidence on Drug--Disease Associations at Scale},
  author  = {Lan, Mengfei and Menke, Joe and Liu, Shiwei and Lu, Zhiyong and Kilicoglu, Halil},
  journal = {medRxiv},
  year    = {2026}
}
```

## License

Code: MIT (see `LICENSE`). DrugCon-Bench annotations: CC BY 4.0. Abstract text
remains under the license of each source article in the PMC Open Access subset.

## Acknowledgements

Supported by the National Library of Medicine of the National Institutes of
Health under award R01LM014079, and in part by the NIH Intramural Research
Program.
