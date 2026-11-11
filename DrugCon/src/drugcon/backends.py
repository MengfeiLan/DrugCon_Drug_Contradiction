"""LLM backends: Amazon Bedrock (Converse API), Azure OpenAI, and OpenAI.

Every backend takes a list of prompt strings and returns one result dict per
prompt with the keys ``raw_response``, ``api_error``, ``in_tokens``,
``out_tokens`` and ``latency_s``. API errors are captured per row instead of
aborting the run.

Credentials are read from the environment (or a local ``.env`` file); they are
never logged or written to disk.
"""
from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .prompts import SYSTEM_PROMPT

CONCURRENCY = int(os.getenv("DRUGCON_CONCURRENCY", "5"))
MAX_TOKENS = 1500

# Short alias -> Bedrock model / inference-profile ID used in the paper.
BEDROCK_MODELS = {
    # proprietary
    "claude-haiku-4.5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-sonnet-4.5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "gpt-5.5": "us.openai.gpt-5.5",
    # open-weight, by size tier
    "ministral-3b": "mistral.ministral-3-3b-instruct",
    "gemma-3-4b": "google.gemma-3-4b-it",
    "mistral-7b": "mistral.mistral-7b-instruct-v0:2",
    "llama-3.1-8b": "us.meta.llama3-1-8b-instruct-v1:0",
    "gemma-3-12b": "google.gemma-3-12b-it",
    "gpt-oss-20b": "openai.gpt-oss-20b-1:0",
    "gemma-3-27b": "google.gemma-3-27b-it",
    "qwen3-32b": "qwen.qwen3-32b-v1:0",
    "llama-3.3-70b": "us.meta.llama3-3-70b-instruct-v1:0",
}

# Reasoning models need a higher token budget and reject a custom temperature.
REASONING_MODELS = {"gpt-5.5", "us.openai.gpt-5.5"}
REASONING_MIN_TOKENS = 4000

# Models whose Converse API rejects a system message; the system instruction is
# prepended to the user turn instead.
NO_SYSTEM_MODELS = {"mistral-7b", "mistral.mistral-7b-instruct-v0:2"}


def load_local_env(path: str | Path = ".env") -> None:
    """Minimal ``.env`` loader (KEY=VALUE lines); existing env vars win."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def is_reasoning_model(model: str) -> bool:
    return model in REASONING_MODELS or BEDROCK_MODELS.get(model, model) in REASONING_MODELS


def _error_result(exc: Exception, t0: float) -> dict:
    return {"raw_response": "", "api_error": f"{type(exc).__name__}: {exc}",
            "in_tokens": None, "out_tokens": None, "latency_s": time.time() - t0}


# --------------------------------------------------------------------------- #
# Amazon Bedrock
# --------------------------------------------------------------------------- #
def run_bedrock(prompts: list[str], model: str, temperature: float | None = None) -> list[dict]:
    """Bedrock Converse API with a thread pool.

    Uses the standard AWS credential chain; set ``AWS_PROFILE`` / ``AWS_REGION``
    to choose an account and region (default region ``us-east-1``).
    """
    import boto3

    model_id = BEDROCK_MODELS.get(model, model)
    reasoning = is_reasoning_model(model)
    region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    client = boto3.Session().client("bedrock-runtime", region_name=region)
    max_tokens = REASONING_MIN_TOKENS if reasoning else MAX_TOKENS
    no_system = model in NO_SYSTEM_MODELS or model_id in NO_SYSTEM_MODELS
    results: list[dict | None] = [None] * len(prompts)

    def one(item):
        i, prompt = item
        t0 = time.time()
        try:
            inference = {"maxTokens": max_tokens}
            if temperature is not None and not reasoning:
                inference["temperature"] = temperature
            text = f"{SYSTEM_PROMPT}\n\n{prompt}" if no_system else prompt
            kwargs = dict(modelId=model_id,
                          messages=[{"role": "user", "content": [{"text": text}]}],
                          inferenceConfig=inference)
            if not no_system:
                kwargs["system"] = [{"text": SYSTEM_PROMPT}]
            resp = client.converse(**kwargs)
            parts = resp["output"]["message"]["content"]
            usage = resp.get("usage", {})
            results[i] = {
                "raw_response": "".join(p.get("text", "") for p in parts),
                "api_error": None,
                "in_tokens": usage.get("inputTokens"),
                "out_tokens": usage.get("outputTokens"),
                "latency_s": time.time() - t0,
            }
        except Exception as exc:  # noqa: BLE001 - keep per-row errors
            results[i] = _error_result(exc, t0)

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        list(pool.map(one, enumerate(prompts)))
    return results  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# OpenAI-compatible (Azure OpenAI / OpenAI)
# --------------------------------------------------------------------------- #
async def _run_chat_completions(client, prompts: list[str], model: str,
                                temperature: float | None) -> list[dict]:
    sem = asyncio.Semaphore(CONCURRENCY)
    results: list[dict | None] = [None] * len(prompts)

    async def one(i: int, prompt: str):
        async with sem:
            t0 = time.time()
            try:
                kwargs = dict(
                    model=model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": prompt}],
                    max_completion_tokens=MAX_TOKENS,
                )
                if temperature is not None:
                    kwargs["temperature"] = temperature
                resp = await client.chat.completions.create(**kwargs)
                usage = resp.usage
                results[i] = {
                    "raw_response": resp.choices[0].message.content or "",
                    "api_error": None,
                    "in_tokens": getattr(usage, "prompt_tokens", None),
                    "out_tokens": getattr(usage, "completion_tokens", None),
                    "latency_s": time.time() - t0,
                }
            except Exception as exc:  # noqa: BLE001
                results[i] = _error_result(exc, t0)

    await asyncio.gather(*[one(i, p) for i, p in enumerate(prompts)])
    return results  # type: ignore[return-value]


def run_azure(prompts: list[str], model: str | None = None,
              temperature: float | None = None) -> list[dict]:
    """Azure OpenAI. Needs ``AZURE_OPENAI_API_KEY``, ``AZURE_OPENAI_ENDPOINT``,
    ``AZURE_OPENAI_API_VERSION``; ``model`` defaults to
    ``AZURE_OPENAI_DEPLOYMENT_NAME``."""
    from openai import AsyncAzureOpenAI

    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    key = os.getenv("AZURE_OPENAI_API_KEY")
    if not all([api_version, endpoint, key]):
        raise SystemExit("Missing AZURE_OPENAI_API_KEY / _ENDPOINT / _API_VERSION in environment or .env")
    model = model or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    if not model:
        raise SystemExit("Pass --model or set AZURE_OPENAI_DEPLOYMENT_NAME")
    client = AsyncAzureOpenAI(api_version=api_version, azure_endpoint=endpoint, api_key=key)
    return asyncio.run(_run_chat_completions(client, prompts, model, temperature))


def run_openai(prompts: list[str], model: str, temperature: float | None = None) -> list[dict]:
    """OpenAI API (api.openai.com). Needs ``OPENAI_API_KEY``."""
    from openai import AsyncOpenAI

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Missing OPENAI_API_KEY in environment or .env")
    client = AsyncOpenAI(api_key=key)
    return asyncio.run(_run_chat_completions(client, prompts, model, temperature))


BACKENDS = {"bedrock": run_bedrock, "azure": run_azure, "openai": run_openai}
