#!/usr/bin/env python3
"""
Smoke test — send 3 chief complaints to every AI Hub text-generation model.

Covers:
  • Azure AI Foundry  (PRIMARY_API_KEY  →  /azure-openai/deployments/{id}/chat/completions)
  • AWS Bedrock       (PRIMARY_API_KEY  →  /aws-bedrock/model/{id}/invoke)

Omitted intentionally: phi-4-mini-instruct (1 TPM), text-embedding-* (3),
tts / tts-hd / whisper, gpt-image-1-5 (image generation).

Usage:
    python scripts/smoke_test_aihub_models.py
    python scripts/smoke_test_aihub_models.py --workers 12
    python scripts/smoke_test_aihub_models.py --json results/smoke_test.json
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

AIHUB_BASE         = "https://aihubapi.stanfordhealthcare.org/azure-openai"
AIHUB_BEDROCK_BASE = "https://aihubapi.stanfordhealthcare.org/aws-bedrock"
AIHUB_VERSION      = "2025-04-01-preview"
KEY_ENV            = "PRIMARY_API_KEY"

COMPLAINTS = ["chest pain", "shortness of breath", "broken arm"]

SYSTEM_PROMPT = (
    "You are an emergency triage assistant. "
    "For each numbered chief complaint below, provide a 2–4 word clinical category. "
    'Respond ONLY with valid JSON: {"results": [{"complaint": "...", "category": "..."}]}'
)
USER_MSG = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(COMPLAINTS))

STD_TOKENS       = 200   # non-reasoning models
REASONING_TOKENS = 800   # reasoning models (gpt-5-x, o-series)


# ─────────────────────────────────────────────────────────────────────────────
# Model inventory
# Each tuple: (deployment_id, token_param, temperature_or_None, max_tokens, notes)
# ─────────────────────────────────────────────────────────────────────────────

FOUNDRY_MODELS: list[tuple[str, str, float | None, int, str]] = [
    # ── Generic ────────────────────────────────────────────────────────────
    ("chat",         "max_completion_tokens", 0,    STD_TOKENS,       "generic routing alias"),
    # ── GPT-4 / GPT-4.1 family ────────────────────────────────────────────
    ("gpt-4o",       "max_completion_tokens", 0,    STD_TOKENS,       ""),
    ("gpt-4-1",      "max_completion_tokens", 0,    STD_TOKENS,       ""),
    ("gpt-4-1-mini", "max_completion_tokens", 0,    STD_TOKENS,       ""),
    ("gpt-4-1-nano", "max_completion_tokens", 0,    STD_TOKENS,       ""),
    # ── GPT-5 family (reasoning, temperature=1 required on aihub) ────────
    ("gpt-5",        "max_completion_tokens", None, REASONING_TOKENS, "securegpt · reasoning"),
    ("gpt-5-mini",   "max_completion_tokens", None, REASONING_TOKENS, "securegpt · reasoning"),
    ("gpt-5-nano",   "max_completion_tokens", 1,    REASONING_TOKENS, "reasoning"),
    ("gpt-5-1",      "max_completion_tokens", 1,    REASONING_TOKENS, "reasoning"),
    ("gpt-5-2",      "max_completion_tokens", 1,    REASONING_TOKENS, "reasoning"),
    ("gpt-5-4",      "max_completion_tokens", 1,    REASONING_TOKENS, "reasoning"),
    ("gpt-5-4-mini", "max_completion_tokens", 1,    REASONING_TOKENS, "reasoning mini"),
    ("gpt-5-4-nano", "max_completion_tokens", 1,    REASONING_TOKENS, "reasoning nano"),
    ("gpt-5-5",      "max_completion_tokens", 1,    REASONING_TOKENS, "reasoning"),
    # ── o-series (temperature must be 1) ──────────────────────────────────
    ("o1",           "max_completion_tokens", 1,    REASONING_TOKENS, "o-series"),
    ("o3",           "max_completion_tokens", 1,    REASONING_TOKENS, "o-series"),
    ("o3-mini",      "max_completion_tokens", 1,    REASONING_TOKENS, "o-series"),
    ("o4-mini",      "max_completion_tokens", 1,    REASONING_TOKENS, "o-series"),
]

BEDROCK_MODEL_IDS: list[str] = [
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-opus-4-1-20250805-v1:0",
    "us.anthropic.claude-opus-4-6-v1",
    "us.anthropic.claude-opus-4-7",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-sonnet-4-6",
]


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    model:            str
    api:              str          # "foundry" | "bedrock"
    notes:            str
    status:           str          # "PASS" | "FAIL"
    latency_s:        float
    response_preview: str          # first 120 chars of content
    categories:       list[str] = field(default_factory=list)
    error:            str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_categories(content: str) -> list[str]:
    """Pull the 'category' values out of the model response JSON (best-effort)."""
    try:
        data = json.loads(content)
        return [r.get("category", "") for r in data.get("results", [])]
    except Exception:
        pass
    # Fallback: look for any JSON blob
    import re
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            return [r.get("category", "") for r in data.get("results", [])]
        except Exception:
            pass
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Per-model test functions
# ─────────────────────────────────────────────────────────────────────────────

def test_foundry(
    deployment_id: str,
    token_param: str,
    temperature: float | None,
    max_tokens: int,
    notes: str,
    key: str,
) -> TestResult:
    url  = (f"{AIHUB_BASE}/deployments/{deployment_id}"
            f"/chat/completions?api-version={AIHUB_VERSION}")
    body: dict = {
        "model":    deployment_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_MSG},
        ],
        token_param: max_tokens,
    }
    if temperature is not None:
        body["temperature"] = temperature

    t0 = time.time()
    try:
        r = requests.post(
            url,
            headers={"api-key": key, "Content-Type": "application/json"},
            json=body,
            timeout=180,
        )
        r.raise_for_status()
        content  = (r.json().get("choices") or [{}])[0].get("message", {}).get("content") or ""
        elapsed  = time.time() - t0
        return TestResult(
            model=deployment_id, api="foundry", notes=notes,
            status="PASS", latency_s=round(elapsed, 1),
            response_preview=content[:120].replace("\n", " "),
            categories=_extract_categories(content),
        )
    except Exception as exc:
        return TestResult(
            model=deployment_id, api="foundry", notes=notes,
            status="FAIL", latency_s=round(time.time() - t0, 1),
            response_preview="", error=str(exc)[:150],
        )


def test_bedrock(model_id: str, key: str) -> TestResult:
    url  = f"{AIHUB_BEDROCK_BASE}/model/{model_id}/invoke"
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "system":  SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": [{"type": "text", "text": USER_MSG}]}],
        "max_tokens": STD_TOKENS,
        "temperature": 0,
    }
    t0 = time.time()
    try:
        r = requests.post(
            url,
            headers={"api-key": key, "Content-Type": "application/json"},
            json=body,
            timeout=180,
        )
        r.raise_for_status()
        data    = r.json()
        texts   = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
        content = texts[0] if texts else ""
        elapsed = time.time() - t0
        return TestResult(
            model=model_id, api="bedrock", notes="",
            status="PASS", latency_s=round(elapsed, 1),
            response_preview=content[:120].replace("\n", " "),
            categories=_extract_categories(content),
        )
    except Exception as exc:
        return TestResult(
            model=model_id, api="bedrock", notes="",
            status="FAIL", latency_s=round(time.time() - t0, 1),
            response_preview="", error=str(exc)[:150],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all(key: str, workers: int, foundry_only: bool = False, bedrock_only: bool = False) -> list[TestResult]:
    futures: dict = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        if not bedrock_only:
            for dep, tp, temp, mtok, notes in FOUNDRY_MODELS:
                futures[pool.submit(test_foundry, dep, tp, temp, mtok, notes, key)] = dep
        if not foundry_only:
            for mid in BEDROCK_MODEL_IDS:
                futures[pool.submit(test_bedrock, mid, key)] = mid

        total   = len(futures)
        done    = 0
        results = []
        for fut in as_completed(futures):
            r    = fut.result()
            done += 1
            icon = "✓" if r.status == "PASS" else "✗"
            cats = "  →  " + " | ".join(r.categories) if r.categories else ""
            print(
                f"  [{done:2d}/{total}] {icon} {r.api:8s}  "
                f"{r.model:46s}  {r.latency_s:5.1f}s{cats}",
                flush=True,
            )
            if r.status == "FAIL":
                print(f"            ERROR: {r.error}", flush=True)
            results.append(r)

    return sorted(results, key=lambda x: (x.api != "foundry", x.model))


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: list[TestResult]) -> None:
    passed  = sum(1 for r in results if r.status == "PASS")
    failed  = len(results) - passed
    width   = 80

    print(f"\n{'═' * width}")
    print(f"  RESULTS  {passed} passed · {failed} failed · {len(results)} total")
    print(f"{'═' * width}")

    if failed:
        print("\nFailed models:")
        for r in (x for x in results if x.status == "FAIL"):
            print(f"  ✗  [{r.api}]  {r.model}")
            print(f"       {r.error}")

    print(f"\n{'─' * width}")
    print(f"  {'Model':<46}  {'API':<9}  {'Status'}  {'Latency':>8}  Categories")
    print(f"{'─' * width}")
    for r in results:
        cats = " | ".join(r.categories) if r.categories else (r.error[:40] if r.error else "—")
        print(f"  {r.model:<46}  {r.api:<9}  {r.status:<6}  {r.latency_s:>7.1f}s  {cats}")
    print(f"{'─' * width}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=10,
                        help="Concurrent workers (default 10)")
    parser.add_argument("--json", metavar="PATH",
                        help="Write JSON results to this file for canvas import")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--foundry-only", action="store_true",
                       help="Test only Azure AI Foundry models")
    group.add_argument("--bedrock-only", action="store_true",
                       help="Test only AWS Bedrock Claude models")
    args = parser.parse_args()

    key = os.environ.get(KEY_ENV, "")
    if not key:
        sys.exit(f"Error: {KEY_ENV} is not set. Add it to your .env file.")

    n_foundry = 0 if args.bedrock_only else len(FOUNDRY_MODELS)
    n_bedrock = 0 if args.foundry_only else len(BEDROCK_MODEL_IDS)
    print(f"AI Hub model smoke test")
    if n_foundry:
        print(f"  {n_foundry} Azure AI Foundry models")
    if n_bedrock:
        print(f"  {n_bedrock} AWS Bedrock models")
    print(f"  Total: {n_foundry + n_bedrock}")
    print(f"  Complaints: {COMPLAINTS}")
    print(f"  Workers: {args.workers}")
    print()

    results = run_all(key, args.workers, foundry_only=args.foundry_only, bedrock_only=args.bedrock_only)
    print_summary(results)

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
        print(f"\nResults written to {out_path}")
        print("Run the canvas update script or open aihub-model-registry.canvas.tsx to visualise.")


if __name__ == "__main__":
    main()
