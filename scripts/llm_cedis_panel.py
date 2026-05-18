#!/usr/bin/env python3
"""
Multi-model CEDIS panel classifier — production-scale version.

Three models classify each chief complaint in parallel, then Claude
adjudicates every non-unanimous row. If Claude hits a content filter,
Gemini takes over as fallback arbiter for that row.

Panel   : gpt-4o-mini · llama-4-scout · gemini-2.0-flash  (run in parallel)
Arbiter : claude-3-5-sonnet  (fallback: gemini-2.0-flash)

Designed for runs of 100 → 9,000+ rows with:
  - Abbreviation pre-expansion (once, before any API calls)
  - 1,000-row chunked output (recoverable if interrupted)
  - Resume support (skips chunks whose output file already exists)
  - Per-run metadata/timing JSON

Usage:
    # Quick test (20 rows, single chunk):
    python scripts/llm_cedis_panel.py --rows 20

    # Full dataset:
    python scripts/llm_cedis_panel.py --all

    # Resume interrupted run:
    python scripts/llm_cedis_panel.py --all --resume --out-dir output/panel_benchmark/panel_20260305_...

    # Different dataset / abbreviation file:
    python scripts/llm_cedis_panel.py --all --raw data/... --abbrev data/abbreviations/site_b.csv
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
import requests
from openai import AzureOpenAI, OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=True)
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent))
from llm_cedis_second_labeller import (
    load_cedis, build_client, extract_json,
    MODEL_CONFIGS, GeminiClient,
)
from abbreviations import load_abbreviations, expand_series


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

APIM_KEY_ENV    = "APIM_API_KEY"
APIM_BASE       = "https://apim.stanfordhealthcare.org"
AIHUB_KEY_ENV   = "PRIMARY_API_KEY"
AIHUB_BASE      = "https://aihubapi.stanfordhealthcare.org/azure-openai"
AIHUB_VERSION   = "2025-04-01-preview"
RAW_CSV_DEFAULT = "data/training_cc/chief_complaints_medevac_all_corpus_deid.csv"
TEXT_COL        = "ValueTXT"
ABBREV_DEFAULT  = "data/abbreviations/medevac_abbreviations.csv"
CHUNK_SIZE      = 1000
RATE_LIMIT_SLEEP = 0.3
ERROR_SLEEP      = 1.0
ARBITER_MODEL               = "gemini-2.0-flash"
ARBITER_FALLBACK_MODEL      = "claude-3-5-sonnet"  # used when Gemini returns null/invalid
ARBITER_BATCH_SIZE          = 10                   # disputes per arbiter API call
CONFIDENCE_ABSTAIN_THRESHOLD = 1                   # confidence <= this → abstain from vote


# ─────────────────────────────────────────────────────────────────────────────
# Panel model configs
# ─────────────────────────────────────────────────────────────────────────────

PANEL_MODEL_CONFIGS: dict[str, dict] = {
    "gpt-4o-mini": {
        "backend":              "securegpt",
        "api_name":             "gpt-4o-mini",
        "url_base":             f"{APIM_BASE}/openai",
        "api_version":          "2024-12-01-preview",
        "token_param":          "max_completion_tokens",
        "temperature":          0,
        "tokens_per_complaint": 150,
        "tokens_floor":         0,
        "batch_size":           10,
        "json_mode":            True,
    },
    "llama-4-scout": {
        "backend":              "openai_compat",
        "api_name":             "Llama-4-Scout-17B-16E-Instruct",
        "url":                  f"{APIM_BASE}/llama4-scout/v1",
        "token_param":          "max_tokens",
        "temperature":          0,
        "tokens_per_complaint": 200,
        "tokens_floor":         0,
        "batch_size":           10,
        "json_mode":            False,
    },
    "claude-3-5-sonnet": {
        "backend":              "claude_apim",
        "url":                  f"{APIM_BASE}/Claude35Sonnetv2/awssig4fa",
        "model_id":             "arn:aws:bedrock:us-west-2:679683451337:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "temperature":          0.0,
        "tokens_per_complaint": 150,
        "tokens_floor":         0,
        "batch_size":           10,
        "json_mode":            False,
    },
    "gemini-2.0-flash": {
        "backend":              "gemini",
        "api_name":             "gemini-2.0-flash",
        "url":                  f"{APIM_BASE}/gcp-gem20flash-fa/apim-gcp-gem20flash-fa",
        "token_param":          "maxOutputTokens",
        "temperature":          0,
        "tokens_per_complaint": 150,
        "tokens_floor":         0,
        "batch_size":           20,
        "json_mode":            True,
    },
}

PANEL_MODELS = ["gpt-4o-mini", "llama-4-scout", "claude-3-5-sonnet"]

# ── Trial panel using new AI Hub endpoint (PRIMARY_API_KEY) ──────────────────
AIHUB_MODEL_CONFIGS: dict[str, dict] = {
    "gpt-5-nano": {
        "backend":              "aihub",
        "api_name":             "gpt-5-nano",
        "token_param":          "max_completion_tokens",
        "temperature":          None,  # gpt-5/o-series: don't set temperature
        # Reasoning model — must allocate tokens for internal chain-of-thought
        # ~700 reasoning + ~150 output per complaint; batch of 6 ≈ 5100 → within 5000 TPM
        "tokens_per_complaint": 850,
        "tokens_floor":         0,
        "batch_size":           5,
        "json_mode":            False,
    },
    "gpt-4-1-nano": {
        "backend":              "aihub",
        "api_name":             "gpt-4-1-nano",
        "token_param":          "max_completion_tokens",
        "temperature":          0,
        "tokens_per_complaint": 150,
        "tokens_floor":         0,
        "batch_size":           10,
        "json_mode":            True,
    },
    "grok-3-mini": {
        "backend":              "aihub",
        "api_name":             "grok-3-mini",
        "token_param":          "max_tokens",  # grok uses max_tokens not max_completion_tokens
        "temperature":          0,
        # 200 TPM sandbox limit — small batches + extra sleep
        "tokens_per_complaint": 120,
        "tokens_floor":         0,
        "batch_size":           3,
        "json_mode":            False,
        "batch_sleep":          8.0,  # rate-limit buffer for 200 TPM
    },
    "o4-mini": {
        "backend":              "aihub",
        "api_name":             "o4-mini",
        "token_param":          "max_completion_tokens",
        "temperature":          1,   # o-series requires temperature=1
        "tokens_per_complaint": 200,
        "tokens_floor":         0,
        "batch_size":           10,
        "json_mode":            False,
    },
}
TRIAL_PANEL_MODELS   = ["gpt-5-nano", "gpt-4-1-nano"]   # grok-3-mini excluded (200 TPM too restrictive with reasoning overhead)
TRIAL_ARBITER_MODEL  = "o4-mini"


def _get_model_cfg(model: str) -> dict:
    """Return config dict from AIHUB_MODEL_CONFIGS or PANEL_MODEL_CONFIGS."""
    if model in AIHUB_MODEL_CONFIGS:
        return AIHUB_MODEL_CONFIGS[model]
    return PANEL_MODEL_CONFIGS[model]


# ─────────────────────────────────────────────────────────────────────────────
# Claude APIM client
# ─────────────────────────────────────────────────────────────────────────────

class ClaudeApimClient:
    def __init__(self, url: str, model_id: str, key: str) -> None:
        self.url      = url
        self.model_id = model_id
        self.key      = key

    def complete(self, prompt_text: str, max_tokens: int = 500,
                 temperature: float = 0.0) -> str:
        r = requests.post(
            self.url,
            headers={"Ocp-Apim-Subscription-Key": self.key,
                     "Content-Type": "application/json"},
            json={"model_id": self.model_id, "prompt_text": prompt_text,
                  "temperature": temperature, "top_p": 0.25,
                  "max_tokens": max_tokens},
            timeout=60,
        )
        r.raise_for_status()
        return r.json().get("content", [{}])[0].get("text", "")


# ─────────────────────────────────────────────────────────────────────────────
# AI Hub direct-request client
# ─────────────────────────────────────────────────────────────────────────────

class _AiHubMessage:
    def __init__(self, content: str) -> None:
        self.content = content

class _AiHubChoice:
    def __init__(self, data: dict) -> None:
        self.message       = _AiHubMessage(data.get("message", {}).get("content", ""))
        self.finish_reason = data.get("finish_reason", "")

class _AiHubResponse:
    def __init__(self, data: dict) -> None:
        self.choices = [_AiHubChoice(c) for c in data.get("choices", [{}])]

class _AiHubCompletions:
    def __init__(self, key: str) -> None:
        self._key = key

    def create(self, model: str, messages: list, **kwargs) -> _AiHubResponse:
        url = (f"{AIHUB_BASE}/deployments/{model}/chat/completions"
               f"?api-version={AIHUB_VERSION}")
        body: dict = {"model": model, "messages": messages}
        for k in ("max_completion_tokens", "max_tokens",
                  "temperature", "response_format"):
            if k in kwargs and kwargs[k] is not None:
                body[k] = kwargs[k]
        r = requests.post(
            url,
            headers={"api-key": self._key, "Content-Type": "application/json"},
            json=body,
            timeout=120,
        )
        r.raise_for_status()
        return _AiHubResponse(r.json())

class _AiHubChat:
    def __init__(self, key: str) -> None:
        self.completions = _AiHubCompletions(key)

class _AiHubClient:
    """Drop-in replacement for AzureOpenAI for the AI Hub endpoint."""
    def __init__(self, key: str) -> None:
        self.chat = _AiHubChat(key)


# ─────────────────────────────────────────────────────────────────────────────
# Thread-safe client builder (no shared-state mutation)
# ─────────────────────────────────────────────────────────────────────────────

def build_panel_client(model: str):
    # Check AI Hub configs first, then fall back to APIM configs
    if model in AIHUB_MODEL_CONFIGS:
        key = os.environ.get(AIHUB_KEY_ENV, "")
        if not key:
            raise EnvironmentError(f"{AIHUB_KEY_ENV} not set")
        return _AiHubClient(key)  # single shared client; model determined per-call

    cfg = PANEL_MODEL_CONFIGS[model]
    key = os.environ.get(APIM_KEY_ENV, "")
    if not key:
        raise EnvironmentError(f"{APIM_KEY_ENV} not set")

    if cfg["backend"] == "claude_apim":
        return ClaudeApimClient(cfg["url"], cfg["model_id"], key)

    if cfg["backend"] == "securegpt":
        base = cfg.get("url_base", f"{APIM_BASE}/openai-eastus2").rstrip("/")
        url  = f"{base}/deployments/{cfg['api_name']}/chat/completions?api-version={cfg['api_version']}"
        return AzureOpenAI(
            api_version       = cfg["api_version"],
            azure_endpoint    = url,
            azure_deployment  = cfg["api_name"],
            default_headers   = {"Ocp-Apim-Subscription-Key": key,
                                  "Content-Type": "application/json"},
            azure_ad_token    = key,
            timeout           = httpx.Timeout(connect=30.0, read=300.0,
                                               write=30.0, pool=30.0),
        )

    if cfg["backend"] == "openai_compat":
        return OpenAI(
            base_url        = cfg["url"].rstrip("/"),
            api_key         = "placeholder",
            default_headers = {"Ocp-Apim-Subscription-Key": key},
        )

    if cfg["backend"] == "gemini":
        # Delegate to existing factory which handles GeminiClient construction
        return build_client(model)

    raise ValueError(f"Unknown backend for panel model {model}: {cfg['backend']}")


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

def _classification_system(cedis_code_list: str) -> str:
    return (
        "You are a clinical informatics expert classifying emergency department "
        "chief complaints using the Canadian Emergency Department Information "
        "System (CEDIS).\n\n"
        "Rules:\n"
        "- Choose ONLY from the valid CEDIS codes listed below.\n"
        "- Prefer the most specific code that fits the complaint.\n"
        "- For each result also provide a confidence score (integer 1–3):\n"
        "    1 = low  : complaint is vague, ambiguous, or uninterpretable\n"
        "    2 = medium: reasonable fit but some ambiguity remains\n"
        "    3 = high : clear, unambiguous match to one code\n"
        "- Return ONLY a JSON object with key \"results\" containing an array, "
        "each item: cedis_code (integer), cedis_complaint (string), "
        "confidence (integer 1-3).\n\n"
        f"Valid CEDIS codes:\n{cedis_code_list}"
    )


def _classification_user(texts: list[str]) -> str:
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    return (
        f"Classify ALL {len(texts)} chief complaints below. "
        f"Return exactly {len(texts)} results in order.\n\n{numbered}"
    )


def _claude_combined(system: str, user: str) -> str:
    return f"{system}\n\n---\n\n{user}"


def _arbiter_system(cedis_code_list: str) -> str:
    return (
        "You are the final arbitrator for CEDIS classification disputes.\n"
        "Multiple AI reviewers disagreed on the code for each complaint.\n\n"
        "CRITICAL: Return ONLY codes from the valid CEDIS list below.\n"
        "Prefer the most specific code. Avoid 866 (Minor NOS) if a better "
        "code exists. Avoid 999 (Unknown) unless truly uninterpretable.\n"
        "Return JSON: {\"results\": [{cedis_code: int, cedis_complaint: str, "
        "rationale: str ≤12 words}, ...]}\n\n"
        f"Valid CEDIS codes:\n{cedis_code_list}"
    )


def _arbiter_user(cases: list[dict], panel_models: list[str] | None = None) -> str:
    active = panel_models or PANEL_MODELS
    lines = []
    for i, c in enumerate(cases, 1):
        vote_parts = []
        for m in active:
            code = c["votes"].get(m, "?")
            conf = c["confs"].get(m, "?")
            abstained = (conf != "?" and int(conf) <= CONFIDENCE_ABSTAIN_THRESHOLD)
            tag = f" [abstained, conf={conf}]" if abstained else f" (conf={conf})"
            vote_parts.append(f"{m}: {code}{tag}")
        votes_str = "  |  ".join(vote_parts)
        lines.append(f"{i}. \"{c['text']}\"\n   Votes — {votes_str}")
    return (
        f"Adjudicate ALL {len(cases)} disputes. "
        f"Return exactly {len(cases)} results.\n\n" + "\n\n".join(lines)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single-model labelling (runs inside a thread)
# ─────────────────────────────────────────────────────────────────────────────

def _max_tokens(cfg: dict, n: int) -> int:
    return max(cfg["tokens_floor"], cfg["tokens_per_complaint"] * n)


def _call_model(client, model: str, texts: list[str], system: str) -> list[dict]:
    """Call API for one batch; returns parsed results list."""
    cfg  = _get_model_cfg(model)
    user = _classification_user(texts)
    mtok = _max_tokens(cfg, len(texts))

    if cfg["backend"] == "claude_apim":
        raw = client.complete(_claude_combined(system, user),
                              max_tokens=max(mtok, 500),
                              temperature=cfg["temperature"])

    elif cfg["backend"] == "gemini":
        raw = client.complete(
            system_prompt=system,
            user_message=user,
            max_output_tokens=mtok,
            temperature=cfg["temperature"] or 0,
            json_mode=cfg["json_mode"],
        )

    elif cfg["backend"] == "openai_compat":
        url  = cfg["url"].rstrip("/") + "/chat/completions"
        body: dict = {
            "model":            cfg["api_name"],
            "messages":         [{"role": "system", "content": system},
                                  {"role": "user",   "content": user}],
            cfg["token_param"]: mtok,
        }
        if cfg["temperature"] is not None:
            body["temperature"] = cfg["temperature"]
        r   = requests.post(
            url,
            headers={"Ocp-Apim-Subscription-Key": os.environ.get(APIM_KEY_ENV, ""),
                     "Content-Type": "application/json"},
            json=body, timeout=60,
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"] or ""

    else:   # securegpt / aihub — both use AzureOpenAI client interface
        kwargs: dict = {
            "model":    cfg["api_name"],
            "messages": [{"role": "system", "content": system},
                         {"role": "user",   "content": user}],
            cfg["token_param"]: mtok,
        }
        if cfg["temperature"] is not None:
            kwargs["temperature"] = cfg["temperature"]
        if cfg["json_mode"]:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        raw  = resp.choices[0].message.content or ""
        if not raw.strip():
            raise ValueError(
                f"Empty response (finish_reason={resp.choices[0].finish_reason})"
            )

    parsed  = extract_json(raw)
    results = parsed.get("results", [])
    if len(results) != len(texts):
        raise ValueError(f"Expected {len(texts)} results, got {len(results)}")
    return results


def _parse_result(r: dict, valid_codes: set[int]) -> tuple[int | None, int]:
    """Extract (cedis_code, confidence) from a single result dict."""
    try:
        c = int(r.get("cedis_code"))
        code = c if c in valid_codes else None
    except (TypeError, ValueError):
        code = None
    try:
        conf = int(r.get("confidence", 2))
        conf = max(1, min(3, conf))   # clamp to 1–3
    except (TypeError, ValueError):
        conf = 2
    return code, conf


def label_with_model(model: str, texts: list[str],
                     system: str, valid_codes: set[int]
                     ) -> tuple[str, list[int | None], list[int], float]:
    """
    Label all texts with one panel model.
    Returns (model_name, list_of_codes, list_of_confidences, elapsed_seconds).
    Runs entirely inside a thread — creates its own client.
    """
    client     = build_panel_client(model)
    cfg        = _get_model_cfg(model)
    bs         = cfg["batch_size"]
    batch_slp  = cfg.get("batch_sleep", RATE_LIMIT_SLEEP)
    codes:  list[int | None] = []
    confs:  list[int]        = []
    t0     = time.time()

    for start in range(0, len(texts), bs):
        batch = texts[start: start + bs]
        for attempt in range(3):
            try:
                results = _call_model(client, model, batch, system)
                for r in results:
                    c, conf = _parse_result(r, valid_codes)
                    codes.append(c)
                    confs.append(conf)
                break
            except Exception as exc:
                print(f"\n    [{model}] batch attempt {attempt+1}: "
                      f"{type(exc).__name__}: {str(exc)[:80]}",
                      end=" ", flush=True)
                time.sleep(ERROR_SLEEP * (attempt + 1))
        else:
            # All batch attempts failed — fall back to individual calls
            print(f"\n    [{model}] individual fallback for {len(batch)} rows ...",
                  end=" ", flush=True)
            for text in batch:
                for ind_attempt in range(2):
                    try:
                        results = _call_model(client, model, [text], system)
                        c, conf = _parse_result(results[0], valid_codes)
                        codes.append(c)
                        confs.append(conf)
                        break
                    except Exception:
                        if ind_attempt == 1:
                            codes.append(None)
                            confs.append(2)
                        time.sleep(ERROR_SLEEP)
                time.sleep(batch_slp)

        time.sleep(batch_slp)

    return model, codes, confs, time.time() - t0


# ─────────────────────────────────────────────────────────────────────────────
# Arbiter (Claude primary, Gemini fallback on content filter)
# ─────────────────────────────────────────────────────────────────────────────

def _is_content_filter(exc: Exception) -> bool:
    """Detect Azure/APIM content filter errors."""
    msg = str(exc).lower()
    return ("content_filter" in msg or "content filter" in msg
            or "responsibleaipolicyviolation" in msg
            or ("400" in msg and "filtered" in msg))


def _call_claude_arbiter(
    client: ClaudeApimClient,
    case: dict,
    system: str,
    cedis_code_list: str,
    valid_codes: set[int],
) -> tuple[int | None, str | None]:
    """
    Ask Claude to adjudicate a single dispute.
    Returns (cedis_code, error_type) where error_type is None on success,
    'content_filter' if filtered, or 'error' for other failures.
    """
    user   = _arbiter_user([case])
    prompt = _claude_combined(system, user)
    max_tok = max(
        PANEL_MODEL_CONFIGS["claude-3-5-sonnet"]["tokens_floor"],
        PANEL_MODEL_CONFIGS["claude-3-5-sonnet"]["tokens_per_complaint"],
    )
    try:
        raw     = client.complete(prompt, max_tokens=max_tok, temperature=0.0)
        parsed  = extract_json(raw)
        results = parsed.get("results", [])
        if results:
            c = int(results[0].get("cedis_code"))
            return (c if c in valid_codes else None), None
        return None, "empty"
    except Exception as exc:
        if _is_content_filter(exc):
            return None, "content_filter"
        return None, "error"


def _call_gemini_arbiter(
    client,
    case: dict,
    system: str,
    valid_codes: set[int],
) -> int | None:
    """Ask Gemini to adjudicate a single dispute (fallback)."""
    cfg  = MODEL_CONFIGS[ARBITER_FALLBACK_MODEL]
    user = _arbiter_user([case])
    mtok = max(cfg["tokens_floor"], cfg["tokens_per_complaint"])
    try:
        raw     = client.complete(system_prompt=system, user_message=user,
                                  max_output_tokens=mtok, temperature=0, json_mode=True)
        parsed  = extract_json(raw)
        results = parsed.get("results", [])
        if results:
            c = int(results[0].get("cedis_code"))
            return c if c in valid_codes else None
    except Exception:
        pass
    return None


def _gemini_batch_arbiter(
    client,
    cases: list[dict],
    system: str,
    valid_codes: set[int],
) -> list[int | None]:
    """
    Ask Gemini to adjudicate a batch of disputes in one call.
    Returns a list of codes (None for any that fail validation).
    Falls back to per-row calls if the batch response is malformed.
    """
    cfg  = MODEL_CONFIGS[ARBITER_MODEL]
    user = _arbiter_user(cases)
    mtok = max(cfg["tokens_floor"], cfg["tokens_per_complaint"] * len(cases))
    try:
        raw    = client.complete(system_prompt=system, user_message=user,
                                 max_output_tokens=mtok, temperature=0, json_mode=True)
        parsed = extract_json(raw)
        results = parsed.get("results", [])
        if len(results) == len(cases):
            codes = []
            for r in results:
                try:
                    c = int(r.get("cedis_code"))
                    codes.append(c if c in valid_codes else None)
                except (TypeError, ValueError):
                    codes.append(None)
            return codes
    except Exception:
        pass
    # Fallback: call one row at a time
    return [_call_gemini_arbiter(client, c, system, valid_codes) for c in cases]


def run_arbiter(
    dispute_rows: list[dict],
    cedis_code_list: str,
    valid_codes: set[int],
) -> tuple[list[int | None], list[dict], float]:
    """
    Arbitrate all disputed rows with Gemini in batches of ARBITER_BATCH_SIZE.
    Rows that hit a content filter are routed to Claude fallback.

    Returns:
        finals          — list of final codes (one per dispute_row)
        fallback_events — list of dicts describing each content-filter fallback
        elapsed         — total seconds
    """
    if not dispute_rows:
        return [], [], 0.0

    gemini_client   = build_panel_client(ARBITER_MODEL)
    claude_client   = ClaudeApimClient(
        PANEL_MODEL_CONFIGS["claude-3-5-sonnet"]["url"],
        PANEL_MODEL_CONFIGS["claude-3-5-sonnet"]["model_id"],
        os.environ.get(APIM_KEY_ENV, ""),
    )
    gemini_system   = _arbiter_system(cedis_code_list)
    claude_system   = _claude_combined(_arbiter_system(cedis_code_list), "")

    finals:          list[int | None] = []
    fallback_events: list[dict]       = []
    t0  = time.time()
    n   = len(dispute_rows)
    bs  = ARBITER_BATCH_SIZE

    for batch_start in range(0, n, bs):
        batch = dispute_rows[batch_start: batch_start + bs]
        batch_codes = _gemini_batch_arbiter(gemini_client, batch, gemini_system, valid_codes)

        for j, (case, code) in enumerate(zip(batch, batch_codes)):
            global_i = batch_start + j
            if code is None:
                # Fallback: try Claude for this individual row
                claude_code, err = _call_claude_arbiter(
                    claude_client, case, claude_system, cedis_code_list, valid_codes
                )
                fallback_events.append({
                    "dispute_index":  global_i,
                    "text":           case["text"],
                    "votes":          {m: str(v) for m, v in case["votes"].items()},
                    "gemini_result":  "null/invalid",
                    "fallback_model": ARBITER_FALLBACK_MODEL,
                    "fallback_code":  claude_code,
                })
                print(f"\n    [arbiter] row {global_i+1} Gemini→null → "
                      f"Claude fallback ({claude_code})", end=" ", flush=True)
                finals.append(claude_code)
            else:
                finals.append(code)

        done = min(batch_start + bs, n)
        print(f"\r    [arbiter] {done}/{n} disputes arbitrated ...",
              end=" ", flush=True)
        time.sleep(RATE_LIMIT_SLEEP)

    return finals, fallback_events, time.time() - t0


def _aihub_batch_arbiter(
    client,
    cases: list[dict],
    system: str,
    valid_codes: set[int],
    panel_models: list[str],
) -> list[int | None]:
    """Ask o4-mini (AI Hub) to adjudicate a batch of disputes in one call."""
    cfg  = AIHUB_MODEL_CONFIGS[TRIAL_ARBITER_MODEL]
    user = _arbiter_user(cases, panel_models)
    mtok = max(cfg["tokens_floor"], cfg["tokens_per_complaint"] * len(cases))
    try:
        resp = client.chat.completions.create(
            model                 = cfg["api_name"],
            messages              = [{"role": "system", "content": system},
                                     {"role": "user",   "content": user}],
            max_completion_tokens = mtok,
            temperature           = cfg["temperature"],
        )
        raw    = resp.choices[0].message.content or ""
        parsed = extract_json(raw)
        results = parsed.get("results", [])
        if len(results) == len(cases):
            codes = []
            for r in results:
                try:
                    c = int(r.get("cedis_code"))
                    codes.append(c if c in valid_codes else None)
                except (TypeError, ValueError):
                    codes.append(None)
            return codes
    except Exception as exc:
        print(f"\n    [trial-arbiter] batch error: {exc}")
    # Per-row fallback
    results_fallback = []
    for case in cases:
        user_single = _arbiter_user([case], panel_models)
        mtok_single = max(cfg["tokens_floor"], cfg["tokens_per_complaint"])
        try:
            resp = client.chat.completions.create(
                model                 = cfg["api_name"],
                messages              = [{"role": "system", "content": system},
                                         {"role": "user",   "content": user_single}],
                max_completion_tokens = mtok_single,
                temperature           = cfg["temperature"],
            )
            raw    = resp.choices[0].message.content or ""
            parsed = extract_json(raw)
            rs     = parsed.get("results", [])
            if rs:
                c = int(rs[0].get("cedis_code"))
                results_fallback.append(c if c in valid_codes else None)
                continue
        except Exception:
            pass
        results_fallback.append(None)
    return results_fallback


def run_trial_arbiter(
    dispute_rows: list[dict],
    cedis_code_list: str,
    valid_codes: set[int],
    panel_models: list[str],
) -> tuple[list[int | None], list[dict], float]:
    """
    Arbitrate disputes with o4-mini (AI Hub) in batches.
    No secondary fallback — null results are recorded.
    """
    if not dispute_rows:
        return [], [], 0.0

    client = build_panel_client(TRIAL_ARBITER_MODEL)
    system = _arbiter_system(cedis_code_list)

    finals:          list[int | None] = []
    fallback_events: list[dict]       = []
    t0 = time.time()
    n  = len(dispute_rows)
    bs = ARBITER_BATCH_SIZE

    for batch_start in range(0, n, bs):
        batch       = dispute_rows[batch_start: batch_start + bs]
        batch_codes = _aihub_batch_arbiter(client, batch, system, valid_codes, panel_models)

        for j, (case, code) in enumerate(zip(batch, batch_codes)):
            if code is None:
                fallback_events.append({
                    "dispute_index": batch_start + j,
                    "text":  case["text"],
                    "votes": {m: str(v) for m, v in case["votes"].items()},
                    "o4mini_result": "null/invalid",
                })
            finals.append(code)

        done = min(batch_start + bs, n)
        print(f"\r    [trial-arbiter] {done}/{n} disputes arbitrated ...",
              end=" ", flush=True)
        time.sleep(RATE_LIMIT_SLEEP)

    return finals, fallback_events, time.time() - t0


# ─────────────────────────────────────────────────────────────────────────────
# Process one chunk (parallel panel + arbiter)
# ─────────────────────────────────────────────────────────────────────────────

def process_chunk(
    chunk_texts: list[str],
    chunk_start_row: int,
    cedis_code_list: str,
    valid_codes: set[int],
    system: str,
    panel_models: list[str] | None = None,
    arbiter_fn=None,
) -> tuple[pd.DataFrame, dict]:
    """
    Run the full panel + arbiter pipeline on one chunk.
    panel_models: list of model names to use (defaults to PANEL_MODELS)
    arbiter_fn: callable(dispute_rows, cedis_code_list, valid_codes) -> (finals, fallbacks, elapsed)
    Returns (result_df, timing_dict, fallback_events).
    """
    if panel_models is None:
        panel_models = PANEL_MODELS
    if arbiter_fn is None:
        arbiter_fn = run_arbiter
    n = len(chunk_texts)

    # ── Step 1: Run all panel models in parallel ─────────────────────────────
    panel_codes: dict[str, list[int | None]] = {}
    panel_confs: dict[str, list[int]]        = {}
    model_times: dict[str, float]            = {}

    with ThreadPoolExecutor(max_workers=len(panel_models)) as pool:
        futures = {
            pool.submit(label_with_model, m, chunk_texts, system, valid_codes): m
            for m in panel_models
        }
        for future in as_completed(futures):
            model_name, codes, confs, elapsed = future.result()
            panel_codes[model_name] = codes
            panel_confs[model_name] = confs
            model_times[model_name] = elapsed
            null_ct    = sum(1 for c in codes if c is None)
            abstain_ct = sum(1 for cf in confs if cf <= CONFIDENCE_ABSTAIN_THRESHOLD)
            print(f"\n    [{model_name}] done in {elapsed:.0f}s "
                  f"({elapsed/n:.1f}s/row) nulls={null_ct} abstains={abstain_ct}",
                  end=" ", flush=True)

    # ── Step 2: Build comparison dataframe ───────────────────────────────────
    df = pd.DataFrame({
        "row":  range(chunk_start_row, chunk_start_row + n),
        "text": chunk_texts,
    })
    for m in panel_models:
        df[m]              = pd.array(panel_codes[m], dtype="Int64")
        df[f"{m}_conf"]    = pd.array(panel_confs[m], dtype="Int64")

    # Effective vote: treat low-confidence (≤ threshold) as abstain
    eff_cols = []
    for m in panel_models:
        col = f"{m}_eff"
        df[col] = df.apply(
            lambda r, m=m: pd.NA
            if (pd.isna(r[m]) or int(r[f"{m}_conf"]) <= CONFIDENCE_ABSTAIN_THRESHOLD)
            else r[m],
            axis=1,
        ).astype("Int64")
        eff_cols.append(col)

    df["n_confident"]  = df[eff_cols].apply(lambda r: r.dropna().count(), axis=1)
    df["n_distinct"]   = df[eff_cols].apply(lambda r: r.dropna().nunique(), axis=1)
    # Unanimous only when ≥2 confident voters all agree
    df["unanimous"]    = (df["n_distinct"] == 1) & (df["n_confident"] >= 2)
    df["majority_code"] = df[eff_cols].apply(
        lambda r: r.dropna().mode().iloc[0] if not r.dropna().empty else pd.NA,
        axis=1,
    ).astype("Int64")

    # ── Step 3: Arbiter on disputes ──────────────────────────────────────────
    dispute_idx = df[~df["unanimous"]].index.tolist()
    dispute_rows = [
        {
            "text":  df.loc[i, "text"],
            "votes": {m: df.loc[i, m]           for m in panel_models},
            "confs": {m: df.loc[i, f"{m}_conf"] for m in panel_models},
        }
        for i in dispute_idx
    ]

    df["arbiter_code"]    = pd.NA
    df["arbiter_model"]   = pd.NA
    arbiter_time = 0.0
    all_fallback_events: list[dict] = []

    if dispute_rows:
        print(f"\n    [arbiter] {len(dispute_rows)} disputes ...",
              end=" ", flush=True)
        finals, fallback_events, arbiter_time = arbiter_fn(
            dispute_rows, cedis_code_list, valid_codes
        )
        all_fallback_events = fallback_events
        fallback_dispute_indices = {e["dispute_index"] for e in fallback_events}

        # Determine which arbiter label to record
        # For default arbiter: fallbacks went to claude; for trial: no fallback
        is_trial = (arbiter_fn is not run_arbiter)
        primary_label  = TRIAL_ARBITER_MODEL if is_trial else ARBITER_MODEL
        fallback_label = TRIAL_ARBITER_MODEL if is_trial else ARBITER_FALLBACK_MODEL

        for local_i, (idx, code) in enumerate(zip(dispute_idx, finals)):
            df.at[idx, "arbiter_code"]  = code
            df.at[idx, "arbiter_model"] = (
                fallback_label
                if local_i in fallback_dispute_indices
                else primary_label
            )

        n_null = len(fallback_events)
        if n_null:
            print(f"\n    [arbiter] null/fallback events: {n_null}",
                  end=" ", flush=True)

    df["arbiter_code"] = df["arbiter_code"].astype("Int64")
    model_times[primary_label if dispute_rows else ARBITER_MODEL] = arbiter_time

    # ── Step 4: Final code + description ─────────────────────────────────────
    df["final_code"] = df.apply(
        lambda r: r["majority_code"] if r["unanimous"] else r["arbiter_code"],
        axis=1,
    ).astype("Int64")

    cedis_df = pd.read_csv(Path(__file__).parent.parent / "data" / "cedis_codes.csv")
    code_map = cedis_df.set_index("code")["complaint"].to_dict()
    df["final_complaint"] = df["final_code"].apply(
        lambda x: code_map.get(int(x), "") if pd.notna(x) else ""
    )

    # Drop internal effective-vote columns before returning
    df = df.drop(columns=[f"{m}_eff" for m in panel_models])

    return df, model_times, all_fallback_events


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_valid_codes() -> set[int]:
    path = Path(__file__).parent.parent / "data" / "cedis_codes.csv"
    return set(pd.read_csv(path)["code"].dropna().astype(int))


def process_single_chunk(
    chunk_texts: list[str],
    chunk_start_row: int,
    model: str,
    valid_codes: set[int],
    system: str,
) -> tuple[pd.DataFrame, float]:
    """
    Label a chunk with a single model (no panel, no arbiter).
    Returns (result_df, elapsed_seconds).

    Output columns: row, text, {model}, {model}_conf, final_code, final_complaint
    """
    model_name, codes, confs, elapsed = label_with_model(
        model, chunk_texts, system, valid_codes
    )
    n = len(chunk_texts)
    print(f"\n    [{model_name}] done in {elapsed:.0f}s ({elapsed/n:.1f}s/row) "
          f"nulls={sum(1 for c in codes if c is None)}",
          flush=True)

    df = pd.DataFrame({
        "row":              range(chunk_start_row, chunk_start_row + n),
        "text":             chunk_texts,
        model_name:         pd.array(codes, dtype="Int64"),
        f"{model_name}_conf": pd.array(confs, dtype="Int64"),
    })
    df["final_code"] = df[model_name].astype("Int64")

    cedis_df = pd.read_csv(Path(__file__).parent.parent / "data" / "cedis_codes.csv")
    code_map = cedis_df.set_index("code")["complaint"].to_dict()
    df["final_complaint"] = df["final_code"].apply(
        lambda x: code_map.get(int(x), "") if pd.notna(x) else ""
    )
    return df, elapsed


def _chunk_path(chunks_dir: Path, start: int, end: int) -> Path:
    return chunks_dir / f"chunk_{start:05d}_{end:05d}.csv"


def print_agreement_summary(df: pd.DataFrame, n: int,
                            panel_models: list[str] | None = None) -> None:
    active = panel_models or PANEL_MODELS
    n_unanimous = int(df["unanimous"].sum())
    n_disputed  = n - n_unanimous
    conf_cols = [f"{m}_conf" for m in active]
    if all(c in df.columns for c in conf_cols):
        n_any_abstain = int(
            df[conf_cols].apply(
                lambda r: any(pd.notna(v) and int(v) <= CONFIDENCE_ABSTAIN_THRESHOLD
                              for v in r), axis=1
            ).sum()
        )
        low1  = {m: int((df[f"{m}_conf"] == 1).sum()) for m in active}
        low2  = {m: int((df[f"{m}_conf"] == 2).sum()) for m in active}
        high3 = {m: int((df[f"{m}_conf"] == 3).sum()) for m in active}
    else:
        n_any_abstain = 0
        low1 = low2 = high3 = {}
    print(f"\nAgreement (panel of {len(active)}):")
    print(f"  Unanimous        : {n_unanimous}/{n} ({100*n_unanimous/n:.0f}%)")
    print(f"  Disputed → arbiter: {n_disputed}/{n} ({100*n_disputed/n:.0f}%)")
    if n_any_abstain:
        print(f"  Rows w/ ≥1 abstain: {n_any_abstain}/{n} ({100*n_any_abstain/n:.0f}%)")
    if low1:
        print(f"\n  Confidence breakdown (conf=1 / conf=2 / conf=3):")
        for m in active:
            print(f"    {m:30s}: {low1[m]:3d} / {low2[m]:3d} / {high3[m]:3d}")
    print(f"  n_distinct breakdown:")
    for k, v in df["n_distinct"].value_counts().sort_index().items():
        print(f"    {k} distinct code(s): {v} rows")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-model CEDIS panel — parallel + chunked",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--rows",  type=int,
                   help="Classify N rows starting from --start-row (default 1)")
    g.add_argument("--all",   action="store_true",
                   help="Classify the full dataset in 1,000-row chunks")
    p.add_argument("--start-row", type=int, default=1,
                   help="1-based row to start from (default 1)")
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    p.add_argument("--resume", action="store_true",
                   help="Skip chunks whose output CSV already exists")
    p.add_argument("--raw",    default=RAW_CSV_DEFAULT)
    p.add_argument("--col",    default=TEXT_COL)
    p.add_argument("--abbrev", default=ABBREV_DEFAULT,
                   help="Abbreviations CSV (pass 'none' to skip)")
    p.add_argument("--save-expanded", action="store_true",
                   help="Write abbreviation-expanded text back to a _expanded.csv")
    p.add_argument("--out-dir", default=None)
    p.add_argument(
        "--panel",
        choices=["default", "trial-aihub"],
        default="default",
        help=(
            "default = gpt-4o-mini + llama-4-scout + claude (APIM key, Gemini arbiter); "
            "trial-aihub = gpt-5-nano + gpt-4-1-nano + grok-3-mini (PRIMARY_API_KEY, o4-mini arbiter)"
        ),
    )
    p.add_argument(
        "--single-model",
        default=None,
        metavar="MODEL",
        help=(
            "Skip the panel entirely and classify every row with a single model. "
            "E.g. --single-model claude-3-5-sonnet. "
            "Output columns: row, text, {model}, {model}_conf, final_code, final_complaint."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load & expand ─────────────────────────────────────────────────────────
    raw_df    = pd.read_csv(args.raw)
    texts_raw = raw_df[args.col].fillna("").astype(str)

    abbrev_path = args.abbrev if args.abbrev.lower() != "none" else None
    abbrevs: dict[str, str] = {}
    if abbrev_path and Path(abbrev_path).exists():
        abbrevs = load_abbreviations(abbrev_path)
        print(f"Abbreviations : {len(abbrevs)} loaded from {abbrev_path}")
        for a, (exp, cs, _orig) in abbrevs.items():
            flag = " [case-sensitive]" if cs else ""
            print(f"  {a} → {exp}{flag}")
    elif abbrev_path:
        print(f"Abbreviations : {abbrev_path} not found — skipping")

    texts_all = expand_series(texts_raw, abbrevs)

    if args.save_expanded:
        exp_path = Path(args.raw).with_suffix("").as_posix() + "_expanded.csv"
        exp_df   = raw_df.copy()
        exp_df[args.col] = texts_all
        exp_df.to_csv(exp_path, index=False)
        print(f"Expanded CSV  : {exp_path}")

    # Subset by start-row and row count
    start_idx = max(0, args.start_row - 1)   # convert 1-based to 0-based
    texts_all = texts_all.iloc[start_idx:]
    if args.rows:
        texts_all = texts_all.head(args.rows)

    texts_list = texts_all.tolist()
    total_rows = len(texts_list)

    # ── Single-model mode ────────────────────────────────────────────────────
    if args.single_model:
        single_model = args.single_model
        stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag     = single_model.replace("/", "-").replace(".", "-")
        out_dir = Path(args.out_dir) if args.out_dir else \
                  Path(f"output/panel_benchmark/single_{tag}_{stamp}_n{total_rows}")
        chunks_dir = out_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nMode          : single-model")
        print(f"Model         : {single_model}")
        print(f"Total rows    : {total_rows}")
        print(f"Chunk size    : {args.chunk_size}")
        print(f"Output        : {out_dir}")
        print(f"Resume        : {args.resume}")
        print("=" * 65)

        _, cedis_code_list = load_cedis()
        valid_codes = load_valid_codes()
        system      = _classification_system(cedis_code_list)

        chunk_paths: list[Path] = []
        total_elapsed = 0.0
        t_wall = time.time()
        cs = args.chunk_size

        for chunk_start in range(0, total_rows, cs):
            chunk_end   = min(chunk_start + cs, total_rows)
            row_start   = start_idx + chunk_start + 1
            row_end     = start_idx + chunk_end
            chunk_texts = texts_list[chunk_start:chunk_end]
            out_path    = _chunk_path(chunks_dir, row_start, row_end)
            chunk_paths.append(out_path)

            if args.resume and out_path.exists():
                existing = pd.read_csv(out_path)
                if len(existing) == len(chunk_texts):
                    print(f"\nChunk {row_start:5d}–{row_end:5d} : skipped (already done)")
                    continue

            print(f"\nChunk {row_start:5d}–{row_end:5d} : {len(chunk_texts)} rows",
                  flush=True)

            chunk_df, elapsed = process_single_chunk(
                chunk_texts, row_start, single_model, valid_codes, system
            )
            total_elapsed += elapsed

            if abbrevs:
                orig = texts_raw.iloc[start_idx + chunk_start:
                                      start_idx + chunk_end].tolist()
                chunk_df.insert(2, "text_original", orig)

            chunk_df.to_csv(out_path, index=False)
            n_coded = chunk_df["final_code"].notna().sum()
            print(f"\n    → saved {out_path.name}  coded={n_coded}/{len(chunk_df)}",
                  flush=True)

        total_wall = time.time() - t_wall
        completed  = [p for p in chunk_paths if p.exists()]
        if not completed:
            print("\nNo chunks found to merge.")
            return

        merged     = pd.concat([pd.read_csv(p) for p in completed], ignore_index=True)
        final_path = out_dir / "panel_results.csv"
        merged.to_csv(final_path, index=False)

        n = len(merged)
        n_coded = merged["final_code"].notna().sum()
        print(f"\n{'='*65}")
        print(f"Merged {n} rows → {final_path}")
        print(f"Coded: {n_coded}/{n} ({100*n_coded/n:.0f}%)")
        print(f"Model time  : {total_elapsed:.0f}s  ({total_elapsed/n:.1f}s/row)")
        print(f"Wall time   : {total_wall:.0f}s  ({total_wall/n:.1f}s/row)")
        return

    # ── Panel / arbiter selection ─────────────────────────────────────────────
    if args.panel == "trial-aihub":
        active_panel_models = TRIAL_PANEL_MODELS
        active_arbiter_name = TRIAL_ARBITER_MODEL
        def active_arbiter_fn(dispute_rows, cedis_code_list, valid_codes):
            return run_trial_arbiter(dispute_rows, cedis_code_list,
                                     valid_codes, active_panel_models)
    else:
        active_panel_models = PANEL_MODELS
        active_arbiter_name = ARBITER_MODEL
        active_arbiter_fn   = run_arbiter

    # ── Output directory ──────────────────────────────────────────────────────
    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    panel_tag = f"_{args.panel}" if args.panel != "default" else ""
    out_dir  = Path(args.out_dir) if args.out_dir else \
               Path(f"output/panel_benchmark/panel{panel_tag}_{stamp}_n{total_rows}")
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nPanel         : {', '.join(active_panel_models)}")
    print(f"Arbiter       : {active_arbiter_name}")
    print(f"Total rows    : {total_rows}")
    print(f"Chunk size    : {args.chunk_size}")
    print(f"Output        : {out_dir}")
    print(f"Resume        : {args.resume}")
    print("=" * 65)

    _, cedis_code_list = load_cedis()
    valid_codes = load_valid_codes()
    system      = _classification_system(cedis_code_list)

    # ── Chunk loop ────────────────────────────────────────────────────────────
    chunk_paths: list[Path] = []
    all_model_times: dict[str, float] = {
        m: 0.0 for m in active_panel_models + [active_arbiter_name]
    }
    all_fallback_events: list[dict]      = []
    t_wall = time.time()

    cs = args.chunk_size
    for chunk_start in range(0, total_rows, cs):
        chunk_end   = min(chunk_start + cs, total_rows)
        row_start   = start_idx + chunk_start + 1   # 1-based source row number
        row_end     = start_idx + chunk_end
        chunk_texts = texts_list[chunk_start:chunk_end]
        out_path    = _chunk_path(chunks_dir, row_start, row_end)
        chunk_paths.append(out_path)

        if args.resume and out_path.exists():
            existing = pd.read_csv(out_path)
            if len(existing) == len(chunk_texts):
                print(f"\nChunk {row_start:5d}–{row_end:5d} : skipped (already done)")
                continue

        print(f"\nChunk {row_start:5d}–{row_end:5d} : {len(chunk_texts)} rows",
              flush=True)

        chunk_df, chunk_times, chunk_fallbacks = process_chunk(
            chunk_texts, row_start, cedis_code_list, valid_codes, system,
            panel_models=active_panel_models,
            arbiter_fn=active_arbiter_fn,
        )
        all_fallback_events.extend(chunk_fallbacks)

        # Store original text when abbreviations were expanded
        if abbrevs:
            orig = texts_raw.iloc[start_idx + chunk_start:
                                  start_idx + chunk_end].tolist()
            chunk_df.insert(2, "text_original", orig)

        chunk_df.to_csv(out_path, index=False)
        for m, t in chunk_times.items():
            all_model_times[m] = all_model_times.get(m, 0.0) + t

        n_ch   = len(chunk_df)
        n_unan = int(chunk_df["unanimous"].sum())
        print(f"\n    → saved {out_path.name}  "
              f"unanimous={n_unan}/{n_ch} ({100*n_unan/n_ch:.0f}%)",
              flush=True)

    total_wall = time.time() - t_wall

    # ── Merge all chunks ──────────────────────────────────────────────────────
    completed = [p for p in chunk_paths if p.exists()]
    if not completed:
        print("\nNo chunks found to merge.")
        return

    merged = pd.concat([pd.read_csv(p) for p in completed], ignore_index=True)
    final_path = out_dir / "panel_results.csv"
    merged.to_csv(final_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(merged)
    print(f"\n{'='*65}")
    print(f"Merged {n} rows → {final_path}")
    print(f"\nPer-model wall time (sum across chunks):")
    for m, t in all_model_times.items():
        tag = "(arbiter)" if m == active_arbiter_name else "(panel) "
        if t > 0:
            print(f"  {m:<35} {tag}  {t:.0f}s  ({t/n:.1f}s/row)")
    print(f"\nTotal wall-clock time: {total_wall:.0f}s  ({total_wall/n:.1f}s/row)")

    print_agreement_summary(merged, n, panel_models=active_panel_models)

    # ── Fallback / null report ────────────────────────────────────────────────
    if all_fallback_events:
        fb_df   = pd.DataFrame(all_fallback_events)
        fb_path = out_dir / "arbiter_fallbacks.csv"
        fb_df.to_csv(fb_path, index=False)
        print(f"\n  Arbiter nulls/fallbacks: {len(all_fallback_events)} row(s)")
        print(f"   Report → {fb_path}")
    else:
        print(f"\n  No arbiter fallbacks")

    # ── Metadata JSON ─────────────────────────────────────────────────────────
    meta = {
        "run_timestamp":   stamp,
        "panel_config":    args.panel,
        "panel_models":    active_panel_models,
        "arbiter_model":   active_arbiter_name,
        "total_rows":      n,
        "chunk_size":      args.chunk_size,
        "abbrev_file":     abbrev_path,
        "abbreviations":   {a: exp for a, (exp, _cs, _orig) in abbrevs.items()},
        "model_times_s":   {m: round(t, 1) for m, t in all_model_times.items() if t > 0},
        "total_wall_s":    round(total_wall, 1),
        "unanimous_pct":   round(100 * merged["unanimous"].sum() / n, 1),
        "arbiter_nulls":   len(all_fallback_events),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Metadata      → {out_dir / 'metadata.json'}")

    # Sample output
    display_cols = (["row", "text"] + active_panel_models +
                    ["unanimous", "arbiter_code", "final_code", "final_complaint"])
    print(f"\nSample (first 10 rows):")
    print(merged[[c for c in display_cols if c in merged.columns]]
          .head(10).to_string(index=False))


if __name__ == "__main__":
    main()
