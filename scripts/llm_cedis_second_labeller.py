#!/usr/bin/env python3
"""
AI second labeller: classify chief complaints to CEDIS codes.
No rule-based logic — pure AI classification for independent second opinion.
Pairs with GPT silver (reviewer 1) for dual-AI review; disagreements go to human reviewer.

Supported models (pick with --model)
--------------------------------------
  gpt-4o          Azure OpenAI (env: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT)
  gpt-5           Stanford SecureGPT (env: APIM_API_KEY)
  gpt-5-mini      Stanford SecureGPT
  gpt-5-nano      Stanford SecureGPT
  llama-3.3-70b   Stanford APIM / Llama (env: APIM_API_KEY)

Usage examples
--------------
  python scripts/llm_cedis_second_labeller.py --all-chunks --model llama-3.3-70b
  python scripts/llm_cedis_second_labeller.py --max-rows 1000 --model gpt-5-nano
  python scripts/llm_cedis_second_labeller.py --all-chunks --model gpt-5 --resume
  python scripts/llm_cedis_second_labeller.py --chunk 1-100 --gpt <path> --model gpt-4o
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests as _requests
from openai import AzureOpenAI, OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# Each entry fully describes how to call the model — no if/elif scattered below.
# ─────────────────────────────────────────────────────────────────────────────

SECUREGPT_BASE_URL = "https://apim.stanfordhealthcare.org/openai-eastus2"
AZURE_API_VERSION = "2023-05-15"

MODEL_CONFIGS: dict[str, dict] = {
    "gpt-4o": {
        "backend":              "securegpt",
        "api_name":             "gpt-4o",
        "api_version":          "2024-12-01-preview",
        "url_base":             "https://apim.stanfordhealthcare.org/openai",
        "token_param":          "max_completion_tokens",
        "temperature":          0,
        "tokens_per_complaint": 150,
        "tokens_floor":         0,
        "batch_size":           20,
        "json_mode":            True,
    },
    "gpt-5": {
        "backend":              "securegpt",
        "api_name":             "gpt-5",
        "api_version":          "2024-12-01-preview",
        "token_param":          "max_completion_tokens",
        "temperature":          None,   # reasoning model; temperature not supported
        "tokens_per_complaint": 1500,   # reasoning tokens dominate; ~1200/complaint observed
        "tokens_floor":         5000,
        "batch_size":           5,      # small batches keep reasoning budget predictable
        "json_mode":            True,
    },
    "gpt-5-mini": {
        "backend":              "securegpt",
        "api_name":             "gpt-5-mini",
        "api_version":          "2024-12-01-preview",
        "token_param":          "max_completion_tokens",
        "temperature":          None,
        "tokens_per_complaint": 1500,
        "tokens_floor":         5000,
        "batch_size":           5,
        "json_mode":            True,
    },
    "gpt-5-nano": {
        "backend":              "securegpt",
        "api_name":             "gpt-5-nano",
        "api_version":          "2024-12-01-preview",
        "token_param":          "max_completion_tokens",
        "temperature":          None,
        "tokens_per_complaint": 1500,
        "tokens_floor":         5000,
        "batch_size":           5,      # small batches keep reasoning budget predictable
        "json_mode":            True,
    },
    "llama-3.3-70b": {
        "backend":              "openai_compat",
        "api_name":             "Llama-3.3-70B-Instruct",
        "url":                  "https://apim.stanfordhealthcare.org/llama3370b/v1",
        "token_param":          "max_tokens",
        "temperature":          0,
        "tokens_per_complaint": 200,
        "tokens_floor":         0,
        "batch_size":           20,
        "json_mode":            False,  # not supported; we parse from prose
    },
    "gemini-2.0-flash": {
        "backend":              "gemini",
        "api_name":             "gemini-2.0-flash",
        "url":                  "https://apim.stanfordhealthcare.org/gcp-gem20flash-fa/apim-gcp-gem20flash-fa",
        "token_param":          "maxOutputTokens",   # Gemini's param name
        "temperature":          0,
        "tokens_per_complaint": 150,
        "tokens_floor":         0,
        "batch_size":           20,
        "json_mode":            True,   # via response_mime_type: application/json
    },
}

ALL_MODELS = set(MODEL_CONFIGS)

# ─────────────────────────────────────────────────────────────────────────────
# Tuning
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_BATCH_SIZE = 20
RATE_LIMIT_SLEEP = 0.3
ERROR_SLEEP = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

CEDIS_CSV = Path(__file__).parent.parent / "data" / "cedis_codes.csv"
RAW_CSV_DEFAULT = "data/training_cc/chief_complaints_medevac_all_corpus_deid.csv"
GPT_SILVER_DIR = Path("output/silver_labelled/cerner_cc")


# ─────────────────────────────────────────────────────────────────────────────
# Auth helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apim_key() -> str:
    """APIM_API_KEY takes precedence; falls back to AZURE_OPENAI_API_KEY."""
    key = os.environ.get("APIM_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY")
    if not key:
        raise EnvironmentError(
            "Set APIM_API_KEY (or AZURE_OPENAI_API_KEY) in your environment or .env file"
        )
    return key


# ─────────────────────────────────────────────────────────────────────────────
# Gemini client  (raw requests wrapper matching the OpenAI client interface)
# ─────────────────────────────────────────────────────────────────────────────

class GeminiClient:
    """
    Thin wrapper around the Stanford APIM Gemini endpoint.
    Translates OpenAI-style (system, user) messages into the Gemini REST schema
    and exposes a .complete() method used by classify_batch.
    """

    def __init__(self, url: str, key: str) -> None:
        self.url = url
        self.headers = {
            "Ocp-Apim-Subscription-Key": key,
            "Content-Type": "application/json",
        }

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        max_output_tokens: int,
        temperature: float = 0,
        json_mode: bool = True,
    ) -> str:
        generation_config: dict = {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        }
        if json_mode:
            generation_config["response_mime_type"] = "application/json"

        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_message}]}],
            "generation_config": generation_config,
        }
        resp = _requests.post(self.url, headers=self.headers, json=payload, timeout=120)
        resp.raise_for_status()
        # Response is a streaming array of chunks — concatenate all part texts
        chunks = resp.json()
        return "".join(
            part["text"]
            for chunk in chunks
            for candidate in chunk.get("candidates", [])
            for part in candidate.get("content", {}).get("parts", [])
        )


# ─────────────────────────────────────────────────────────────────────────────
# Client construction  (config-driven, no scattered if/elif)
# ─────────────────────────────────────────────────────────────────────────────

def build_client(model: str) -> AzureOpenAI | OpenAI | GeminiClient:
    cfg = MODEL_CONFIGS[model]

    if cfg["backend"] == "azure":
        for var in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"):
            if not os.environ.get(var):
                raise EnvironmentError(f"Missing required environment variable: {var}")
        return AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=AZURE_API_VERSION,
        )

    if cfg["backend"] == "securegpt":
        key = _apim_key()
        # Per-model url_base overrides the global SECUREGPT_BASE_URL
        base = cfg.get("url_base", os.environ.get("OPENAI_ENDPOINT", SECUREGPT_BASE_URL)).rstrip("/")
        url = (
            f"{base}/deployments/{cfg['api_name']}"
            f"/chat/completions?api-version={cfg['api_version']}"
        )
        import httpx
        return AzureOpenAI(
            api_version=cfg["api_version"],
            azure_endpoint=url,
            azure_deployment=cfg["api_name"],
            default_headers={
                "Ocp-Apim-Subscription-Key": key,
                "Content-Type": "application/json",
            },
            azure_ad_token=key,
            timeout=httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0),
        )

    if cfg["backend"] == "openai_compat":
        return OpenAI(
            base_url=cfg["url"],
            api_key="placeholder",   # auth is via the header below
            default_headers={"Ocp-Apim-Subscription-Key": _apim_key()},
        )

    if cfg["backend"] == "gemini":
        return GeminiClient(url=cfg["url"], key=_apim_key())

    raise ValueError(f"Unknown backend '{cfg['backend']}' for model '{model}'")


# ─────────────────────────────────────────────────────────────────────────────
# CEDIS helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_cedis(path: Path = CEDIS_CSV) -> tuple[dict[int, str], str]:
    """Return (code_map, formatted_list_string) from the CEDIS CSV."""
    df = pd.read_csv(path, dtype={"code": str})
    code_map = {int(r.code): r.complaint for _, r in df.iterrows()}
    code_list = "\n".join(
        f"{code}: {complaint}" for code, complaint in sorted(code_map.items())
    )
    return code_map, code_list


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

def build_system_prompt(cedis_code_list: str) -> str:
    return (
        "You are a clinical informaticist specialising in emergency department coding.\n"
        "Independently classify each chief complaint using the CEDIS "
        "(Canadian Emergency Department Information System) Presenting Complaint List.\n\n"
        "Rules:\n"
        "- Choose ONLY from the codes listed below.\n"
        "- Use the most specific matching code; avoid 866 (Minor complaints NOS) "
        "when a more specific code fits.\n"
        "- Return a separate classification for every complaint in the batch.\n\n"
        "Valid CEDIS codes:\n"
        f"{cedis_code_list}"
    )


def build_user_message(complaints: list[str], json_mode: bool) -> str:
    numbered = "\n".join(f"{i + 1}. {cc}" for i, cc in enumerate(complaints))
    json_instruction = (
        f'Return ONLY valid JSON: {{"results": [array of {len(complaints)} objects]}}. '
        f"Each object: cedis_code (integer), cedis_complaint (string), confidence (float 0–1)."
        if json_mode else
        f"Return a JSON object with key \"results\" containing an array of {len(complaints)} objects. "
        f"Each object: cedis_code (integer), cedis_complaint (string), confidence (float 0–1). "
        f"Output the JSON block only, no prose."
    )
    return (
        f"Classify ALL {len(complaints)} chief complaints below.\n"
        f"{json_instruction}\n\n"
        f"{numbered}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Token budget  (config-driven)
# ─────────────────────────────────────────────────────────────────────────────

def batch_size_for(model: str) -> int:
    return MODEL_CONFIGS[model].get("batch_size", DEFAULT_BATCH_SIZE)


def max_tokens_for_batch(model: str, n: int) -> int:
    cfg = MODEL_CONFIGS[model]
    return max(cfg["tokens_floor"], cfg["tokens_per_complaint"] * n)


# ─────────────────────────────────────────────────────────────────────────────
# JSON extraction (handles both strict JSON mode and prose-embedded JSON)
# ─────────────────────────────────────────────────────────────────────────────

def extract_json(content: str) -> dict:
    """
    Parse JSON from model output.
    Tries strict parse first; falls back to extracting the first {...} block
    for models that wrap JSON in prose.
    """
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    # Extract first JSON object from prose
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON found in response: {content[:200]!r}")


# ─────────────────────────────────────────────────────────────────────────────
# API call
# ─────────────────────────────────────────────────────────────────────────────

def classify_batch(
    client: AzureOpenAI | OpenAI | GeminiClient,
    model: str,
    complaints: list[str],
    system_prompt: str,
) -> list[dict]:
    """Send one batch to the model and return a list of result dicts."""
    cfg = MODEL_CONFIGS[model]
    user_msg = build_user_message(complaints, cfg["json_mode"])
    max_tok = max_tokens_for_batch(model, len(complaints))

    if cfg["backend"] == "gemini":
        content = client.complete(
            system_prompt=system_prompt,
            user_message=user_msg,
            max_output_tokens=max_tok,
            temperature=cfg["temperature"] or 0,
            json_mode=cfg["json_mode"],
        )
    else:
        kwargs: dict = {
            "model": cfg["api_name"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            cfg["token_param"]: max_tok,
        }
        if cfg["temperature"] is not None:
            kwargs["temperature"] = cfg["temperature"]
        if cfg["json_mode"]:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        if not content.strip():
            raise ValueError(
                f"Empty response (finish_reason={response.choices[0].finish_reason}). "
                f"Token budget may be too low. usage={response.usage}"
            )

    parsed = extract_json(content)
    results = parsed.get("results", [])
    if len(results) != len(complaints):
        raise ValueError(f"Expected {len(complaints)} results, got {len(results)}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Chunk-level processing
# ─────────────────────────────────────────────────────────────────────────────

def _classify_one(
    client: AzureOpenAI | OpenAI | GeminiClient,
    model: str,
    text: str,
    system_prompt: str,
) -> dict:
    """Classify a single complaint. Used as fallback when batch count mismatches."""
    results = classify_batch(client, model, [text], system_prompt)
    return results[0]


def label_texts(
    client: AzureOpenAI | OpenAI | GeminiClient,
    model: str,
    texts: list[str],
    system_prompt: str,
) -> dict[int, dict]:
    """
    Classify all texts in BATCH_SIZE batches.
    Returns {text_index: {"cedis_code": int|None, "cedis_complaint": str}}.
    Blank texts are skipped (mapped to None without an API call).

    On count mismatch the batch is retried once, then falls back to
    one-at-a-time classification so no rows are lost.
    """
    result_map: dict[int, dict] = {}
    non_empty_idx = [i for i, t in enumerate(texts) if str(t).strip()]
    for i, t in enumerate(texts):
        if not str(t).strip():
            result_map[i] = {"cedis_code": None, "cedis_complaint": None}

    for batch_start in range(0, len(non_empty_idx), batch_size_for(model)):
        idx_slice = non_empty_idx[batch_start: batch_start + batch_size_for(model)]
        batch = [texts[i] for i in idx_slice]

        preds = None
        for attempt in range(2):          # try batch twice before falling back
            try:
                preds = classify_batch(client, model, batch, system_prompt)
                break
            except ValueError as exc:     # count mismatch
                batch_num = batch_start // batch_size_for(model) + 1
                print(f"\n  batch {batch_num} attempt {attempt+1}: {exc}", end=" ")
                time.sleep(ERROR_SLEEP)
            except Exception as exc:      # API / network error — no retry
                batch_num = batch_start // batch_size_for(model) + 1
                print(f"\n  batch {batch_num} error: {exc}", end=" ")
                for i in idx_slice:
                    result_map[i] = {"cedis_code": None, "cedis_complaint": str(exc)}
                time.sleep(ERROR_SLEEP)
                break
        else:
            # Both batch attempts failed — fall back to one-at-a-time
            print(f"\n  falling back to individual classification for {len(idx_slice)} rows ...", end=" ")
            preds = []
            for i in idx_slice:
                try:
                    pred = _classify_one(client, model, texts[i], system_prompt)
                except Exception as exc:
                    pred = {"cedis_code": None, "cedis_complaint": str(exc)}
                preds.append(pred)
                time.sleep(RATE_LIMIT_SLEEP)

        if preds is None:
            continue

        for i, pred in zip(idx_slice, preds):
            try:
                code = int(pred.get("cedis_code"))
                compl = str(pred.get("cedis_complaint", ""))
            except (TypeError, ValueError):
                code, compl = None, str(pred)
            result_map[i] = {"cedis_code": code, "cedis_complaint": compl}

        time.sleep(RATE_LIMIT_SLEEP)

    return result_map


def build_ai_df(texts: list[str], result_map: dict[int, dict], model: str) -> pd.DataFrame:
    df = pd.DataFrame([
        {
            "ValueTXT": t,
            "cedis_code": result_map[i]["cedis_code"],
            "cedis_complaint": result_map[i]["cedis_complaint"],
            "labeller": model,
        }
        for i, t in enumerate(texts)
    ])
    return _clean_codes(df)


def build_review_rows(
    texts: list[str],
    result_map: dict[int, dict],
    gpt_df: pd.DataFrame,
    chunk: str,
    chunk_start: int,
) -> list[dict]:
    rows = []
    for i, t in enumerate(texts):
        ai_code = result_map[i]["cedis_code"]
        ai_compl = result_map[i]["cedis_complaint"]
        gpt_code_raw = gpt_df.iloc[i].get("cedis_code")
        gpt_code = int(gpt_code_raw) if pd.notna(gpt_code_raw) else None
        gpt_compl = gpt_df.iloc[i].get("cedis_complaint", "")
        rows.append({
            "row": chunk_start + i,
            "text": t,
            "reviewer1_gpt_code": gpt_code,
            "reviewer1_gpt_complaint": gpt_compl,
            "reviewer2_cursor_code": ai_code,
            "reviewer2_cursor_complaint": ai_compl,
            "agree": gpt_code == ai_code,
            "reviewer3_final_code": "",
            "reviewer3_final_complaint": "",
            "chunk": chunk,
        })
    return rows


def process_chunk(
    chunk: str,
    raw_path: Path,
    gpt_path: Path,
    cc_col: str,
    client: AzureOpenAI | OpenAI | GeminiClient,
    model: str,
    system_prompt: str,
) -> tuple[pd.DataFrame, list[dict]]:
    """Label one chunk and return (ai_labels_df, review_rows)."""
    start, end = map(int, chunk.split("-"))
    raw_df = pd.read_csv(raw_path).iloc[start - 1: end].reset_index(drop=True)
    gpt_df = pd.read_csv(gpt_path)
    texts = raw_df[cc_col].fillna("").tolist()

    result_map = label_texts(client, model, texts, system_prompt)
    ai_df = build_ai_df(texts, result_map, model)
    review_rows = build_review_rows(texts, result_map, gpt_df, chunk, start)
    return ai_df, review_rows


# ─────────────────────────────────────────────────────────────────────────────
# Resume helper
# ─────────────────────────────────────────────────────────────────────────────

def resume_chunk(
    chunk: str,
    ai_out: Path,
    gpt_path: Path,
) -> tuple[pd.DataFrame, list[dict]]:
    """Reconstruct review rows from an already-written AI output CSV (no API call)."""
    ai_df = pd.read_csv(ai_out)
    gpt_df = pd.read_csv(gpt_path)
    start, _ = map(int, chunk.split("-"))
    rows = []
    for i in range(len(ai_df)):
        t = ai_df.iloc[i]["ValueTXT"]
        ai_code_raw = ai_df.iloc[i]["cedis_code"]
        ai_code = int(ai_code_raw) if pd.notna(ai_code_raw) else None
        ai_compl = str(ai_df.iloc[i].get("cedis_complaint", ""))
        gpt_code_raw = gpt_df.iloc[i].get("cedis_code")
        gpt_code = int(gpt_code_raw) if pd.notna(gpt_code_raw) else None
        gpt_compl = str(gpt_df.iloc[i].get("cedis_complaint", ""))
        rows.append({
            "row": start + i, "text": t,
            "reviewer1_gpt_code": gpt_code, "reviewer1_gpt_complaint": gpt_compl,
            "reviewer2_cursor_code": ai_code, "reviewer2_cursor_complaint": ai_compl,
            "agree": gpt_code == ai_code,
            "reviewer3_final_code": "", "reviewer3_final_complaint": "",
            "chunk": chunk,
        })
    return ai_df, rows


# ─────────────────────────────────────────────────────────────────────────────
# Review file output
# ─────────────────────────────────────────────────────────────────────────────

def _write_disagreement_groups(disagree: pd.DataFrame, review_dir: Path) -> None:
    """
    Write disagreements_by_group.csv and disagreement_groups_summary.csv.
    Groups every unique (gpt_code, ai_code) pair, sorted by frequency descending.
    """
    df = disagree.copy()
    df["gpt_code_str"] = _to_code_str(df["reviewer1_gpt_code"])
    df["ai_code_str"]  = _to_code_str(df["reviewer2_cursor_code"])
    df["gpt_compl_"]   = df["reviewer1_gpt_complaint"].fillna("")
    df["ai_compl_"]    = df["reviewer2_cursor_complaint"].fillna("")

    group_keys = list(
        df.groupby(["gpt_code_str", "gpt_compl_", "ai_code_str", "ai_compl_"], sort=False).groups
    )
    group_id_map = {key: i for i, key in enumerate(group_keys, start=1)}

    df["group_id"] = df.apply(
        lambda r: group_id_map[(r["gpt_code_str"], r["gpt_compl_"], r["ai_code_str"], r["ai_compl_"])],
        axis=1,
    )
    df["group_label"] = (
        "GPT: " + df["reviewer1_gpt_complaint"].fillna("") + " (" + df["gpt_code_str"] + ")  vs  "
        "AI: "  + df["reviewer2_cursor_complaint"].fillna("") + " (" + df["ai_code_str"] + ")"
    )

    df = df.sort_values(["group_id", "row"]).reset_index(drop=True)
    df = df.drop(columns=["gpt_code_str", "ai_code_str", "gpt_compl_", "ai_compl_"], errors="ignore")

    cols = [
        "group_id", "group_label", "row", "text",
        "reviewer1_gpt_code", "reviewer1_gpt_complaint",
        "reviewer2_cursor_code", "reviewer2_cursor_complaint",
        "reviewer3_final_code", "reviewer3_final_complaint", "chunk",
    ]
    df[[c for c in cols if c in df.columns]].to_csv(
        review_dir / "disagreements_by_group.csv", index=False
    )

    summary = (
        df.groupby("group_id")
        .agg(
            gpt_code=("reviewer1_gpt_code", "first"),
            gpt_complaint=("reviewer1_gpt_complaint", "first"),
            ai_code=("reviewer2_cursor_code", "first"),
            ai_complaint=("reviewer2_cursor_complaint", "first"),
            count=("row", "count"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    summary.to_csv(review_dir / "disagreement_groups_summary.csv", index=False)
    print(f"  disagreements_by_group.csv + disagreement_groups_summary.csv ({len(summary)} groups)")


VAGUE_CODES = {"999", "866"}   # Unknown / Minor complaints NOS — always need human review


def _to_code_str(series: pd.Series) -> pd.Series:
    """Normalise cedis_code to a plain integer string ('866.0' → '866', NaN → '')."""
    def _convert(x):
        try:
            return str(int(float(x)))
        except (ValueError, TypeError):
            return ""
    return series.apply(_convert)


def _needs_review(full: pd.DataFrame) -> pd.DataFrame:
    """
    Return all rows that need human review, with a review_reason column:
      - 'disagree'        : models assigned different codes
      - 'both_unknown'    : both models assigned 999 (Unknown)
      - 'both_minor_nos'  : both models assigned 866 (Minor complaints NOS)
    Rows that truly agree on a specific code are excluded.
    """
    gpt = _to_code_str(full["reviewer1_gpt_code"])
    ai  = _to_code_str(full["reviewer2_cursor_code"])

    mask_disagree      = ~full["agree"]
    mask_both_unknown  = full["agree"] & (gpt == "999")
    mask_both_minor    = full["agree"] & (gpt == "866")

    parts = []
    if mask_disagree.any():
        df = full[mask_disagree].copy()
        df["review_reason"] = "disagree"
        parts.append(df)
    if mask_both_unknown.any():
        df = full[mask_both_unknown].copy()
        df["review_reason"] = "both_unknown"
        parts.append(df)
    if mask_both_minor.any():
        df = full[mask_both_minor].copy()
        df["review_reason"] = "both_minor_nos"
        parts.append(df)

    if not parts:
        return pd.DataFrame(columns=list(full.columns) + ["review_reason"])

    return pd.concat(parts, ignore_index=True).sort_values("row").reset_index(drop=True)


CODE_COLUMNS = ["reviewer1_gpt_code", "reviewer2_cursor_code",
                "reviewer3_final_code", "cedis_code"]


def _clean_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any float code columns (866.0) to nullable integers (866)."""
    df = df.copy()
    for col in CODE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def write_review_outputs(full: pd.DataFrame, review_dir: Path) -> None:
    """Write all review and disagreement files from the combined comparison DataFrame."""
    review_dir.mkdir(parents=True, exist_ok=True)

    full = _clean_codes(full)
    full.to_csv(review_dir / "review_all_gpt_vs_ai.csv", index=False)

    needs_review = _needs_review(full)
    needs_review.to_csv(review_dir / "disagreements_all_for_reviewer3.csv", index=False)

    reason_counts = needs_review["review_reason"].value_counts().to_dict()
    n_disagree = reason_counts.get("disagree", 0)
    n_unknown  = reason_counts.get("both_unknown", 0)
    n_minor    = reason_counts.get("both_minor_nos", 0)

    max_row = int(full["row"].max())
    n_batches = 0
    for batch_start in range(1, max_row + 1, 1000):
        batch_end = min(batch_start + 999, max_row)
        label = f"{batch_start}-{batch_end}"
        subset = needs_review[(needs_review["row"] >= batch_start) & (needs_review["row"] <= batch_end)]
        if not subset.empty:
            subset.to_csv(review_dir / f"disagreements_{label}_for_reviewer3.csv", index=False)
        n_batches += 1

    # Group analysis on true disagreements only (vague-code rows skew groups)
    true_disagree = needs_review[needs_review["review_reason"] == "disagree"]
    if not true_disagree.empty:
        _write_disagreement_groups(true_disagree, review_dir)

    n_agree = int(full["agree"].sum()) - n_unknown - n_minor
    print(
        f"Total rows: {len(full)} | "
        f"True agree: {n_agree} | "
        f"Disagree: {n_disagree} | "
        f"Both Unknown: {n_unknown} | "
        f"Both Minor NOS: {n_minor} | "
        f"Total for review: {len(needs_review)}"
    )
    print(f"Review files → {review_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _default_out_dir(model: str) -> str:
    if model == "gpt-4o":
        return "ai_second_labelled"   # backward compat
    return model.replace("-", "_") + "_labelled"


def _default_review_dir(out_dir: str) -> Path:
    suffix = out_dir.replace("_labelled", "").replace("ai_second", "gpt4o")
    return Path(f"output/review_disagreements_{suffix}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AI second labeller — classify chief complaints via LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default="gpt-4o", choices=sorted(ALL_MODELS))
    p.add_argument("--chunk", help="Single chunk label e.g. 1-100")
    p.add_argument("--gpt", help="Path to GPT silver CSV (required for --chunk)")
    p.add_argument("--all-chunks", action="store_true")
    p.add_argument("--max-rows", type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--raw", default=RAW_CSV_DEFAULT)
    p.add_argument("--col", default="ValueTXT")
    p.add_argument("--out-dir")
    p.add_argument("--review-dir")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = args.out_dir or _default_out_dir(args.model)
    review_dir = Path(args.review_dir) if args.review_dir else _default_review_dir(out_dir)
    out_ai = Path("output/silver_labelled") / out_dir
    raw_path = Path(args.raw)

    client = build_client(args.model)
    _, cedis_code_list = load_cedis()
    system_prompt = build_system_prompt(cedis_code_list)

    print(f"Model: {args.model} | Output: {out_ai} | Review: {review_dir}")

    # ── Batch mode ────────────────────────────────────────────────────────────
    if args.all_chunks or args.max_rows:
        chunks = sorted(
            [d.name for d in GPT_SILVER_DIR.iterdir() if d.is_dir() and "-" in d.name],
            key=lambda x: (int(x.split("-")[0]), int(x.split("-")[1])),
        )
        if args.max_rows:
            chunks = [c for c in chunks if int(c.split("-")[0]) <= args.max_rows]

        all_review: list[pd.DataFrame] = []

        for chunk in chunks:
            gpt_files = list((GPT_SILVER_DIR / chunk).glob("*.csv"))
            if not gpt_files:
                print(f"{chunk}: no GPT silver file, skipping")
                continue

            gpt_path = max(gpt_files, key=lambda p: p.stat().st_mtime)
            ai_out = out_ai / chunk / f"chief_complaints_{chunk}_ai_second.csv"

            if args.resume and ai_out.exists():
                ai_df, review_rows = resume_chunk(chunk, ai_out, gpt_path)
                review_df = pd.DataFrame(review_rows)
                n_disagree = int((~review_df["agree"]).sum())
                print(f"{chunk}: (resumed) {len(ai_df)} loaded, {n_disagree} disagreements")
            else:
                print(f"{chunk}: labelling ...", end=" ", flush=True)
                ai_df, review_rows = process_chunk(
                    chunk, raw_path, gpt_path, args.col, client, args.model, system_prompt
                )
                ai_out.parent.mkdir(parents=True, exist_ok=True)
                ai_df.to_csv(ai_out, index=False)
                review_df = pd.DataFrame(review_rows)
                n_disagree = int((~review_df["agree"]).sum())
                print(f"{len(ai_df)} labelled, {n_disagree} disagreements")

            all_review.append(review_df)

        if all_review:
            write_review_outputs(pd.concat(all_review, ignore_index=True), review_dir)
        return

    # ── Single-chunk mode ─────────────────────────────────────────────────────
    if not args.chunk or not args.gpt:
        raise SystemExit(
            "Provide --chunk + --gpt for a single chunk, "
            "or --all-chunks / --max-rows N for batch mode."
        )

    ai_df, review_rows = process_chunk(
        args.chunk, raw_path, Path(args.gpt), args.col, client, args.model, system_prompt
    )
    ai_out = out_ai / args.chunk / f"chief_complaints_{args.chunk}_ai_second.csv"
    ai_out.parent.mkdir(parents=True, exist_ok=True)
    ai_df.to_csv(ai_out, index=False)

    review_df = pd.DataFrame(review_rows)
    disagree_df = review_df[~review_df["agree"]]
    review_dir.mkdir(parents=True, exist_ok=True)
    disagree_df.to_csv(review_dir / "disagreements_all_for_reviewer3.csv", index=False)
    print(f"{args.chunk}: {len(ai_df)} labelled, {len(disagree_df)} disagreements")


if __name__ == "__main__":
    main()
