#!/usr/bin/env python3
"""
Multi-model CEDIS panel classifier — production-scale version.

Two cheap models classify each chief complaint in parallel; a stronger
adjudicator resolves every non-unanimous row. All three run against Stanford
AI Hub's AWS Bedrock (Anthropic) endpoint, using the shared model registry in
llm_cedis_second_labeller.py — deliberately avoiding the Azure AI Foundry
endpoint, whose mandatory content moderation blocks routine, legitimate
clinical-necessity content (assault, self-harm, overdose, etc.) common in
real ED chief complaints.

If the arbiter's pick agrees with neither panel model (i.e. it overruled
both rather than just breaking a tie between them), the row's `final_code`
is left blank and `needs_hand_review` is set to True instead of trusting the
arbiter outright — see `needs_hand_review.csv` in the run's output dir.

Panel    : claude-haiku-4-5 + claude-sonnet-4-5 (both AWS Bedrock)
Arbiter  : claude-opus-4-6 (AWS Bedrock) — only invoked on disagreement

See llm_cedis_second_labeller.py's MODEL_CONFIGS for the full model registry,
including a documented list of currently-broken Bedrock models to avoid.

Supports both medevac and commercial cohorts via --cohort, which resolves
--raw/--col defaults from local_paths.cfg (see local_paths.cfg.template).

Designed for runs of 100 → 9,000+ rows with:
  - Abbreviation pre-expansion (once, before any API calls)
  - 1,000-row chunked output (recoverable if interrupted)
  - Resume support (skips chunks whose output file already exists)
  - Per-run metadata/timing JSON
  - A minimal {cohort}_labelled.csv export matching medevac_db's expected
    schema (row, text, text_original, cedis_code, cedis_complaint,
    needs_hand_review), alongside the full diagnostic CSV for QA

Usage:
    # Quick test (20 rows, single chunk):
    python scripts/llm_cedis_panel.py --rows 20 --cohort medevac

    # Full dataset:
    python scripts/llm_cedis_panel.py --all --cohort commercial

    # Resume interrupted run:
    python scripts/llm_cedis_panel.py --all --cohort medevac --resume --out-dir output/panel_medevac_20260305_.../

    # Different/explicit dataset or abbreviation file:
    python scripts/llm_cedis_panel.py --all --raw data/... --col ValueTXT --abbrev data/abbreviations/site_b.csv
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=True)
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent))
from llm_cedis_second_labeller import (
    load_cedis, build_client, extract_json,
    MODEL_CONFIGS, ContentFilterError,
    CODING_RULES,
    _LOCAL,
)
from abbreviations import load_abbreviations, expand_series


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

APIM_KEY_ENV    = "APIM_API_KEY"
APIM_BASE       = "https://apim.stanfordhealthcare.org"
AIHUB_KEY_ENV   = "PRIMARY_API_KEY"

# ── Cohort support (medevac vs. commercial) ──────────────────────────────────
# Both cohorts are exported by medevac_db as de-identified NER corpora with an
# identical schema (ValueTXT text column) — only the file path differs.
# See docs/PHI_MACHINE_SETUP.md and medevac_db's
# docs/COMMERCIAL_CEDIS_LABELING_HANDOFF.md.
COHORTS         = ("medevac", "commercial")
COHORT_RAW_KEYS = {"medevac": "raw_cc_medevac", "commercial": "raw_cc_commercial"}
COHORT_RAW_DEFAULTS = {
    "medevac":    "data/raw/chief_complaints_phi.csv",
    "commercial": "data/raw/chief_complaints_phi_commercial.csv",
}
TEXT_COL        = _LOCAL.get("cc_column", "ValueTXT")
ABBREV_DEFAULT  = "data/abbreviations/medevac_abbreviations.csv"
CHUNK_SIZE      = 1000
RATE_LIMIT_SLEEP = 0.3
ERROR_SLEEP      = 1.0

# ── Model roster — 2 cheap panel models + 1 adjudicator on disagreement only.
# See llm_cedis_second_labeller.py's MODEL_CONFIGS for backend details and a
# documented list of currently-broken Bedrock models (opus-4-8/opus-5/sonnet-5)
# to avoid using as the arbiter.
#
# All three are Bedrock-backed (Anthropic), not a mix of Azure Foundry +
# Bedrock. Real ED chief complaints routinely and legitimately describe
# assault, sexual assault, self-harm, overdose, and other content that Azure
# AI Foundry's mandatory content moderation blocks outright (see
# ContentFilterError in llm_cedis_second_labeller.py) — confirmed in practice
# on real medevac PHI, where gpt-4-1-mini (Azure) got its entire batch blocked
# while claude-haiku-4-5 (Bedrock) sailed through the same rows. Azure's
# filter can only be loosened via a formal Microsoft "modified content
# filter" request through the AI Hub admins, which is not something we
# control per-run — so for now, keep every model in this pipeline on the
# Bedrock backend, which is materially more permissive for clinical-necessity
# text. label_with_model() still isolates and retries batch-level content
# filter blocks row-by-row as a safety net, in case any single row still
# trips Anthropic's own moderation.
PANEL_MODELS    = ["claude-haiku-4-5", "claude-sonnet-4-5"]
ARBITER_MODEL   = "claude-opus-4-6"
ARBITER_BATCH_SIZE          = 10   # disputes per arbiter API call
CONFIDENCE_ABSTAIN_THRESHOLD = 1   # confidence <= this → abstain from vote

# NOTE: all panel/arbiter models are defined in llm_cedis_second_labeller.py's
# shared MODEL_CONFIGS registry (imported above) and built via build_client().
# The legacy PANEL_MODEL_CONFIGS / AIHUB_MODEL_CONFIGS duplicate dicts, the
# gpt-5-nano/gpt-4-1-nano/o4-mini "trial-aihub" panel, and the ClaudeApimClient
# + duplicate _AiHub* client classes have been retired — build_client() covers
# every backend (securegpt/openai_compat/gemini/aihub/bedrock) that MODEL_CONFIGS
# declares.


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

def _classification_system(cedis_code_list: str) -> str:
    return (
        "You are a clinical informatics expert classifying emergency department "
        "chief complaints using the Canadian Emergency Department Information "
        "System (CEDIS).\n\n"
        f"{CODING_RULES}\n\n"
        "For each result also provide a confidence score (integer 1–3):\n"
        "    1 = low  : complaint is vague, ambiguous, or uninterpretable\n"
        "    2 = medium: reasonable fit but some ambiguity remains\n"
        "    3 = high : clear, unambiguous match to one code\n"
        "Return ONLY a JSON object with key \"results\" containing an array, "
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
        f"{CODING_RULES}\n\n"
        "Avoid 999 (Unknown) unless the complaint is truly uninterpretable.\n"
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
    cfg  = MODEL_CONFIGS[model]
    user = _classification_user(texts)
    mtok = _max_tokens(cfg, len(texts))

    if cfg["backend"] == "gemini":
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

    else:   # securegpt / aihub / bedrock — all use the .chat.completions.create() interface
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
        resp   = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        raw    = choice.message.content or ""
        if not raw.strip():
            if choice.finish_reason == "content_filter":
                raise ContentFilterError(
                    f"Response was filtered by moderation for model '{model}'."
                )
            raise ValueError(
                f"Empty response (finish_reason={choice.finish_reason})"
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
    client     = build_client(model)
    cfg        = MODEL_CONFIGS[model]
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
            except ContentFilterError as exc:
                if len(batch) == 1:
                    # Single row — this IS the offending content. Not
                    # transient (retrying identical content against the same
                    # model/policy fails identically), so abstain immediately
                    # and let it fall through to the arbiter (a different
                    # backend/moderation stack) instead of wasting retries.
                    print(f"\n    [{model}] content filter blocked row: "
                          f"{str(exc)[:100]} — abstaining, no retry",
                          end=" ", flush=True)
                    codes.append(None)
                    confs.append(CONFIDENCE_ABSTAIN_THRESHOLD)
                    break
                # A batch-level block does NOT mean every row in it is
                # offending — one flagged chief complaint can take the whole
                # batch down with it. Isolate to individual calls so the
                # other (legitimate) rows aren't needlessly abstained too;
                # this matters a lot on real ED data, where assault/self-harm/
                # overdose language is common and legitimate.
                print(f"\n    [{model}] content filter blocked batch of "
                      f"{len(batch)} row(s): {str(exc)[:100]} — retrying "
                      f"individually to isolate offending row(s)",
                      end=" ", flush=True)
                for text in batch:
                    try:
                        results = _call_model(client, model, [text], system)
                        c, conf = _parse_result(results[0], valid_codes)
                        codes.append(c)
                        confs.append(conf)
                    except ContentFilterError:
                        codes.append(None)
                        confs.append(CONFIDENCE_ABSTAIN_THRESHOLD)
                    except Exception:
                        codes.append(None)
                        confs.append(2)
                    time.sleep(batch_slp)
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
                    except ContentFilterError:
                        codes.append(None)
                        confs.append(CONFIDENCE_ABSTAIN_THRESHOLD)
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
# Arbiter — claude-opus-4-6, only invoked on panel disagreement
# ─────────────────────────────────────────────────────────────────────────────

def _call_arbiter_batch(
    client,
    cases: list[dict],
    system: str,
    panel_models: list[str],
) -> list[dict]:
    """Send one batch of disputes to ARBITER_MODEL; returns parsed results list."""
    cfg  = MODEL_CONFIGS[ARBITER_MODEL]
    user = _arbiter_user(cases, panel_models)
    mtok = max(cfg["tokens_floor"], cfg["tokens_per_complaint"] * len(cases))
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

    resp   = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    raw    = choice.message.content or ""
    if not raw.strip():
        if choice.finish_reason == "content_filter":
            raise ContentFilterError(
                f"Arbiter response was filtered by moderation for model "
                f"'{ARBITER_MODEL}'."
            )
        raise ValueError(f"Empty arbiter response (finish_reason={choice.finish_reason})")

    parsed  = extract_json(raw)
    results = parsed.get("results", [])
    if len(results) != len(cases):
        raise ValueError(f"Expected {len(cases)} results, got {len(results)}")
    return results


def _parse_arbiter_results(
    results: list[dict], valid_codes: set[int]
) -> list[tuple[int | None, str]]:
    """
    Extract (cedis_code, rationale) per result. The rationale is the only
    audit trail available for disputed rows on real PHI data, where there's
    no gold label to spot-check against — always keep it, even when the code
    itself is invalid, so a human reviewer can see *why* the arbiter landed
    (or failed to land) on a code.
    """
    parsed: list[tuple[int | None, str]] = []
    for r in results:
        rationale = str(r.get("rationale", "") or "")
        try:
            c = int(r.get("cedis_code"))
            parsed.append((c if c in valid_codes else None, rationale))
        except (TypeError, ValueError):
            parsed.append((None, rationale))
    return parsed


def run_arbiter(
    dispute_rows: list[dict],
    cedis_code_list: str,
    valid_codes: set[int],
    panel_models: list[str] | None = None,
) -> tuple[list[int | None], list[str], list[dict], float]:
    """
    Arbitrate all disputed rows with ARBITER_MODEL in batches of
    ARBITER_BATCH_SIZE. Falls back to per-row calls if a batch response is
    malformed. Content-filter blocks are first isolated to the individual
    offending row(s) (see the ContentFilterError handling below) — a genuine
    per-row block is recorded as a null since there's no further fallback
    model to route it to.

    Returns:
        finals          — list of final codes (one per dispute_row)
        rationales      — list of the arbiter's stated rationale per row
                           (empty string if unavailable) — this is the only
                           audit trail on real PHI data where there's no gold
                           label to check disputed rows against
        fallback_events — list of dicts describing each row the arbiter
                           could not resolve (null result)
        elapsed         — total seconds
    """
    if not dispute_rows:
        return [], [], [], 0.0

    active = panel_models or PANEL_MODELS
    client = build_client(ARBITER_MODEL)
    system = _arbiter_system(cedis_code_list)

    finals:          list[int | None] = []
    rationales:      list[str]        = []
    fallback_events: list[dict]       = []
    t0 = time.time()
    n  = len(dispute_rows)
    bs = ARBITER_BATCH_SIZE

    for batch_start in range(0, n, bs):
        batch = dispute_rows[batch_start: batch_start + bs]
        try:
            results    = _call_arbiter_batch(client, batch, system, active)
            batch_pairs = _parse_arbiter_results(results, valid_codes)
        except ContentFilterError as exc:
            if len(batch) == 1:
                print(f"\n    [arbiter] content filter blocked row: "
                      f"{str(exc)[:100]}", end=" ", flush=True)
                batch_pairs = [(None, "")]
            else:
                # Don't null the whole batch over one flagged case — isolate
                # to individual calls so the other disputes still get
                # resolved (see the matching panel-side fix above for why).
                print(f"\n    [arbiter] content filter blocked batch of "
                      f"{len(batch)} row(s): {str(exc)[:100]} — retrying "
                      f"individually to isolate offending row(s)",
                      end=" ", flush=True)
                batch_pairs = []
                for case in batch:
                    try:
                        results = _call_arbiter_batch(client, [case], system, active)
                        batch_pairs.extend(_parse_arbiter_results(results, valid_codes))
                    except Exception:
                        batch_pairs.append((None, ""))
        except Exception as exc:
            print(f"\n    [arbiter] batch error: {type(exc).__name__}: "
                  f"{str(exc)[:80]} — falling back to per-row", end=" ", flush=True)
            batch_pairs = []
            for case in batch:
                try:
                    results = _call_arbiter_batch(client, [case], system, active)
                    batch_pairs.extend(_parse_arbiter_results(results, valid_codes))
                except Exception:
                    batch_pairs.append((None, ""))

        for j, (code, rationale) in enumerate(batch_pairs):
            global_i = batch_start + j
            if code is None:
                fallback_events.append({
                    "dispute_index": global_i,
                    "text":  dispute_rows[global_i]["text"],
                    "votes": {m: str(v) for m, v in dispute_rows[global_i]["votes"].items()},
                    "arbiter_result": "null/invalid",
                    "arbiter_rationale": rationale,
                })
            finals.append(code)
            rationales.append(rationale)

        done = min(batch_start + bs, n)
        print(f"\r    [arbiter] {done}/{n} disputes arbitrated ...",
              end=" ", flush=True)
        time.sleep(RATE_LIMIT_SLEEP)

    return finals, rationales, fallback_events, time.time() - t0


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
    arbiter_fn: callable(dispute_rows, cedis_code_list, valid_codes, panel_models)
                -> (finals, rationales, fallbacks, elapsed)
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

    df["arbiter_code"]      = pd.NA
    df["arbiter_model"]     = pd.NA
    # Audit trail for disputed rows — the only thing to check adjudication
    # quality against on real PHI data where there's no gold label available.
    df["arbiter_rationale"] = pd.NA
    # True when the arbiter picked a code that neither panel model voted for —
    # i.e. it overruled both cheap models rather than just breaking a tie
    # between them. Don't trust the arbiter blindly in that case; flag it for
    # a human to look at instead of silently taking its word for it.
    df["needs_hand_review"] = False
    arbiter_time = 0.0
    all_fallback_events: list[dict] = []
    overrode_both_events: list[dict] = []

    if dispute_rows:
        print(f"\n    [arbiter] {len(dispute_rows)} disputes ...",
              end=" ", flush=True)
        finals, rationales, fallback_events, arbiter_time = arbiter_fn(
            dispute_rows, cedis_code_list, valid_codes, panel_models
        )
        all_fallback_events = fallback_events

        for idx, code, rationale in zip(dispute_idx, finals, rationales):
            df.at[idx, "arbiter_code"]      = code
            df.at[idx, "arbiter_model"]     = ARBITER_MODEL
            df.at[idx, "arbiter_rationale"] = rationale

            panel_votes    = {m: df.loc[idx, m] for m in panel_models}
            non_null_votes = {v for v in panel_votes.values() if pd.notna(v)}
            overrode_both  = code is not None and code not in non_null_votes
            if overrode_both:
                df.at[idx, "needs_hand_review"] = True
                overrode_both_events.append({
                    "row":            df.loc[idx, "row"],
                    "text":           df.loc[idx, "text"],
                    "votes":          {m: str(v) for m, v in panel_votes.items()},
                    "arbiter_code":   code,
                    "arbiter_rationale": rationale,
                })

        n_null = len(fallback_events)
        if n_null:
            print(f"\n    [arbiter] null results: {n_null}",
                  end=" ", flush=True)
        n_overrode = len(overrode_both_events)
        if n_overrode:
            print(f"\n    [arbiter] overruled both panel models: {n_overrode} "
                  f"(flagged needs_hand_review)", end=" ", flush=True)

    df["arbiter_code"] = df["arbiter_code"].astype("Int64")
    model_times[ARBITER_MODEL] = arbiter_time

    # ── Step 4: Final code + description ─────────────────────────────────────
    # When the arbiter overruled both panel models, don't take its word as
    # final — leave final_code blank (pending) so it's obvious downstream
    # that a human needs to resolve it. `needs_hand_review` marks the row and
    # `arbiter_code`/`arbiter_rationale` still record what the arbiter guessed.
    df["final_code"] = df.apply(
        lambda r: (
            r["majority_code"] if r["unanimous"]
            else pd.NA if r["needs_hand_review"]
            else r["arbiter_code"]
        ),
        axis=1,
    ).astype("Int64")

    cedis_df = pd.read_csv(Path(__file__).parent.parent / "data" / "cedis_codes.csv")
    code_map = cedis_df.set_index("code")["complaint"].to_dict()
    df["final_complaint"] = df["final_code"].apply(
        lambda x: code_map.get(int(x), "") if pd.notna(x) else ""
    )

    # Drop internal effective-vote columns before returning
    df = df.drop(columns=[f"{m}_eff" for m in panel_models])

    return df, model_times, all_fallback_events, overrode_both_events


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


def _restore_nullable_int_dtypes(df: pd.DataFrame,
                                  model_names: list[str] | None = None) -> pd.DataFrame:
    """
    Each chunk is written to CSV with proper pandas nullable Int64 columns
    (e.g. arbiter_code, which is blank for every unanimous row), but
    pd.read_csv() doesn't know about that dtype on the way back in — any
    column with at least one blank comes back as plain float64, which
    renders whole numbers as "999.0" instead of "999" in every downstream
    consumer: the printed sample/rationale tables, panel_results.csv,
    and (worse) the `cedis_code` column of {cohort}_labelled.csv. Restore
    Int64 on every column we know is meant to hold whole CEDIS codes/confidences.
    """
    candidates = ["final_code", "arbiter_code", "majority_code",
                  "n_confident", "n_distinct"]
    for m in (model_names or []):
        candidates += [m, f"{m}_conf", f"{m}_eff"]
    for col in candidates:
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    return df


def write_cohort_labelled_csv(merged: pd.DataFrame, out_dir: Path, cohort: str) -> Path:
    """
    Export the minimal `{cohort}_labelled.csv` alongside the full diagnostic
    CSV, matching medevac_db's expected schema for attach_cedis_labels.py:
        row, text, text_original, cedis_code, cedis_complaint, needs_hand_review
    `row` is 1-based and must align with the source chief_complaints_phi.csv
    row order (see medevac_db's docs/COMMERCIAL_CEDIS_LABELING_HANDOFF.md).
    `cedis_code` is blank for rows where the arbiter overruled both panel
    models (`needs_hand_review == True`) — treat those as unlabelled until a
    human resolves them, not as a confident final answer.
    """
    labelled = merged.rename(
        columns={"final_code": "cedis_code", "final_complaint": "cedis_complaint"}
    )[["row", "text", "text_original", "cedis_code", "cedis_complaint",
       "needs_hand_review"]]
    out_path = out_dir / f"{cohort}_labelled.csv"
    labelled.to_csv(out_path, index=False)
    return out_path


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
    p.add_argument(
        "--cohort",
        choices=COHORTS,
        default=None,
        help=(
            "Which cohort to run: medevac or commercial. Resolves --raw's default "
            "from local_paths.cfg's raw_cc_medevac/raw_cc_commercial keys, and "
            "--out-dir's default naming, unless overridden explicitly."
        ),
    )
    p.add_argument("--raw",    default=None,
                   help="Raw/de-identified chief complaint CSV. Defaults to the "
                        "path for --cohort from local_paths.cfg if --cohort is set.")
    p.add_argument("--col",    default=TEXT_COL,
                   help=f"Text column name (default: {TEXT_COL!r} — same for both cohorts)")
    p.add_argument("--abbrev", default=ABBREV_DEFAULT,
                   help="Abbreviations CSV (pass 'none' to skip)")
    p.add_argument("--save-expanded", action="store_true",
                   help="Write abbreviation-expanded text back to a _expanded.csv")
    p.add_argument("--out-dir", default=None)
    p.add_argument(
        "--single-model",
        default=None,
        metavar="MODEL",
        help=(
            "Skip the panel entirely and classify every row with a single model. "
            "E.g. --single-model claude-opus-4-6. "
            "Output columns: row, text, {model}, {model}_conf, final_code, final_complaint."
        ),
    )
    args = p.parse_args()
    if args.raw is None:
        if args.cohort is None:
            p.error("--raw is required when --cohort is not set")
        cfg_key   = COHORT_RAW_KEYS[args.cohort]
        args.raw = _LOCAL.get(cfg_key, COHORT_RAW_DEFAULTS[args.cohort])
    return args


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
        merged     = _restore_nullable_int_dtypes(merged, [single_model])
        final_path = out_dir / "panel_results.csv"
        merged.to_csv(final_path, index=False)

        n = len(merged)
        n_coded = merged["final_code"].notna().sum()
        print(f"\n{'='*65}")
        print(f"Merged {n} rows → {final_path}")
        print(f"Coded: {n_coded}/{n} ({100*n_coded/n:.0f}%)")
        print(f"Model time  : {total_elapsed:.0f}s  ({total_elapsed/n:.1f}s/row)")
        print(f"Wall time   : {total_wall:.0f}s  ({total_wall/n:.1f}s/row)")

        if args.cohort:
            labelled_path = write_cohort_labelled_csv(merged, out_dir, args.cohort)
            print(f"Labelled CSV : {labelled_path}")
        return

    # ── Panel / arbiter selection ─────────────────────────────────────────────
    active_panel_models = PANEL_MODELS
    active_arbiter_name = ARBITER_MODEL
    active_arbiter_fn    = run_arbiter

    # ── Output directory ──────────────────────────────────────────────────────
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.cohort:
        out_dir = Path(f"output/panel_{args.cohort}_{stamp}_n{total_rows}")
    else:
        out_dir = Path(f"output/panel_benchmark/panel_{stamp}_n{total_rows}")
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
    all_hand_review_events: list[dict]   = []
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

        chunk_df, chunk_times, chunk_fallbacks, chunk_hand_review = process_chunk(
            chunk_texts, row_start, cedis_code_list, valid_codes, system,
            panel_models=active_panel_models,
            arbiter_fn=active_arbiter_fn,
        )
        all_fallback_events.extend(chunk_fallbacks)
        all_hand_review_events.extend(chunk_hand_review)

        # Always keep the pre-abbreviation-expansion original text alongside
        # the (possibly expanded) text shown to the model — medevac_db's
        # {cohort}_labelled.csv schema expects both columns unconditionally.
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
    merged = _restore_nullable_int_dtypes(merged, active_panel_models)
    final_path = out_dir / "panel_results.csv"
    merged.to_csv(final_path, index=False)

    labelled_path = None
    if args.cohort:
        labelled_path = write_cohort_labelled_csv(merged, out_dir, args.cohort)

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(merged)
    print(f"\n{'='*65}")
    print(f"Merged {n} rows → {final_path}")
    if labelled_path:
        print(f"Labelled CSV (medevac_db schema) → {labelled_path}")
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

    # ── Hand-review report ────────────────────────────────────────────────────
    # Rows where the arbiter overruled BOTH panel models. final_code is left
    # blank (Int64 NA) for these rows in panel_results.csv / the labelled CSV —
    # we do not take the arbiter's word as final without a human sign-off.
    if all_hand_review_events:
        hr_df   = pd.DataFrame(all_hand_review_events)
        hr_path = out_dir / "needs_hand_review.csv"
        hr_df.to_csv(hr_path, index=False)
        print(f"\n  Needs hand review (arbiter overruled both panel models): "
              f"{len(all_hand_review_events)} row(s)")
        print(f"   Report → {hr_path}")
    else:
        print(f"\n  No rows need hand review")

    # ── Metadata JSON ─────────────────────────────────────────────────────────
    meta = {
        "run_timestamp":   stamp,
        "cohort":          args.cohort,
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
        "needs_hand_review": len(all_hand_review_events),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Metadata      → {out_dir / 'metadata.json'}")

    # Sample output
    display_cols = (["row", "text"] + active_panel_models +
                    ["unanimous", "arbiter_code", "final_code", "final_complaint",
                     "needs_hand_review"])
    print(f"\nSample (first 10 rows):")
    print(merged[[c for c in display_cols if c in merged.columns]]
          .head(10).to_string(index=False))

    # Arbitrated rows' rationale — the audit trail to spot-check adjudication
    # quality on cohorts with no gold-standard labels available.
    arbitrated = merged[merged["arbiter_code"].notna()] if "arbiter_code" in merged.columns else merged.iloc[0:0]
    if len(arbitrated):
        print(f"\nArbitrated rows (first 10, with rationale):")
        print(arbitrated[["row", "text", "arbiter_code", "arbiter_rationale"]]
              .head(10).to_string(index=False))


if __name__ == "__main__":
    main()
