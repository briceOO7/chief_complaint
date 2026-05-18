#!/usr/bin/env python3
"""
GPT-5 third reviewer: resolves disagreements and vague-code agreements
between the GPT-4o silver labels (reviewer 1) and Gemini (reviewer 2).

For each row it receives the complaint text and both prior labels, then
returns a final ruling with reasoning.

Usage:
    # First 100 rows:
    python scripts/llm_cedis_third_reviewer.py --max-rows 100

    # All rows:
    python scripts/llm_cedis_third_reviewer.py --all

    # Resume an interrupted run:
    python scripts/llm_cedis_third_reviewer.py --all --resume

    # Different model:
    python scripts/llm_cedis_third_reviewer.py --all --model gpt-5
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from llm_cedis_second_labeller import (
    MODEL_CONFIGS, ALL_MODELS,
    build_client, load_cedis,
    batch_size_for, max_tokens_for_batch,
    extract_json, _to_code_str,
)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

DISAGREEMENTS_CSV = Path(
    "output/review_disagreements_gemini_2.0_flash/disagreements_all_for_reviewer3.csv"
)
OUT_DIR = Path("output/gpt5_third_reviewer")
DEFAULT_MODEL = "gpt-5-mini"

RATE_LIMIT_SLEEP = 0.3
ERROR_SLEEP = 1.0
MAX_CORRECTION_PASSES = 2

# Third-reviewer token budgets: adjudication task + full CEDIS list in prompt.
# The CEDIS list adds ~800 tokens of system-prompt overhead.
THIRD_REVIEWER_TOKEN_OVERRIDES: dict[str, dict] = {
    "gpt-5":      {"tokens_per_complaint": 800, "tokens_floor": 4000},
    "gpt-5-mini": {"tokens_per_complaint": 800, "tokens_floor": 4000},
    "gpt-5-nano": {"tokens_per_complaint": 600, "tokens_floor": 3000},
}


# ─────────────────────────────────────────────────────────────────────────────
# Valid-code set (loaded once from data/cedis_codes.csv)
# ─────────────────────────────────────────────────────────────────────────────

def load_valid_codes() -> set[int]:
    """Return the set of valid integer CEDIS codes from data/cedis_codes.csv."""
    codes_path = Path(__file__).parent.parent / "data" / "cedis_codes.csv"
    df = pd.read_csv(codes_path)
    return set(df["code"].dropna().astype(int).tolist())


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

def build_system_prompt(cedis_code_list: str) -> str:
    """Full CEDIS list is always included — hallucinations are not acceptable."""
    return (
        "You are a senior clinical informaticist and the final arbitrator for "
        "CEDIS (Canadian Emergency Department Information System) classification disputes.\n\n"
        "Two independent AI reviewers have already classified each chief complaint. "
        "Your job is to review their labels and make the definitive ruling.\n\n"
        "CRITICAL RULES:\n"
        "- You MUST return ONLY codes from the valid CEDIS list below. "
        "Any code not on this list is an error.\n"
        "- Prefer the most specific code that fits the complaint.\n"
        "- Avoid code 866 (Minor complaints NOS) when a more specific code fits.\n"
        "- Avoid code 999 (Unknown) unless the complaint is truly uninterpretable.\n"
        "- You may agree with either reviewer or choose a third code if both are wrong.\n"
        "- Provide a brief rationale (1 sentence) for your ruling.\n\n"
        "Valid CEDIS codes (ONLY use codes from this list):\n"
        f"{cedis_code_list}"
    )


def build_correction_prompt(cases: list[dict], valid_codes_str: str, json_mode: bool) -> str:
    """Prompt sent when a previous response contained invalid (hallucinated) codes."""
    lines = []
    for i, c in enumerate(cases, 1):
        r1 = f"{c['r1_code']} ({c['r1_complaint']})" if c['r1_code'] else "none"
        r2 = f"{c['r2_code']} ({c['r2_complaint']})" if c['r2_code'] else "none"
        bad = c.get("bad_code", "?")
        lines.append(
            f"{i}. Complaint: \"{c['text']}\"\n"
            f"   Reviewer 1 (GPT-4o): {r1}\n"
            f"   Reviewer 2 (Gemini): {r2}\n"
            f"   ⚠ Your previous answer ({bad}) is NOT a valid CEDIS code."
        )
    cases_text = "\n\n".join(lines)
    json_instruction = (
        f'Return ONLY valid JSON: {{"results": [array of {len(cases)} objects]}}. '
        f"Each object: cedis_code (integer), cedis_complaint (string), rationale (string ≤15 words)."
        if json_mode else
        f"Return a JSON object with key \"results\" containing {len(cases)} objects. "
        f"Each: cedis_code (integer from the list), cedis_complaint (string), rationale (≤15 words). "
        f"Output JSON only."
    )
    return (
        f"CORRECTION REQUIRED. You previously returned invalid CEDIS codes. "
        f"Re-review EACH case and return a valid code from the CEDIS list ONLY.\n"
        f"{json_instruction}\n\n{cases_text}"
    )


def build_user_message(cases: list[dict], json_mode: bool) -> str:
    lines = []
    for i, c in enumerate(cases, 1):
        r1 = f"{c['r1_code']} ({c['r1_complaint']})" if c['r1_code'] else "none"
        r2 = f"{c['r2_code']} ({c['r2_complaint']})" if c['r2_code'] else "none"
        reason = c.get("review_reason", "disagree")
        lines.append(
            f"{i}. Complaint: \"{c['text']}\"\n"
            f"   Reviewer 1 (GPT-4o): {r1}\n"
            f"   Reviewer 2 (Gemini): {r2}\n"
            f"   Reason flagged: {reason}"
        )
    cases_text = "\n\n".join(lines)
    json_instruction = (
        f'Return ONLY valid JSON: {{"results": [array of {len(cases)} objects]}}. '
        f"Each object: cedis_code (integer), cedis_complaint (string), "
        f"rationale (string, ≤15 words)."
        if json_mode else
        f"Return a JSON object with key \"results\" containing {len(cases)} objects. "
        f"Each: cedis_code (integer), cedis_complaint (string), rationale (string ≤15 words). "
        f"Output JSON only."
    )
    return f"Rule on ALL {len(cases)} cases below.\n{json_instruction}\n\n{cases_text}"


# ─────────────────────────────────────────────────────────────────────────────
# API call
# ─────────────────────────────────────────────────────────────────────────────

def _third_review_max_tokens(model: str, n: int) -> int:
    overrides = THIRD_REVIEWER_TOKEN_OVERRIDES.get(model, {})
    cfg = {**MODEL_CONFIGS[model], **overrides}
    return max(cfg["tokens_floor"], cfg["tokens_per_complaint"] * n)


def _call_api(client, model: str, system_prompt: str, user_msg: str, max_tok: int) -> str:
    """Single API call, returns raw content string."""
    cfg = MODEL_CONFIGS[model]
    if cfg["backend"] == "gemini":
        return client.complete(
            system_prompt=system_prompt,
            user_message=user_msg,
            max_output_tokens=max_tok,
            temperature=cfg["temperature"] or 0,
            json_mode=cfg["json_mode"],
        )
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
            f"Empty response (finish_reason={response.choices[0].finish_reason})"
        )
    return content


def review_batch(
    client, model: str, cases: list[dict], system_prompt: str,
    correction: bool = False, valid_codes_str: str = "",
) -> list[dict]:
    cfg = MODEL_CONFIGS[model]
    if correction:
        user_msg = build_correction_prompt(cases, valid_codes_str, cfg["json_mode"])
    else:
        user_msg = build_user_message(cases, cfg["json_mode"])
    max_tok = _third_review_max_tokens(model, len(cases))
    content = _call_api(client, model, system_prompt, user_msg, max_tok)
    parsed = extract_json(content)
    results = parsed.get("results", [])
    if len(results) != len(cases):
        raise ValueError(f"Expected {len(cases)} results, got {len(results)}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Hallucination validation + correction
# ─────────────────────────────────────────────────────────────────────────────

def validate_and_correct(
    result_map: dict[int, dict],
    cases_by_row: dict[int, dict],
    client,
    model: str,
    system_prompt: str,
    valid_codes: set[int],
    cedis_code_list: str,
) -> tuple[dict[int, dict], int]:
    """
    Check result_map for hallucinated codes and re-send offending rows to
    the LLM for correction. Returns updated result_map and total corrections made.
    """
    total_corrected = 0

    for pass_num in range(1, MAX_CORRECTION_PASSES + 1):
        bad_rows = [
            row_num for row_num, res in result_map.items()
            if res["reviewer3_final_code"] is not None
            and res["reviewer3_final_code"] not in valid_codes
        ]
        if not bad_rows:
            break

        print(f"\n  [correction pass {pass_num}] {len(bad_rows)} hallucinated code(s): "
              f"{[result_map[r]['reviewer3_final_code'] for r in bad_rows]}", end=" ")

        for row_num in bad_rows:
            case = {
                **cases_by_row[row_num],
                "bad_code": result_map[row_num]["reviewer3_final_code"],
            }
            try:
                preds = review_batch(
                    client, model, [case], system_prompt,
                    correction=True, valid_codes_str=cedis_code_list,
                )
                pred = preds[0]
                try:
                    code = int(pred.get("cedis_code"))
                    compl = str(pred.get("cedis_complaint", ""))
                except (TypeError, ValueError):
                    code, compl = None, str(pred)
                result_map[row_num] = {
                    "reviewer3_final_code":      code,
                    "reviewer3_final_complaint": compl,
                    "reviewer3_rationale":       str(pred.get("rationale", "")) + " [corrected]",
                }
                total_corrected += 1
            except Exception as exc:
                print(f"\n    row {row_num} correction failed: {exc}", end=" ")
            time.sleep(RATE_LIMIT_SLEEP)

    # Final tally of any still-bad codes after all correction passes
    still_bad = [
        row_num for row_num, res in result_map.items()
        if res["reviewer3_final_code"] is not None
        and res["reviewer3_final_code"] not in valid_codes
    ]
    if still_bad:
        print(f"\n  ⚠ {len(still_bad)} row(s) still have invalid codes after correction: "
              f"{[(r, result_map[r]['reviewer3_final_code']) for r in still_bad]}", end=" ")

    return result_map, total_corrected


# ─────────────────────────────────────────────────────────────────────────────
# Processing
# ─────────────────────────────────────────────────────────────────────────────

def run_third_review(
    rows: pd.DataFrame,
    client,
    model: str,
    system_prompt: str,
    valid_codes: set[int],
    cedis_code_list: str,
) -> tuple[pd.DataFrame, dict]:
    """Send all flagged rows to the third reviewer and return annotated DataFrame + stats."""
    cases = []
    for _, r in rows.iterrows():
        cases.append({
            "row":           r["row"],
            "text":          str(r["text"]),
            "r1_code":       _to_code_str(pd.Series([r["reviewer1_gpt_code"]]))[0],
            "r1_complaint":  str(r.get("reviewer1_gpt_complaint", "")),
            "r2_code":       _to_code_str(pd.Series([r["reviewer2_cursor_code"]]))[0],
            "r2_complaint":  str(r.get("reviewer2_cursor_complaint", "")),
            "review_reason": str(r.get("review_reason", "disagree")),
        })
    cases_by_row = {c["row"]: c for c in cases}

    # GPT-5 reasoning models: single-case calls reduce reasoning-token waste
    cfg = MODEL_CONFIGS[model]
    bs = 1 if cfg["backend"] == "securegpt" else min(batch_size_for(model), 5)
    result_map: dict[int, dict] = {}
    hallucinations_caught = 0

    for batch_start in range(0, len(cases), bs):
        batch = cases[batch_start: batch_start + bs]
        row_nums = [c["row"] for c in batch]

        preds = None
        for attempt in range(3):
            try:
                preds = review_batch(client, model, batch, system_prompt)
                break
            except Exception as exc:
                exc_name = type(exc).__name__
                print(f"\n  batch attempt {attempt+1} ({exc_name}): {exc}", end=" ", flush=True)
                time.sleep(ERROR_SLEEP * (attempt + 1))
        else:
            print(f"\n  falling back to individual for {len(batch)} rows ...", end=" ", flush=True)
            preds = []
            for case in batch:
                for ind_attempt in range(2):
                    try:
                        pred = review_batch(client, model, [case], system_prompt)[0]
                        break
                    except Exception as exc:
                        if ind_attempt == 1:
                            pred = {"cedis_code": None, "cedis_complaint": str(exc), "rationale": ""}
                        time.sleep(ERROR_SLEEP)
                preds.append(pred)
                time.sleep(RATE_LIMIT_SLEEP)

        for row_num, pred in zip(row_nums, preds):
            try:
                code = int(pred.get("cedis_code"))
                compl = str(pred.get("cedis_complaint", ""))
            except (TypeError, ValueError):
                code, compl = None, str(pred)
            result_map[row_num] = {
                "reviewer3_final_code":      code,
                "reviewer3_final_complaint": compl,
                "reviewer3_rationale":       str(pred.get("rationale", "")),
            }

        # Validate this batch immediately and correct any hallucinations
        batch_map = {r: result_map[r] for r in row_nums if r in result_map}
        batch_cases_by_row = {r: cases_by_row[r] for r in row_nums}
        batch_map, corrected = validate_and_correct(
            batch_map, batch_cases_by_row, client, model,
            system_prompt, valid_codes, cedis_code_list,
        )
        result_map.update(batch_map)
        hallucinations_caught += corrected

        time.sleep(RATE_LIMIT_SLEEP)

    out = rows.copy()
    out["reviewer3_final_code"]      = out["row"].map(lambda r: result_map.get(r, {}).get("reviewer3_final_code"))
    out["reviewer3_final_complaint"] = out["row"].map(lambda r: result_map.get(r, {}).get("reviewer3_final_complaint", ""))
    out["reviewer3_rationale"]       = out["row"].map(lambda r: result_map.get(r, {}).get("reviewer3_rationale", ""))

    # Normalise code to nullable int
    out["reviewer3_final_code"] = pd.to_numeric(out["reviewer3_final_code"], errors="coerce").astype("Int64")

    # Verify final validity
    final_invalid = out[
        out["reviewer3_final_code"].notna() &
        ~out["reviewer3_final_code"].apply(lambda x: int(x) if pd.notna(x) else -1).isin(valid_codes)
    ]

    # Tally agreement
    r1 = _to_code_str(out["reviewer1_gpt_code"])
    r2 = _to_code_str(out["reviewer2_cursor_code"])
    r3 = out["reviewer3_final_code"].apply(lambda x: str(int(x)) if pd.notna(x) else "")
    out["r3_agrees_with"] = "neither"
    out.loc[r3 == r1, "r3_agrees_with"] = "reviewer1_gpt"
    out.loc[r3 == r2, "r3_agrees_with"] = "reviewer2_gemini"
    out.loc[(r3 == r1) & (r3 == r2), "r3_agrees_with"] = "both"

    stats = {
        "hallucinations_caught":    hallucinations_caught,
        "still_invalid_after_fix":  len(final_invalid),
    }
    return out, stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPT-5 third reviewer for CEDIS disagreements")
    p.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(ALL_MODELS))
    p.add_argument("--max-rows", type=int, help="Only review rows ≤ N")
    p.add_argument("--limit", type=int, help="Process only the first N flagged rows (for testing)")
    p.add_argument("--all", action="store_true", help="Review all flagged rows")
    p.add_argument("--resume", action="store_true", help="Skip rows already in output")
    p.add_argument("--disagreements", default=str(DISAGREEMENTS_CSV),
                   help="Path to disagreements_all CSV (default: Gemini vs GPT-4o)")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.all and not args.max_rows and not args.limit:
        raise SystemExit("Provide --all, --max-rows N, or --limit N")

    disagree_path = Path(args.disagreements)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "third_review_results.csv"

    rows = pd.read_csv(disagree_path)
    if args.max_rows:
        rows = rows[rows["row"] <= args.max_rows].copy()
    if args.limit:
        rows = rows.head(args.limit).copy()

    if args.resume and out_path.exists():
        done = pd.read_csv(out_path)["row"].tolist()
        rows = rows[~rows["row"].isin(done)].copy()
        print(f"Resuming: {len(done)} already done, {len(rows)} remaining")

    if rows.empty:
        print("Nothing to review.")
        return

    client = build_client(args.model)
    _, cedis_code_list = load_cedis()
    valid_codes = load_valid_codes()
    system_prompt = build_system_prompt(cedis_code_list)

    print(f"Model         : {args.model}", flush=True)
    print(f"Rows to review: {len(rows)}", flush=True)
    print(f"Valid codes   : {len(valid_codes)}", flush=True)
    print(f"Output        : {out_path}", flush=True)
    print("-" * 60, flush=True)

    t0 = time.time()
    result, stats = run_third_review(
        rows, client, args.model, system_prompt, valid_codes, cedis_code_list
    )
    elapsed = time.time() - t0

    # Append to existing if resuming
    if args.resume and out_path.exists():
        existing = pd.read_csv(out_path)
        result = pd.concat([existing, result], ignore_index=True).sort_values("row")

    result.to_csv(out_path, index=False)

    # Summary
    agree_counts = result["r3_agrees_with"].value_counts().to_dict()
    n = len(result)
    null_count = result["reviewer3_final_code"].isna().sum()

    print(f"\n{'='*60}")
    print(f"Completed {n} rows in {elapsed:.0f}s ({elapsed/n:.1f}s/row)")
    print(f"\nCode validity:")
    print(f"  Hallucinations caught + corrected : {stats['hallucinations_caught']}")
    print(f"  Still invalid after correction    : {stats['still_invalid_after_fix']}")
    print(f"  Null/failed codes                 : {null_count}")
    print(f"\nReviewer 3 agreement:")
    print(f"  Agreed with Reviewer 1 (GPT-4o)  : {agree_counts.get('reviewer1_gpt', 0)}")
    print(f"  Agreed with Reviewer 2 (Gemini)  : {agree_counts.get('reviewer2_gemini', 0)}")
    print(f"  Agreed with both                 : {agree_counts.get('both', 0)}")
    print(f"  Chose a different code (neither) : {agree_counts.get('neither', 0)}")
    print(f"\nSample results:")
    cols = ["row", "text", "reviewer1_gpt_code", "reviewer2_cursor_code",
            "reviewer3_final_code", "r3_agrees_with", "reviewer3_rationale"]
    print(result[[c for c in cols if c in result.columns]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
