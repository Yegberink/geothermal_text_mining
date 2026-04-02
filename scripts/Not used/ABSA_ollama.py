#!/usr/bin/env python3
"""
Sentence-level ABSA (Aspect-Based Sentiment Analysis) via Ollama, resumable + cached.

Input:  output/text/sentences_stageA_geo_flagged.csv
Output: output/text/sentences_absa_ollama.csv
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm.auto import tqdm


# =============================
# CONFIG (edit these)
# =============================

PROJECT_DIR = Path("/Users/Yannick/Documents/PhD/text_mining/geothermal")
os.chdir(PROJECT_DIR)

INPUT_SENTENCES_CSV = PROJECT_DIR / "output" / "text" / "sentences_stageA_geo_flagged.csv"

OUTPUT_DIR = PROJECT_DIR / "output" / "text"
OUTPUT_ABSA_CSV = OUTPUT_DIR / "sentences_absa_ollama.csv"
PARTIAL_ABSA_CSV = OUTPUT_DIR / "sentences_absa_partial.csv"

CHECKPOINT_PARQUET = OUTPUT_DIR / "absa_checkpoint.parquet"
CACHE_JSONL = OUTPUT_DIR / "absa_cache.jsonl"

# Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"
SLEEP_S = 0.0
SAVE_EVERY = 50

# Which sentences to analyze
ANALYZE_ONLY_GEO_RELEVANT = True  # recommended
GEO_RELEVANT_LABELS = {"direct", "contextual"}

# ABSA categories (soft scaffold)
ASPECT_CATEGORIES = [
    "economics",
    "environment",
    "safety_risk",
    "technology",
    "governance_policy",
    "social_acceptance",
    "project_feasibility",
    "energy_system_comparison",
    "other",
]

SYSTEM = (
    "You are an assistant that performs aspect-based sentiment analysis on Dutch newspaper sentences "
    "about energy technologies, especially geothermal energy. You extract opinions without relying "
    "on sentiment keyword lists."
)


# =============================
# Utilities
# =============================

def _fingerprint(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()

def safe_str(x) -> str:
    return "" if x is None else str(x)

def add_prev_next_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds prev_sentence and next_sentence within each paragraph.
    Tries to group by a paragraph identifier if present, otherwise falls back to paragraph_text.
    """
    df = df.copy()

    # Try to find a stable paragraph grouping key
    group_cols = None
    for cand in (["uid"], ["source", "document_title", "publish_date", "paragraph_id"], ["paragraph_text"]):
        if all(c in df.columns for c in cand):
            group_cols = cand
            break
    if group_cols is None:
        # last resort: treat entire df as one group
        df["_grp"] = 0
        group_cols = ["_grp"]

    # Ensure deterministic order within paragraph group
    if "sentence_id" in df.columns:
        df = df.sort_values(group_cols + ["sentence_id"])
    else:
        # if you don't have sentence_id, keep current row order within group
        df["_row_order"] = range(len(df))
        df = df.sort_values(group_cols + ["_row_order"])

    df["prev_sentence"] = df.groupby(group_cols)["sentence_text"].shift(1).fillna("")
    df["next_sentence"] = df.groupby(group_cols)["sentence_text"].shift(-1).fillna("")
    return df


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError, requests.HTTPError)),
)
def llm_absa_sentence(sentence: str, prev_sentence: str, next_sentence: str) -> dict:
    cats = ", ".join(ASPECT_CATEGORIES)
    prompt = f"""
Task:
Analyze the SENTENCE below and determine whether it expresses an evaluation, opinion, concern, expectation, or stance.
If yes, extract the evaluated aspect(s).

Rules:
- If the sentence is purely factual/descriptive, set is_opinion=false and aspects=[].
- Extract only aspects that are actually evaluated.
- aspect_phrase must be a short phrase copied or minimally normalized from the sentence.
- aspect_category must be ONE of: {cats}
- polarity must be one of: positive, negative, mixed, neutral
- certainty must be a number from 0 to 1
- Output MUST be valid JSON only with keys: is_opinion, aspects, reasoning_short
- aspects is a list of objects with keys: aspect_phrase, aspect_category, polarity, certainty

SENTENCE:
{sentence}

CONTEXT (optional):
PREV: {prev_sentence}
NEXT: {next_sentence}
""".strip()

    payload = {
        "model": MODEL,
        "system": SYSTEM,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 220},
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    out = (r.json().get("response") or "").strip()

    start = out.find("{")
    end = out.rfind("}")
    if start == -1 or end == -1:
        return {"is_opinion": False, "aspects": [], "reasoning_short": "No JSON returned."}

    try:
        obj = json.loads(out[start:end + 1])
    except json.JSONDecodeError:
        return {"is_opinion": False, "aspects": [], "reasoning_short": "Invalid JSON."}

    # Light validation / normalization
    is_op = bool(obj.get("is_opinion", False))
    aspects = obj.get("aspects", []) or []
    if not isinstance(aspects, list):
        aspects = []

    cleaned_aspects = []
    for a in aspects:
        if not isinstance(a, dict):
            continue
        phrase = safe_str(a.get("aspect_phrase")).strip()
        cat = safe_str(a.get("aspect_category")).strip()
        pol = safe_str(a.get("polarity")).strip().lower()
        cert = a.get("certainty", 0.0)
        try:
            cert = float(cert)
        except Exception:
            cert = 0.0
        cert = max(0.0, min(1.0, cert))

        if not phrase:
            continue
        if cat not in ASPECT_CATEGORIES:
            cat = "other"
        if pol not in {"positive", "negative", "mixed", "neutral"}:
            pol = "neutral"

        cleaned_aspects.append(
            {"aspect_phrase": phrase, "aspect_category": cat, "polarity": pol, "certainty": cert}
        )

    return {
        "is_opinion": is_op and len(cleaned_aspects) > 0,
        "aspects": cleaned_aspects,
        "reasoning_short": safe_str(obj.get("reasoning_short")).strip(),
    }


def load_cache(cache_path: Path) -> Dict[str, dict]:
    cache: Dict[str, dict] = {}
    if not cache_path.exists():
        return cache
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cache[obj["key"]] = obj["value"]
            except Exception:
                pass
    return cache

def cache_put(cache_path: Path, cache: Dict[str, dict], key: str, value: dict) -> None:
    if key in cache:
        return
    cache[key] = value
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


# =============================
# Main
# =============================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_SENTENCES_CSV.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_SENTENCES_CSV}")

    df = pd.read_csv(INPUT_SENTENCES_CSV, encoding="utf-8")
    # Expect at least these
    if "sentence_text" not in df.columns:
        raise ValueError("Input CSV must contain a 'sentence_text' column.")
    if "paragraph_text" not in df.columns:
        df["paragraph_text"] = ""

    # Optional geo relevance filter
    if ANALYZE_ONLY_GEO_RELEVANT and "geo_relevance" in df.columns:
        before = len(df)
        df = df[df["geo_relevance"].isin(GEO_RELEVANT_LABELS)].copy()
        print(f"[main] Geo-relevant sentences: {before} -> {len(df)}")

    # Add prev/next context (cheap)
    df = add_prev_next_sentences(df)

    # Load/resume checkpoint
    if CHECKPOINT_PARQUET.exists():
        out = pd.read_parquet(CHECKPOINT_PARQUET)
        out = out.reindex(df.index)
        # ensure any new columns exist
        for c in df.columns:
            if c not in out.columns:
                out[c] = df[c]
        print(f"[resume] Loaded checkpoint: {CHECKPOINT_PARQUET} (rows={len(out)})")
    else:
        out = df.copy()
        out["absa_is_opinion"] = False
        out["absa_aspects_json"] = "[]"
        out["absa_reasoning_short"] = ""
        out["absa_status"] = None
        out["absa_error"] = None

    cache = load_cache(CACHE_JSONL)

    def is_done(row) -> bool:
        return row.get("absa_status") in ("ok", "empty")

    todo_idx = [i for i, r in out.iterrows() if not is_done(r)]
    print(f"[main] To process: {len(todo_idx)} / {len(out)}")

    ok = int((out.get("absa_status") == "ok").sum()) if "absa_status" in out.columns else 0
    err = int((out.get("absa_status") == "error").sum()) if "absa_status" in out.columns else 0

    processed_since_save = 0
    pbar = tqdm(todo_idx, desc="ABSA (Ollama)", unit="sent")

    try:
        for i in pbar:
            sent = safe_str(out.at[i, "sentence_text"]).strip()
            prev_s = safe_str(out.at[i, "prev_sentence"]).strip()
            next_s = safe_str(out.at[i, "next_sentence"]).strip()

            if not sent:
                out.at[i, "absa_status"] = "empty"
                out.at[i, "absa_is_opinion"] = False
                out.at[i, "absa_aspects_json"] = "[]"
                out.at[i, "absa_reasoning_short"] = "Empty sentence."
                out.at[i, "absa_error"] = None
                processed_since_save += 1
                continue

            key = _fingerprint(MODEL, sent, prev_s, next_s)
            cached = cache.get(key)

            try:
                res = cached if cached is not None else llm_absa_sentence(sent, prev_s, next_s)
                if cached is None:
                    cache_put(CACHE_JSONL, cache, key, res)

                out.at[i, "absa_is_opinion"] = bool(res.get("is_opinion", False))
                out.at[i, "absa_aspects_json"] = json.dumps(res.get("aspects", []), ensure_ascii=False)
                out.at[i, "absa_reasoning_short"] = safe_str(res.get("reasoning_short"))
                out.at[i, "absa_status"] = "ok"
                out.at[i, "absa_error"] = None
                ok += 1
            except Exception as e:
                out.at[i, "absa_status"] = "error"
                out.at[i, "absa_error"] = repr(e)
                err += 1

            processed_since_save += 1
            pbar.set_postfix({"ok": ok, "error": err, "cached": cached is not None})

            if SLEEP_S:
                time.sleep(SLEEP_S)

            if processed_since_save >= SAVE_EVERY:
                out.to_parquet(CHECKPOINT_PARQUET, index=True)
                out.to_csv(PARTIAL_ABSA_CSV, index=False, encoding="utf-8")
                processed_since_save = 0

    except KeyboardInterrupt:
        out.to_parquet(CHECKPOINT_PARQUET, index=True)
        out.to_csv(PARTIAL_ABSA_CSV, index=False, encoding="utf-8")
        print(
            f"\nStopped by user. Progress saved to:\n"
            f"- {CHECKPOINT_PARQUET}\n"
            f"- {PARTIAL_ABSA_CSV}\n"
        )
        return  # <-- only exit early on interrupt

    # Normal final save (only reached if no interrupt)
    out.to_parquet(CHECKPOINT_PARQUET, index=True)
    out.to_csv(OUTPUT_ABSA_CSV, index=False, encoding="utf-8")
    print(f"[main] Wrote: {OUTPUT_ABSA_CSV} (rows={len(out)})")




if __name__ == "__main__":
    main()
