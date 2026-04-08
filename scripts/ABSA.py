#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm.auto import tqdm

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[1]
SYSTEM = (
    "You are a careful multilingual newspaper-analysis assistant. "
    "You classify whether a paragraph is positive, neutral, or negative toward geothermal energy specifically."
)
SENTIMENT_MAP = {
    "positive": "positive",
    "pos": "positive",
    "supportive": "positive",
    "neutral": "neutral",
    "neu": "neutral",
    "mixed": "neutral",
    "uncertain": "neutral",
    "negative": "negative",
    "neg": "negative",
    "critical": "negative",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=str, default=str(DEFAULT_PROJECT_DIR))
    ap.add_argument("--input-csv", type=str, default="output/text/paragraph_locations_ollama.csv")
    ap.add_argument("--output-csv", type=str, default="output/text/paragraph_sentiment_llm.csv")
    ap.add_argument("--checkpoint", type=str, default="cache/sentiment_checkpoint.parquet")
    ap.add_argument("--cache", type=str, default="cache/sentiment_cache.jsonl")
    ap.add_argument("--partial-csv", type=str, default="cache/sentiment_partial_results.csv")
    ap.add_argument("--ollama-url", type=str, default="http://localhost:11434/api/generate")
    ap.add_argument("--model", type=str, default="llama3.1:8b")
    ap.add_argument("--country", type=str, default="")
    ap.add_argument("--text-col", type=str, default="paragraph_text")
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--sleep-s", type=float, default=0.0)
    return ap.parse_args()


def make_uid(row: Dict[str, object]) -> str:
    if str(row.get("uid", "")).strip():
        return str(row["uid"])
    base = "||".join(
        [
            str(row.get("source", "")),
            str(row.get("document_title", "")),
            str(row.get("publish_date", "")),
            str(row.get("paragraph_id", "")),
            str(row.get("paragraph_text", "")),
        ]
    )
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def _fingerprint(text: str, country: str) -> str:
    h = hashlib.sha256()
    h.update((country or "").encode("utf-8"))
    h.update(b"\n")
    h.update((text or "").encode("utf-8"))
    return h.hexdigest()


def save_checkpoint(out: pd.DataFrame, checkpoint_path: Path, partial_csv_path: Optional[Path]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    to_store = out.copy()
    if to_store.index.name == "uid" and "uid" in to_store.columns:
        to_store = to_store.reset_index(drop=True)
    to_store.to_parquet(tmp_path, index=True)
    tmp_path.replace(checkpoint_path)
    if partial_csv_path is not None:
        partial_csv_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(partial_csv_path, index=False)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError, requests.HTTPError)),
)
def llm_geothermal_sentiment(
    text: str,
    country: str,
    ollama_url: str,
    model: str,
) -> dict:
    prompt = f"""
Task: Classify the stance of this newspaper paragraph toward geothermal energy specifically.

Context:
- Main country of interest: {country or "not specified"}
- The paragraph has already been identified as being about geothermal energy.

Label definitions:
- positive: the paragraph is supportive of geothermal, presents it as beneficial/desirable/feasible, or reports approval.
- negative: the paragraph is critical of geothermal, presents it as risky/harmful/undesirable, or reports opposition/rejection.
- neutral: the paragraph is mainly factual, balanced, mixed, or does not clearly lean positive or negative.

Rules:
- Judge sentiment toward geothermal specifically, not the overall mood of the writing.
- If benefits and drawbacks are both presented without a clear dominant stance, return neutral.
- Use the full paragraph context.
- Output valid JSON only with keys: sentiment, confidence, evidence_short
- sentiment must be one of: positive, neutral, negative
- confidence must be a number from 0 to 1
- evidence_short must be 20 words or fewer

Paragraph:
{text}
""".strip()

    payload = {
        "model": model,
        "system": SYSTEM,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 120},
    }

    r = requests.post(ollama_url, json=payload, timeout=120)
    r.raise_for_status()
    out = (r.json().get("response") or "").strip()

    start = out.find("{")
    end = out.rfind("}")
    if start == -1 or end == -1:
        return {"sentiment": "neutral", "confidence": 0.0, "evidence_short": "No JSON returned."}

    try:
        obj = json.loads(out[start:end + 1])
    except json.JSONDecodeError:
        return {"sentiment": "neutral", "confidence": 0.0, "evidence_short": "Invalid JSON."}

    raw_sentiment = str(obj.get("sentiment", "neutral")).strip().lower()
    sentiment = SENTIMENT_MAP.get(raw_sentiment, "neutral")

    try:
        confidence = float(obj.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    evidence = str(obj.get("evidence_short", "") or "").strip()
    return {"sentiment": sentiment, "confidence": confidence, "evidence_short": evidence}


def batch_sentiment_resumable(
    df: pd.DataFrame,
    text_col: str,
    checkpoint_path: Path,
    cache_path: Path,
    partial_csv_path: Optional[Path],
    save_every: int,
    sleep_s: float,
    ollama_url: str,
    model: str,
    country: str,
) -> pd.DataFrame:
    if checkpoint_path.exists():
        out = pd.read_parquet(checkpoint_path)
        if "uid" in out.columns and out.index.name != "uid":
            out = out.set_index("uid", drop=True)
        out = out.reindex(df.index)
        for c in df.columns:
            if c not in out.columns:
                out[c] = df[c]
    else:
        out = df.copy()
        out["sentiment"] = None
        out["sentiment_norm"] = None
        out["sentiment_confidence"] = 0.0
        out["sentiment_evidence_short"] = None
        out["sentiment_status"] = None
        out["sentiment_error"] = None

    cache: Dict[str, dict] = {}
    if cache_path.exists():
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

    def cache_get(key: str):
        return cache.get(key)

    def cache_put(key: str, value: dict):
        if key in cache:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache[key] = value
        with cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")

    def is_done(row) -> bool:
        return row.get("sentiment_status") in ("ok", "empty")

    todo_idx = [i for i, row in out.iterrows() if not is_done(row)]
    pbar = tqdm(todo_idx, desc="Geothermal sentiment", unit="row")
    processed_since_save = 0

    try:
        for i in pbar:
            text = str(out.at[i, text_col] if text_col in out.columns else "") or ""
            if not text.strip():
                out.at[i, "sentiment"] = "neutral"
                out.at[i, "sentiment_norm"] = "neutral"
                out.at[i, "sentiment_confidence"] = 0.0
                out.at[i, "sentiment_evidence_short"] = "Empty paragraph."
                out.at[i, "sentiment_status"] = "empty"
                out.at[i, "sentiment_error"] = None
                processed_since_save += 1
                continue

            key = _fingerprint(text, country)
            cached = cache_get(key)

            try:
                res = cached if cached is not None else llm_geothermal_sentiment(
                    text=text,
                    country=country,
                    ollama_url=ollama_url,
                    model=model,
                )
                if cached is None:
                    cache_put(key, res)

                out.at[i, "sentiment"] = res["sentiment"]
                out.at[i, "sentiment_norm"] = res["sentiment"]
                out.at[i, "sentiment_confidence"] = float(res["confidence"])
                out.at[i, "sentiment_evidence_short"] = res["evidence_short"]
                out.at[i, "sentiment_status"] = "ok"
                out.at[i, "sentiment_error"] = None
            except Exception as exc:
                out.at[i, "sentiment_status"] = "error"
                out.at[i, "sentiment_error"] = repr(exc)

            processed_since_save += 1
            if sleep_s:
                time.sleep(sleep_s)
            if processed_since_save >= save_every:
                save_checkpoint(out, checkpoint_path, partial_csv_path)
                processed_since_save = 0
    except KeyboardInterrupt:
        save_checkpoint(out, checkpoint_path, partial_csv_path)
        raise

    save_checkpoint(out, checkpoint_path, partial_csv_path)
    return out


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    os.chdir(project_dir)

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    checkpoint = Path(args.checkpoint)
    cache = Path(args.cache)
    partial_csv = Path(args.partial_csv)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    cache.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    if args.text_col not in df.columns:
        raise ValueError(f"Expected text column '{args.text_col}' in input CSV.")

    if "uid" not in df.columns:
        df["uid"] = df.apply(make_uid, axis=1)
    df = df.set_index("uid", drop=False)

    out = batch_sentiment_resumable(
        df=df,
        text_col=args.text_col,
        checkpoint_path=checkpoint,
        cache_path=cache,
        partial_csv_path=partial_csv,
        save_every=args.save_every,
        sleep_s=args.sleep_s,
        ollama_url=args.ollama_url,
        model=args.model,
        country=args.country,
    )
    out = out.reset_index(drop=True)
    out.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Wrote: {output_csv} (rows={len(out)})")


if __name__ == "__main__":
    main()
