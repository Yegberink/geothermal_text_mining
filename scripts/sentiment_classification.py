#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from tqdm.auto import tqdm
from transformers import pipeline

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=str, default=str(DEFAULT_PROJECT_DIR))
    ap.add_argument("--input-csv", type=str, default="output/text/sentence_locations_ollama.csv")
    ap.add_argument("--output-csv", type=str, default="output/text/sentence_sentiment_llm.csv")
    ap.add_argument("--checkpoint", type=str, default="cache/sentence_sentiment_checkpoint.parquet")
    ap.add_argument("--cache", type=str, default="cache/sentence_sentiment_cache.jsonl")
    ap.add_argument("--partial-csv", type=str, default="cache/sentence_sentiment_partial_results.csv")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--text-col", type=str, default="sentence_text")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    return ap.parse_args()


def make_uid(row: Dict[str, object]) -> str:
    if str(row.get("sentence_uid", "")).strip():
        return str(row["sentence_uid"])
    if str(row.get("uid", "")).strip():
        return str(row["uid"])
    base = "||".join(
        [
            str(row.get("source", "")),
            str(row.get("document_title", "")),
            str(row.get("publish_date", "")),
            str(row.get("paragraph_uid", "")),
            str(row.get("sentence_id", "")),
            str(row.get("sentence_text", "")),
            str(row.get("paragraph_id", "")),
            str(row.get("paragraph_text", "")),
        ]
    )
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def _fingerprint(text: str) -> str:
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()


def save_checkpoint(out: pd.DataFrame, checkpoint_path: Path, partial_csv_path: Optional[Path]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    to_store = out.copy()
    if to_store.index.name and to_store.index.name in to_store.columns:
        to_store = to_store.reset_index(drop=True)
    to_store.to_parquet(tmp_path, index=True)
    tmp_path.replace(checkpoint_path)
    if partial_csv_path is not None:
        partial_csv_path.parent.mkdir(parents=True, exist_ok=True)
        partial_out = out.copy()
        if partial_out.index.name and partial_out.index.name in partial_out.columns:
            partial_out = partial_out.reset_index(drop=True)
        partial_out.to_csv(partial_csv_path, index=False, encoding="utf-8")


def _install_interrupt_handlers() -> None:
    def _handle_interrupt(signum, _frame):
        signame = signal.Signals(signum).name
        raise KeyboardInterrupt(f"Received {signame}")

    signal.signal(signal.SIGINT, _handle_interrupt)
    signal.signal(signal.SIGTERM, _handle_interrupt)


def load_classifier(model_name: str):
    return pipeline(
        task="sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
    )


def parse_star_rating(label: str) -> Optional[int]:
    match = re.search(r"([1-5])", str(label))
    if not match:
        return None
    return int(match.group(1))


def map_sentiment(label: str) -> str:
    stars = parse_star_rating(label)
    if stars is None:
        return "neutral"
    if stars <= 2:
        return "negative"
    if stars == 3:
        return "neutral"
    return "positive"


def classify_texts(classifier, texts: list[str], batch_size: int, max_length: int) -> list[dict]:
    preds = classifier(
        texts,
        batch_size=batch_size,
        truncation=True,
        max_length=max_length,
    )
    results = []
    for pred in preds:
        raw_label = str(pred.get("label", "") or "").strip()
        score = float(pred.get("score", 0.0) or 0.0)
        sentiment = map_sentiment(raw_label)
        results.append(
            {
                "sentiment": sentiment,
                "confidence": max(0.0, min(1.0, score)),
                "evidence_short": raw_label or "model_label_missing",
            }
        )
    return results


def batch_sentiment_resumable(
    df: pd.DataFrame,
    text_col: str,
    checkpoint_path: Path,
    cache_path: Path,
    partial_csv_path: Optional[Path],
    batch_size: int,
    max_length: int,
    model_name: str,
) -> pd.DataFrame:
    if checkpoint_path.exists():
        out = pd.read_parquet(checkpoint_path)
        if out.index.name and out.index.name in out.columns:
            out = out.set_index(out.index.name, drop=False)
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
    pbar = tqdm(total=len(todo_idx), desc="Sentence sentiment", unit="row")
    classifier = load_classifier(model_name)

    try:
        for start in range(0, len(todo_idx), batch_size):
            batch_idx = todo_idx[start : start + batch_size]
            uncached_idx: list[object] = []
            uncached_texts: list[str] = []

            for i in batch_idx:
                text = str(out.at[i, text_col] if text_col in out.columns else "") or ""
                if not text.strip():
                    out.at[i, "sentiment"] = "neutral"
                    out.at[i, "sentiment_norm"] = "neutral"
                    out.at[i, "sentiment_confidence"] = 0.0
                    out.at[i, "sentiment_evidence_short"] = "Empty sentence."
                    out.at[i, "sentiment_status"] = "empty"
                    out.at[i, "sentiment_error"] = None
                    continue

                key = _fingerprint(text)
                cached = cache_get(key)
                if cached is not None:
                    out.at[i, "sentiment"] = cached["sentiment"]
                    out.at[i, "sentiment_norm"] = cached["sentiment"]
                    out.at[i, "sentiment_confidence"] = float(cached["confidence"])
                    out.at[i, "sentiment_evidence_short"] = cached["evidence_short"]
                    out.at[i, "sentiment_status"] = "ok"
                    out.at[i, "sentiment_error"] = None
                else:
                    uncached_idx.append(i)
                    uncached_texts.append(text)

            if uncached_texts:
                try:
                    batch_results = classify_texts(
                        classifier=classifier,
                        texts=uncached_texts,
                        batch_size=batch_size,
                        max_length=max_length,
                    )
                    for i, text, res in zip(uncached_idx, uncached_texts, batch_results):
                        cache_put(_fingerprint(text), res)
                        out.at[i, "sentiment"] = res["sentiment"]
                        out.at[i, "sentiment_norm"] = res["sentiment"]
                        out.at[i, "sentiment_confidence"] = float(res["confidence"])
                        out.at[i, "sentiment_evidence_short"] = res["evidence_short"]
                        out.at[i, "sentiment_status"] = "ok"
                        out.at[i, "sentiment_error"] = None
                except Exception as exc:
                    for i in uncached_idx:
                        out.at[i, "sentiment_status"] = "error"
                        out.at[i, "sentiment_error"] = repr(exc)

            pbar.update(len(batch_idx))
            save_checkpoint(out, checkpoint_path, partial_csv_path)
    except KeyboardInterrupt:
        save_checkpoint(out, checkpoint_path, partial_csv_path)
        raise
    finally:
        pbar.close()

    save_checkpoint(out, checkpoint_path, partial_csv_path)
    return out


def sentiment_result_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        "sentiment",
        "sentiment_norm",
        "sentiment_confidence",
        "sentiment_evidence_short",
        "sentiment_status",
        "sentiment_error",
    ]
    return [c for c in cols if c in df.columns]


def main() -> None:
    _install_interrupt_handlers()
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

    source_df = df.copy()
    dedup_df = df.drop_duplicates(subset=[args.text_col], keep="first").copy()

    if "uid" not in dedup_df.columns and "sentence_uid" not in dedup_df.columns:
        dedup_df["uid"] = dedup_df.apply(make_uid, axis=1)
    uid_col = "sentence_uid" if "sentence_uid" in dedup_df.columns else "uid"
    dedup_df = dedup_df.set_index(uid_col, drop=False)

    try:
        dedup_out = batch_sentiment_resumable(
            df=dedup_df,
            text_col=args.text_col,
            checkpoint_path=checkpoint,
            cache_path=cache,
            partial_csv_path=partial_csv,
            batch_size=args.batch_size,
            max_length=args.max_length,
            model_name=args.model,
        )
    except KeyboardInterrupt as exc:
        print(str(exc) or "Interrupted.", file=sys.stderr)
        print(f"Saved intermediate CSV: {partial_csv}", file=sys.stderr)
        print(f"Saved checkpoint: {checkpoint}", file=sys.stderr)
        raise SystemExit(130)

    result_cols = sentiment_result_columns(dedup_out)
    dedup_results = dedup_out.reset_index(drop=True)[[args.text_col] + result_cols].copy()
    out = source_df.merge(dedup_results, on=args.text_col, how="left")
    out = out.reset_index(drop=True)
    out.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Wrote: {output_csv} (rows={len(out)})")


if __name__ == "__main__":
    main()
