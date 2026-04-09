#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=str, default=str(DEFAULT_PROJECT_DIR))
    ap.add_argument("--input-csv", type=str, default="output/text/sentences_with_categories_admin.csv")
    ap.add_argument("--output-csv", type=str, default="annotation/sentences_for_annotation.csv")
    ap.add_argument("--province-filter-regex", type=str, default="")
    ap.add_argument("--exclude-neutral", type=str, default="True")
    return ap.parse_args()


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def normalize_sentiment(series: pd.Series) -> pd.Series:
    mapping = {
        "positive": "positive",
        "pos": "positive",
        "negative": "negative",
        "neg": "negative",
        "neutral": "neutral",
        "neu": "neutral",
        "neutral/uncertain": "neutral",
    }
    return series.astype(str).str.strip().str.lower().map(mapping)


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    os.chdir(project_dir)

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    uid_col = "sentence_uid" if "sentence_uid" in df.columns else ("paragraph_uid" if "paragraph_uid" in df.columns else "uid")
    text_col = "sentence_text" if "sentence_text" in df.columns else "paragraph_text"
    required_cols = {uid_col, text_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {sorted(missing)}")

    sentiment_source_col = "sentiment_norm" if "sentiment_norm" in df.columns else "sentiment"
    if sentiment_source_col not in df.columns:
        raise ValueError("Expected either 'sentiment_norm' or 'sentiment' column in input CSV.")

    out = df.copy()
    out["_sent_norm"] = normalize_sentiment(out[sentiment_source_col])

    if args.province_filter_regex and "province_name" in out.columns:
        out = out[out["province_name"].astype(str).str.contains(args.province_filter_regex, case=False, na=False)].copy()

    if parse_bool(args.exclude_neutral):
        out = out[out["_sent_norm"].isin(["positive", "negative"])].copy()

    out = out.drop_duplicates(subset=[uid_col]).copy()

    out = out.reset_index(drop=True)
    out["sentence_uid"] = out[uid_col]
    out["sentence_text"] = out[text_col]
    out["annotation_id"] = range(1, len(out) + 1)
    out["predicted_sentiment"] = out[sentiment_source_col]
    if "matched_categories_str" not in out.columns:
        out["matched_categories_str"] = None
    if "matched_keywords_str" not in out.columns:
        out["matched_keywords_str"] = None
    if "llm_location" not in out.columns:
        out["llm_location"] = None
    if "geo_name_matched" not in out.columns:
        out["geo_name_matched"] = None
    if "paragraph_text" not in out.columns:
        out["paragraph_text"] = out[text_col]
    out = out.drop(columns=["_sent_norm"])

    out.to_csv(output_csv, index=False)
    print(f"Wrote: {output_csv} (rows={len(out)})")


if __name__ == "__main__":
    main()
