#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=str, default=str(DEFAULT_PROJECT_DIR))
    ap.add_argument("--input-csv", type=str, default="output/text/sentences_with_categories_admin.csv")
    ap.add_argument("--paragraph-input-csv", type=str, default="")
    ap.add_argument("--paragraph-geothermal-csv", type=str, default="")
    ap.add_argument("--paragraph-location-csv", type=str, default="")
    ap.add_argument("--sentiment-csv", type=str, default="")
    ap.add_argument("--keywords-csv", type=str, default="")
    ap.add_argument("--output-csv", type=str, default="annotation/sentences_for_annotation.csv")
    ap.add_argument("--province-filter-regex", type=str, default="")
    ap.add_argument("--exclude-neutral", type=str, default="True")
    ap.add_argument("--sentence-sample-size", type=int, default=500)
    ap.add_argument("--paragraph-sample-size", type=int, default=100)
    ap.add_argument("--paragraph-location-sample-size", type=int, default=None)
    ap.add_argument("--random-state", type=int, default=42)
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


def parse_keyword_csv(path: Path) -> dict[str, list[str]]:
    keyword_df = pd.read_csv(path)
    keyword_df = keyword_df.loc[:, ~keyword_df.columns.astype(str).str.match(r"^Unnamed")]
    categories = {}
    for col in keyword_df.columns:
        category = str(col).strip()
        if not category:
            continue
        keywords = [
            str(value).strip().lower()
            for value in keyword_df[col].dropna().tolist()
            if str(value).strip()
        ]
        if keywords:
            categories[category] = keywords
    return categories


def add_keyword_predictions(df: pd.DataFrame, keywords_csv: Path | None) -> pd.DataFrame:
    if keywords_csv is None or not keywords_csv.exists():
        return df
    categories = parse_keyword_csv(keywords_csv)
    if not categories:
        return df

    out = df.copy()
    text_col = "sentence_text" if "sentence_text" in out.columns else "paragraph_text"
    texts = out[text_col].fillna("").astype(str).str.lower()
    patterns = {
        category: re.compile(r"\b(" + "|".join(re.escape(keyword) for keyword in keywords) + r")\b")
        for category, keywords in categories.items()
    }

    matched_categories = []
    matched_keywords = []
    for value in texts:
        row_categories = []
        row_keywords = []
        seen_keywords = set()
        for category, pattern in patterns.items():
            found = pattern.findall(value)
            if not found:
                continue
            row_categories.append(category)
            for keyword in found:
                if keyword not in seen_keywords:
                    seen_keywords.add(keyword)
                    row_keywords.append(keyword)
        matched_categories.append(";".join(row_categories) if row_categories else None)
        matched_keywords.append(";".join(row_keywords) if row_keywords else None)

    out["matched_categories_str"] = matched_categories
    out["matched_keywords_str"] = matched_keywords
    out["n_categories"] = [0 if value is None else len(str(value).split(";")) for value in matched_categories]
    return out


def merge_sentence_sentiment(sentence_df: pd.DataFrame, sentiment_csv: Path | None) -> pd.DataFrame:
    if sentiment_csv is None or not sentiment_csv.exists() or "sentence_uid" not in sentence_df.columns:
        return sentence_df
    sentiment_df = pd.read_csv(sentiment_csv)
    if "sentence_uid" not in sentiment_df.columns:
        return sentence_df
    sentiment_cols = [
        col
        for col in [
            "sentence_uid",
            "sentiment",
            "sentiment_norm",
            "sentiment_confidence",
            "sentiment_evidence_short",
            "sentiment_status",
            "sentiment_error",
        ]
        if col in sentiment_df.columns
    ]
    sentiment_df = sentiment_df[sentiment_cols].drop_duplicates(subset=["sentence_uid"])
    return sentence_df.merge(sentiment_df, on="sentence_uid", how="left", suffixes=("", "_sentiment"))


def sample_frame(df: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    if n <= 0 or df.empty:
        return df.iloc[0:0].copy()
    if len(df) <= n:
        return df.sample(frac=1, random_state=random_state).copy()
    return df.sample(n=n, random_state=random_state).copy()


def has_valid_location(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str).str.strip()
    return ~values.str.lower().isin({"", "none", "nan", "null"})


def prepare_sentence_items(
    df: pd.DataFrame,
    province_filter_regex: str,
    exclude_neutral: bool,
    sample_size: int,
    random_state: int,
    keywords_csv: Path | None,
    stage: str,
) -> pd.DataFrame:
    uid_col = "sentence_uid" if "sentence_uid" in df.columns else ("paragraph_uid" if "paragraph_uid" in df.columns else "uid")
    text_col = "sentence_text" if "sentence_text" in df.columns else "paragraph_text"
    required_cols = {uid_col, text_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in sentence input CSV: {sorted(missing)}")

    out = add_keyword_predictions(df.copy(), keywords_csv)
    sentiment_source_col = "sentiment_norm" if "sentiment_norm" in out.columns else ("sentiment" if "sentiment" in out.columns else None)
    if sentiment_source_col:
        out["_sent_norm"] = normalize_sentiment(out[sentiment_source_col])

    if province_filter_regex and "province_name" in out.columns:
        out = out[out["province_name"].astype(str).str.contains(province_filter_regex, case=False, na=False)].copy()

    if exclude_neutral and sentiment_source_col:
        out = out[out["_sent_norm"].isin(["positive", "negative"])].copy()

    out = out.drop_duplicates(subset=[uid_col]).copy()
    out = sample_frame(out, sample_size, random_state).reset_index(drop=True)
    out["item_type"] = "sentence"
    out["evaluation_stage"] = stage
    out["evaluation_uid"] = stage + ":sentence:" + out[uid_col].astype(str)
    out["evaluation_text"] = out[text_col]
    out["sentence_uid"] = out[uid_col]
    out["sentence_text"] = out[text_col]
    out["predicted_sentiment"] = out[sentiment_source_col] if sentiment_source_col else None
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
    return out.drop(columns=["_sent_norm"], errors="ignore")


def prepare_paragraph_items(
    df: pd.DataFrame,
    province_filter_regex: str,
    sample_size: int,
    random_state: int,
    stage: str,
    require_location: bool = False,
) -> pd.DataFrame:
    uid_col = "paragraph_uid" if "paragraph_uid" in df.columns else ("uid" if "uid" in df.columns else None)
    text_col = "paragraph_text" if "paragraph_text" in df.columns else None
    if uid_col is None or text_col is None:
        raise ValueError("Paragraph input CSV must include paragraph_uid or uid, plus paragraph_text.")

    out = df.copy()
    if require_location:
        if "llm_location" not in out.columns:
            return out.iloc[0:0].copy()
        out = out[has_valid_location(out["llm_location"])].copy()

    if province_filter_regex and "province_name" in out.columns:
        out = out[out["province_name"].astype(str).str.contains(province_filter_regex, case=False, na=False)].copy()

    out = out.drop_duplicates(subset=[uid_col]).copy()
    out = sample_frame(out, sample_size, random_state + 1).reset_index(drop=True)
    out["item_type"] = "paragraph"
    out["evaluation_stage"] = stage
    out["evaluation_uid"] = stage + ":paragraph:" + out[uid_col].astype(str)
    out["evaluation_text"] = out[text_col]
    out["paragraph_uid"] = out[uid_col]
    out["paragraph_text"] = out[text_col]
    out["sentence_uid"] = None
    out["sentence_text"] = None
    out["predicted_sentiment"] = None
    out["matched_categories_str"] = None
    out["matched_keywords_str"] = None
    if "llm_location" not in out.columns:
        out["llm_location"] = None
    if "geo_name_matched" not in out.columns:
        out["geo_name_matched"] = None
    return out


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    os.chdir(project_dir)

    input_csv = Path(args.input_csv)
    paragraph_input_csv = Path(args.paragraph_input_csv) if args.paragraph_input_csv else None
    paragraph_geothermal_csv = Path(args.paragraph_geothermal_csv) if args.paragraph_geothermal_csv else paragraph_input_csv
    paragraph_location_csv = Path(args.paragraph_location_csv) if args.paragraph_location_csv else None
    sentiment_csv = Path(args.sentiment_csv) if args.sentiment_csv else None
    keywords_csv = Path(args.keywords_csv) if args.keywords_csv else None
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    paragraph_location_sample_size = (
        args.paragraph_location_sample_size
        if args.paragraph_location_sample_size is not None
        else args.paragraph_sample_size
    )

    frame_items = prepare_sentence_items(
        pd.read_csv(input_csv),
        province_filter_regex=args.province_filter_regex,
        exclude_neutral=parse_bool(args.exclude_neutral),
        sample_size=args.sentence_sample_size,
        random_state=args.random_state,
        keywords_csv=keywords_csv,
        stage="sentence_frame",
    )

    frames = [frame_items]
    if sentiment_csv is not None:
        sentiment_items = prepare_sentence_items(
            pd.read_csv(sentiment_csv),
            province_filter_regex=args.province_filter_regex,
            exclude_neutral=parse_bool(args.exclude_neutral),
            sample_size=args.sentence_sample_size,
            random_state=args.random_state + 10,
            keywords_csv=keywords_csv,
            stage="sentence_sentiment",
        )
        frames.append(sentiment_items)

    if paragraph_geothermal_csv is not None:
        paragraph_geothermal_items = prepare_paragraph_items(
            pd.read_csv(paragraph_geothermal_csv),
            province_filter_regex=args.province_filter_regex,
            sample_size=args.paragraph_sample_size,
            random_state=args.random_state,
            stage="paragraph",
        )
        frames.append(paragraph_geothermal_items)

    if paragraph_location_csv is not None:
        paragraph_location_items = prepare_paragraph_items(
            pd.read_csv(paragraph_location_csv),
            province_filter_regex=args.province_filter_regex,
            sample_size=paragraph_location_sample_size,
            random_state=args.random_state + 20,
            stage="paragraph_location",
            require_location=True,
        )
        frames.append(paragraph_location_items)

    out = pd.concat(frames, ignore_index=True, sort=False)
    out["annotation_id"] = range(1, len(out) + 1)

    out.to_csv(output_csv, index=False)
    counts = out["evaluation_stage"].value_counts().to_dict()
    print(f"Wrote: {output_csv} (rows={len(out)}, counts={counts})")


if __name__ == "__main__":
    main()
