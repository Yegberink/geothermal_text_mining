#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd

from language_resources import load_keyword_csv

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=str, default=str(DEFAULT_PROJECT_DIR))
    ap.add_argument("--sentences-csv", type=str, default="output/text/sentence_locations_ollama.csv")
    ap.add_argument("--keywords-csv", type=str, default="cache/keywords_topics_effective.csv")
    ap.add_argument("--output-csv", type=str, default="annotation/frame_keyword_review_candidates.csv")
    return ap.parse_args()


def load_vocab_patterns(path: Path) -> tuple[list[str], dict[str, re.Pattern[str]]]:
    vocab_df = load_keyword_csv(path)

    categories: list[str] = []
    patterns: dict[str, re.Pattern[str]] = {}
    for col in vocab_df.columns:
        category = str(col).strip()
        if not category:
            continue
        keywords = [value.strip().lower() for value in vocab_df[col].dropna().astype(str).tolist() if value.strip()]
        categories.append(category)
        if keywords:
            patterns[category] = re.compile(r"\b(" + "|".join(re.escape(keyword) for keyword in keywords) + r")\b")
    return categories, patterns


def score_sentences(df: pd.DataFrame, patterns: dict[str, re.Pattern[str]]) -> pd.DataFrame:
    out = df.copy()
    text_col = "sentence_text" if "sentence_text" in out.columns else "paragraph_text"
    text_series = out[text_col].fillna("").astype(str).str.lower()

    matched_categories: list[list[str]] = []
    matched_keywords: list[list[str]] = []
    for text in text_series:
        categories = []
        keywords = []
        seen = set()
        for category, pattern in patterns.items():
            found = pattern.findall(text)
            if found:
                categories.append(category)
                for keyword in found:
                    keyword = keyword.strip()
                    if keyword and keyword not in seen:
                        seen.add(keyword)
                        keywords.append(keyword)
        matched_categories.append(categories)
        matched_keywords.append(keywords)

    out["matched_categories_current"] = matched_categories
    out["matched_keywords_current"] = matched_keywords
    out["matched_categories_current_str"] = out["matched_categories_current"].apply(lambda values: ";".join(values) if values else None)
    out["matched_keywords_current_str"] = out["matched_keywords_current"].apply(lambda values: ";".join(values) if values else None)
    out["n_categories_current"] = out["matched_categories_current"].apply(len)
    return out


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    os.chdir(project_dir)

    sentences_csv = Path(args.sentences_csv)
    keywords_csv = Path(args.keywords_csv)
    output_csv = Path(args.output_csv)

    sentences = pd.read_csv(sentences_csv)
    if "sentence_uid" not in sentences.columns:
        raise ValueError(f"Expected 'sentence_uid' column in {sentences_csv}")
    if "sentence_text" not in sentences.columns and "paragraph_text" not in sentences.columns:
        raise ValueError(f"Expected 'sentence_text' or 'paragraph_text' in {sentences_csv}")

    _, patterns = load_vocab_patterns(keywords_csv)
    scored = score_sentences(sentences, patterns)
    candidates = scored[scored["n_categories_current"] == 0].copy()
    candidates = candidates.sort_values("sentence_uid").reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_csv, index=False)
    print(f"Wrote frame keyword review candidates: {output_csv} (rows={len(candidates)})")


if __name__ == "__main__":
    main()
