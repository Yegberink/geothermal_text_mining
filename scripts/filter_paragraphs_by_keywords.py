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
    ap.add_argument("--input-csv", type=str, default="output/text/newspapers_cleaned_paragraphs.csv")
    ap.add_argument("--keywords-csv", type=str, default="cache/keywords_topics_effective.csv")
    ap.add_argument("--output-csv", type=str, default="output/text/newspapers_keyword_filtered_paragraphs.csv")
    ap.add_argument("--text-col", type=str, default="paragraph_text")
    return ap.parse_args()


def load_keyword_pattern(path: Path) -> re.Pattern[str]:
    keywords_df = pd.read_csv(path)
    keywords_df = keywords_df.loc[:, ~keywords_df.columns.astype(str).str.match(r"^Unnamed")]

    keywords = (
        keywords_df.stack()
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .tolist()
    )
    keywords = sorted({keyword for keyword in keywords if keyword}, key=len, reverse=True)
    if not keywords:
        return re.compile(r"a\A")

    return re.compile(r"\b(" + "|".join(re.escape(keyword) for keyword in keywords) + r")\b")


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    os.chdir(project_dir)

    input_csv = Path(args.input_csv)
    keywords_csv = Path(args.keywords_csv)
    output_csv = Path(args.output_csv)

    paragraphs = pd.read_csv(input_csv)
    if args.text_col not in paragraphs.columns:
        raise ValueError(f"Expected text column {args.text_col!r} in {input_csv}")

    pattern = load_keyword_pattern(keywords_csv)
    text = paragraphs[args.text_col].fillna("").astype(str).str.lower()
    keep = text.map(lambda value: bool(pattern.search(value)))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out = paragraphs.loc[keep].copy()
    out.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"Wrote keyword-filtered paragraphs: {output_csv}")
    print(f"Paragraphs kept: {len(out)} / {len(paragraphs)}")
    print(f"[workflow_table] paragraphs_before_keyword_filter: {len(paragraphs)}")
    print(f"[workflow_table] paragraphs_after_keyword_filter: {len(out)}")


if __name__ == "__main__":
    main()
