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
    ap.add_argument("--base-csv", type=str, default="vocab/keywords_topics.csv")
    ap.add_argument("--review-csv", type=str, default="annotation/frame_keyword_reviews.csv")
    ap.add_argument("--output-csv", type=str, default="cache/keywords_topics_effective.csv")
    ap.add_argument("--audit-csv", type=str, default="output/text/keyword_framework_updates.csv")
    return ap.parse_args()


def parse_bool(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_semicolon_values(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def read_csv_with_encoding_fallback(path: Path, **kwargs: object) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", **kwargs)


def load_keyword_table(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    df = read_csv_with_encoding_fallback(path)
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]

    category_order: list[str] = []
    category_keywords: dict[str, list[str]] = {}
    for col in df.columns:
        category = str(col).strip()
        if not category:
            continue
        values = [value.strip() for value in df[col].dropna().astype(str).tolist() if value.strip()]
        category_order.append(category)
        category_keywords[category] = values
    return category_order, category_keywords


def write_keyword_table(category_order: list[str], category_keywords: dict[str, list[str]], out_path: Path) -> None:
    max_len = max((len(category_keywords.get(category, [])) for category in category_order), default=0)
    data: dict[str, list[str | None]] = {}
    for category in category_order:
        values = list(category_keywords.get(category, []))
        padded = values + [None] * (max_len - len(values))
        data[category] = padded
    out_df = pd.DataFrame(data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    os.chdir(project_dir)

    base_csv = Path(args.base_csv)
    review_csv = Path(args.review_csv)
    output_csv = Path(args.output_csv)
    audit_csv = Path(args.audit_csv)

    category_order, category_keywords = load_keyword_table(base_csv)
    normalized_lookup = {
        category: {keyword.strip().lower() for keyword in keywords}
        for category, keywords in category_keywords.items()
    }

    audit_rows: list[dict[str, object]] = []

    if review_csv.exists():
        reviews = read_csv_with_encoding_fallback(review_csv)
        for row in reviews.to_dict(orient="records"):
            if not parse_bool(row.get("include_in_vocab")):
                continue

            categories = parse_semicolon_values(row.get("matched_categories_true"))
            keywords = parse_semicolon_values(row.get("keywords_to_add"))
            if not categories:
                continue

            for category in categories:
                if category not in category_keywords:
                    category_order.append(category)
                    category_keywords[category] = []
                    normalized_lookup[category] = set()
                    audit_rows.append(
                        {
                            "sentence_uid": row.get("sentence_uid"),
                            "category": category,
                            "keyword": None,
                            "status": "created_category",
                            "notes": row.get("notes"),
                        }
                    )

                for keyword in keywords:
                    keyword_norm = keyword.lower()
                    if keyword_norm in normalized_lookup[category]:
                        status = "already_present"
                    else:
                        category_keywords[category].append(keyword)
                        normalized_lookup[category].add(keyword_norm)
                        status = "added_keyword"
                    audit_rows.append(
                        {
                            "sentence_uid": row.get("sentence_uid"),
                            "category": category,
                            "keyword": keyword,
                            "status": status,
                            "notes": row.get("notes"),
                        }
                    )

    write_keyword_table(category_order, category_keywords, output_csv)

    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows, columns=["sentence_uid", "category", "keyword", "status", "notes"]).to_csv(
        audit_csv,
        index=False,
    )

    print(f"Wrote effective keyword framework: {output_csv}")
    print(f"Wrote keyword update audit: {audit_csv}")


if __name__ == "__main__":
    main()
