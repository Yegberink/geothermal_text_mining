#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_CSV = "output/generic/text/data_tracking.csv"

TEXT_STAGE_FILES = {
    "articles": "articles_cleaned.csv",
    "paragraphs": "newspapers_cleaned_paragraphs.csv",
    "keyword_paragraphs": "newspapers_keyword_filtered_paragraphs.csv",
    "geothermal_paragraphs": "paragraph_geothermal_ollama.csv",
    "location_paragraphs": "paragraph_locations_ollama.csv",
    "shape_geocoded_paragraphs": "paragraph_shapes_geocoding.csv",
    "final_geocoded_paragraphs": "paragraphs_with_geo.csv",
    "sentences": "sentence_locations_ollama.csv",
    "frames": "sentences_with_frames_long.csv",
    "sentiment": "sentence_sentiment_llm.csv",
    "categories": "sentences_with_categories_long.csv",
    "admin": "sentences_with_categories_admin.csv",
}

DOC_KEY_COLUMNS = [
    "body_hash",
    "document_id",
    "article_id",
    "document_uid",
]
RAW_DOC_KEY_COLUMNS = [
    "source_path",
    "source_relative_path",
    "source_file",
    "title",
    "newspaper",
    "date",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=str, default=str(DEFAULT_PROJECT_DIR))
    ap.add_argument("--languages", nargs="+", required=True)
    ap.add_argument("--output-csv", type=str, default=DEFAULT_OUTPUT_CSV)
    ap.add_argument("--output-root", type=str, default="output")
    ap.add_argument("--cache-root", type=str, default="cache")
    return ap.parse_args()


def title_case_language(language: str) -> str:
    return str(language).replace("_", " ").title()


def ordered_languages(languages: list[str]) -> list[str]:
    preferred = ["dutch", "italian", "german"]
    by_lower = {language.lower(): language for language in languages}
    ordered = [by_lower[item] for item in preferred if item in by_lower]
    ordered.extend(language for language in languages if language.lower() not in preferred)
    return ordered


def non_empty(series: pd.Series) -> pd.Series:
    values = series.dropna().astype(str).str.strip()
    return values[~values.str.lower().isin(["", "nan", "none", "null", "<na>"])]


def truthy(series: pd.Series) -> pd.Series:
    return series.fillna(False).map(
        lambda value: str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
    )


def valid_location(series: pd.Series) -> pd.Series:
    return ~series.fillna("").astype(str).str.strip().str.lower().isin(
        ["", "none", "nan", "null", "<na>"]
    )


def has_geo(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for col in ["geom_point_wkt", "geom_poly_wkt"]:
        if col in df.columns:
            mask = mask | non_empty(df[col]).reindex(df.index, fill_value="").astype(str).ne("")
    return mask


def read_csv_selected(path: Path, wanted_columns: Iterable[str] = ()) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    header = pd.read_csv(path, nrows=0)
    available = list(header.columns)
    wanted = [col for col in wanted_columns if col in available]
    if not wanted and available:
        wanted = [available[0]]
    if not wanted:
        return pd.DataFrame()
    return pd.read_csv(path, usecols=wanted, dtype=str, low_memory=False)


def read_many_csv_selected(paths: Iterable[Path], wanted_columns: Iterable[str]) -> pd.DataFrame:
    frames = [read_csv_selected(path, wanted_columns) for path in paths]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def document_count(df: pd.DataFrame, raw: bool = False) -> int | None:
    if df.empty:
        return 0

    for col in DOC_KEY_COLUMNS:
        if col in df.columns:
            return int(non_empty(df[col]).nunique())

    key_cols = RAW_DOC_KEY_COLUMNS if raw else []
    present = [col for col in key_cols if col in df.columns]
    if present:
        keys = (
            df[present]
            .fillna("")
            .astype(str)
            .agg("||".join, axis=1)
            .str.strip()
        )
        keys = keys[keys.ne("")]
        return int(keys.nunique())

    return None


def numeric_sum(series: pd.Series) -> int:
    return int(pd.to_numeric(series, errors="coerce").fillna(0).sum())


def anti_join(before: pd.DataFrame, after: pd.DataFrame, uid_col: str) -> pd.DataFrame:
    if uid_col not in before.columns or uid_col not in after.columns:
        return pd.DataFrame()
    after_ids = set(non_empty(after[uid_col]))
    return before[~before[uid_col].fillna("").astype(str).isin(after_ids)].copy()


def count_subset(df: pd.DataFrame, mask: pd.Series | None = None) -> tuple[int, int | None]:
    subset = df if mask is None else df.loc[mask.reindex(df.index, fill_value=False)]
    return len(subset), document_count(subset)


def language_text_path(output_root: Path, language: str, filename: str) -> Path:
    return output_root / language / "text" / filename


def language_workflow_path(output_root: Path, language: str, filename: str) -> Path:
    return output_root / language / "workflow" / filename


def raw_article_paths(cache_root: Path, language: str) -> list[Path]:
    return sorted((cache_root / language / "preprocess_rtf_chunks" / "raw_articles").glob("*.csv"))


def unmatched_path(output_root: Path, language: str) -> Path:
    return language_text_path(output_root, language, "geocoding_unmatched.csv")


class TrackingTable:
    def __init__(self, languages: list[str]) -> None:
        self.languages = languages
        self.rows: OrderedDict[str, dict[str, object]] = OrderedDict()

    def set(
        self,
        metric: str,
        step: int,
        unit: str,
        description: str,
        language: str,
        count: int | None,
        unique_documents: int | None = None,
    ) -> None:
        row = self.rows.setdefault(
            metric,
            {
                "step": step,
                "metric": metric,
                "unit": unit,
                "description": description,
            },
        )
        row[f"{language}_count"] = "" if count is None else int(count)
        row[f"{language}_unique_documents"] = (
            "" if unique_documents is None else int(unique_documents)
        )

    def to_frame(self) -> pd.DataFrame:
        rows = list(self.rows.values())
        columns = ["step", "metric", "unit", "description"]
        for language in self.languages:
            columns.extend([f"{language}_count", f"{language}_unique_documents"])
        out = pd.DataFrame(rows)
        for col in columns:
            if col not in out.columns:
                out[col] = ""
        return out[columns].sort_values(["step", "metric"], kind="stable")


def add_rows_for_language(
    table: TrackingTable,
    language: str,
    output_root: Path,
    cache_root: Path,
) -> None:
    columns = [*DOC_KEY_COLUMNS, "uid", "sentence_uid", "paragraph_text",
               "llm_is_geothermal", "geom_point_wkt", "geom_poly_wkt",
               "sentiment", "sentiment_norm", "sentiment_status"]
    stages = ["articles", "paragraphs", "geothermal_paragraphs",
              "final_geocoded_paragraphs", "sentences", "frames", "sentiment"]
    frames = {
        key: read_csv_selected(language_workflow_path(output_root, language, TEXT_STAGE_FILES[key]), columns)
        for key in stages
    }
    raw = read_many_csv_selected(raw_article_paths(cache_root, language), RAW_DOC_KEY_COLUMNS)
    paragraphs = frames["paragraphs"]
    word_counts = paragraphs.get("paragraph_text", pd.Series(index=paragraphs.index, dtype=str)).fillna("").str.split().str.len()
    length_filtered = paragraphs.loc[word_counts.ge(20) & word_counts.lt(500)]
    geothermal = frames["geothermal_paragraphs"]
    yes = geothermal.loc[geothermal.get("llm_is_geothermal", pd.Series(index=geothermal.index, dtype=str)).fillna("").str.strip().str.upper().eq("YES")]
    geo = frames["final_geocoded_paragraphs"]
    geo = geo.loc[has_geo(geo)]
    sentiments = frames["sentiment"]
    sentiment_col = "sentiment_norm" if "sentiment_norm" in sentiments else "sentiment"
    valid = sentiments.get(sentiment_col, pd.Series(index=sentiments.index, dtype=str)).fillna("").str.strip().str.lower().isin(["negative", "neutral", "positive"])
    if "sentiment_status" in sentiments:
        valid &= sentiments["sentiment_status"].fillna("").str.lower().eq("ok")
    sentiments = sentiments.loc[valid]
    frame_sentences = frames["frames"]
    sentiment_ids = set(non_empty(sentiments.get("sentence_uid", pd.Series(dtype=str))))
    both = frame_sentences.loc[frame_sentences.get("sentence_uid", pd.Series(index=frame_sentences.index, dtype=str)).isin(sentiment_ids)]
    rows = [
        ("documents_extracted_from_files", "documents", raw),
        ("unique_documents", "documents", frames["articles"]),
        ("total_paragraphs", "paragraphs", paragraphs),
        ("paragraphs_after_length_filter", "paragraphs", length_filtered),
        ("paragraphs_about_geothermal", "paragraphs", yes),
        ("paragraphs_with_geo", "paragraphs", geo),
        ("total_sentences", "sentences", frames["sentences"]),
        ("sentences_with_sentiment", "sentences", sentiments),
        ("sentences_with_frames", "sentences", frame_sentences),
        ("sentences_with_frames_and_sentiment", "sentences", both),
    ]
    for step, (metric, unit, subset) in enumerate(rows, start=1):
        count = document_count(subset) if metric == "unique_documents" else len(subset)
        table.set(metric, step, unit, metric.replace("_", " ").capitalize(), language,
                  count, document_count(subset, raw=step == 1))


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    os.chdir(project_dir)

    languages = ordered_languages(
        [str(language).strip() for language in args.languages if str(language).strip()]
    )
    if not languages:
        raise ValueError("At least one language is required.")

    table = TrackingTable(languages)
    output_root = Path(args.output_root)
    cache_root = Path(args.cache_root)
    for language in languages:
        add_rows_for_language(table, language, output_root, cache_root)

    out = table.to_frame()
    out.insert(1, "label", out["metric"].str.replace("_", " ").str.title())
    for language in languages:
        count_col = f"{language}_count"
        doc_col = f"{language}_unique_documents"
        out.rename(
            columns={
                count_col: f"{title_case_language(language)} count",
                doc_col: f"{title_case_language(language)} unique documents",
            },
            inplace=True,
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Wrote data tracking table: {output_csv} (rows={len(out)})")


if __name__ == "__main__":
    main()
