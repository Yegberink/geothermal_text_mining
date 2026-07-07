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
    frames: dict[str, pd.DataFrame] = {}

    stage_columns = {
        "articles": ["body_hash"],
        "paragraphs": ["body_hash", "uid", "paragraph_sentence_count"],
        "keyword_paragraphs": [
            "body_hash",
            "uid",
            "paragraph_word_count",
            "paragraph_sentence_count",
        ],
        "geothermal_paragraphs": ["body_hash", "uid", "llm_is_geothermal", "llm_status"],
        "location_paragraphs": [
            "body_hash",
            "uid",
            "llm_status",
            "llm_location",
            "llm_location_source",
            "llm_paragraph_returned_none",
        ],
        "shape_geocoded_paragraphs": ["body_hash", "geom_point_wkt", "geom_poly_wkt"],
        "final_geocoded_paragraphs": [
            "body_hash",
            "geom_point_wkt",
            "geom_poly_wkt",
            "geo_source",
            "geo_match_type",
        ],
        "sentences": ["body_hash", "sentence_uid", "geom_point_wkt", "geom_poly_wkt"],
        "frames": ["body_hash", "sentence_uid"],
        "sentiment": [
            "body_hash",
            "sentence_uid",
            "sentiment",
            "sentiment_norm",
            "sentiment_status",
        ],
        "categories": ["body_hash", "sentence_uid"],
        "admin": ["body_hash"],
    }
    for key, filename in TEXT_STAGE_FILES.items():
        frames[key] = read_csv_selected(
            language_workflow_path(output_root, language, filename),
            stage_columns.get(key, []),
        )

    raw = read_many_csv_selected(raw_article_paths(cache_root, language), RAW_DOC_KEY_COLUMNS)
    raw_docs = document_count(raw, raw=True)
    table.set(
        "raw_article_blocks",
        1,
        "documents",
        "Article blocks extracted from source RTF exports before cleaning and deduplication.",
        language,
        len(raw) if not raw.empty else None,
        raw_docs,
    )

    articles = frames["articles"]
    table.set(
        "cleaned_articles",
        2,
        "documents",
        "Articles retained after body cleanup, minimum article length filtering, optional geo-hit filtering, and deduplication.",
        language,
        *count_subset(articles),
    )

    paragraphs = frames["paragraphs"]
    table.set(
        "paragraphs_after_split_filter",
        3,
        "paragraphs",
        "Paragraphs generated from cleaned articles after paragraph splitting and paragraph-size cleanup.",
        language,
        *count_subset(paragraphs),
    )
    if "paragraph_sentence_count" in paragraphs.columns:
        table.set(
            "sentences_estimated_after_paragraph_split",
            4,
            "sentences",
            "Sentence total estimated during paragraph splitting across all retained paragraphs.",
            language,
            numeric_sum(paragraphs["paragraph_sentence_count"]),
            document_count(paragraphs),
        )
    table.set(
        "documents_with_paragraphs",
        5,
        "documents",
        "Unique documents represented by at least one retained paragraph.",
        language,
        document_count(paragraphs),
        document_count(paragraphs),
    )

    keyword_paragraphs = frames["keyword_paragraphs"]
    table.set(
        "paragraphs_after_keyword_filter",
        6,
        "paragraphs",
        "Paragraphs containing at least one keyword from the effective frame keyword table.",
        language,
        *count_subset(keyword_paragraphs),
    )
    removed_keyword = anti_join(paragraphs, keyword_paragraphs, "uid")
    table.set(
        "paragraphs_removed_by_keyword_filter",
        7,
        "paragraphs",
        "Paragraphs removed because they did not match the effective keyword table.",
        language,
        *count_subset(removed_keyword),
    )
    if "paragraph_sentence_count" in keyword_paragraphs.columns:
        table.set(
            "sentences_estimated_after_keyword_filter",
            8,
            "sentences",
            "Sentence total estimated in keyword-matched paragraphs before geothermal classification.",
            language,
            numeric_sum(keyword_paragraphs["paragraph_sentence_count"]),
            document_count(keyword_paragraphs),
        )

    geothermal = frames["geothermal_paragraphs"]
    removed_length = anti_join(keyword_paragraphs, geothermal, "uid")
    table.set(
        "paragraphs_after_geothermal_length_filter",
        9,
        "paragraphs",
        "Keyword-matched paragraphs retained for geothermal LLM classification after the 20-499 word length filter.",
        language,
        *count_subset(geothermal),
    )
    table.set(
        "paragraphs_removed_by_geothermal_length_filter",
        10,
        "paragraphs",
        "Keyword-matched paragraphs removed by the geothermal classifier length filter.",
        language,
        *count_subset(removed_length),
    )

    table.set(
        "paragraphs_sent_to_geothermal_classifier",
        11,
        "paragraphs",
        "Paragraphs written by the geothermal LLM classification stage.",
        language,
        *count_subset(geothermal),
    )
    if "llm_status" in geothermal.columns:
        for label, step in [("ok", 12), ("empty", 13), ("error", 14)]:
            mask = geothermal["llm_status"].fillna("").astype(str).str.lower().eq(label)
            table.set(
                f"paragraphs_geothermal_status_{label}",
                step,
                "paragraphs",
                f"Geothermal classification rows with llm_status={label}.",
                language,
                *count_subset(geothermal, mask),
            )
    if "llm_is_geothermal" in geothermal.columns:
        labels = [("yes", 15), ("maybe", 16), ("no", 17)]
        for label, step in labels:
            mask = geothermal["llm_is_geothermal"].fillna("").astype(str).str.lower().eq(label)
            table.set(
                f"paragraphs_classified_geothermal_{label}",
                step,
                "paragraphs",
                f"Paragraphs classified as geothermal={label.upper()} by the geothermal LLM.",
                language,
                *count_subset(geothermal, mask),
            )
        keep_mask = geothermal["llm_is_geothermal"].fillna("").astype(str).str.upper().eq("YES")
        table.set(
            "paragraphs_removed_by_geothermal_classifier",
            18,
            "paragraphs",
            "Paragraphs not sent onward to location extraction because geothermal classification was NO or MAYBE.",
            language,
            *count_subset(geothermal, ~keep_mask),
        )

    locations = frames["location_paragraphs"]
    table.set(
        "paragraphs_sent_to_location_extraction",
        19,
        "paragraphs",
        "Geothermal YES paragraphs processed by location extraction.",
        language,
        *count_subset(locations),
    )
    if "llm_status" in locations.columns:
        for label, step in [("ok", 20), ("empty", 21), ("error", 22)]:
            mask = locations["llm_status"].fillna("").astype(str).str.lower().eq(label)
            table.set(
                f"paragraphs_location_status_{label}",
                step,
                "paragraphs",
                f"Location extraction rows with llm_status={label}.",
                language,
                *count_subset(locations, mask),
            )
    if "llm_location" in locations.columns:
        located = valid_location(locations["llm_location"])
        table.set(
            "paragraphs_with_extracted_location_after_fallback",
            23,
            "paragraphs",
            "Paragraphs with a usable location after paragraph-, document-, and country-fallback location handling.",
            language,
            *count_subset(locations, located),
        )
        table.set(
            "paragraphs_without_extracted_location_after_fallback",
            24,
            "paragraphs",
            "Paragraphs without a usable location after location fallback handling.",
            language,
            *count_subset(locations, ~located),
        )
    if "llm_paragraph_returned_none" in locations.columns:
        none_mask = truthy(locations["llm_paragraph_returned_none"])
        table.set(
            "paragraph_location_llm_none_responses",
            25,
            "paragraphs",
            "Paragraph-level location calls that returned no usable location before fallback handling.",
            language,
            *count_subset(locations, none_mask),
        )
    if "llm_location_source" in locations.columns:
        source_steps = [
            ("paragraph", 26),
            ("document_single_location", 27),
            ("document_llm", 28),
            ("country_all_document_locations", 29),
            ("country_fallback", 30),
            ("multi_country_fallback", 31),
        ]
        source_values = locations["llm_location_source"].fillna("").astype(str)
        for source, step in source_steps:
            source_mask = source_values.eq(source)
            table.set(
                f"paragraph_locations_from_{source}",
                step,
                "paragraphs",
                f"Paragraph locations whose final location source is {source}.",
                language,
                *count_subset(locations, source_mask),
            )

    shape_geo = frames["shape_geocoded_paragraphs"]
    table.set(
        "paragraphs_after_shapes_geocoding",
        32,
        "paragraphs",
        "Paragraph rows after offline shape-based geocoding.",
        language,
        *count_subset(shape_geo),
    )
    if not shape_geo.empty:
        shape_has_geo = has_geo(shape_geo)
        table.set(
            "paragraphs_geocoded_shapes",
            33,
            "paragraphs",
            "Paragraphs matched to geometry by offline shape lookup.",
            language,
            *count_subset(shape_geo, shape_has_geo),
        )
        table.set(
            "paragraphs_not_geocoded_shapes",
            34,
            "paragraphs",
            "Paragraphs not matched to geometry by offline shape lookup.",
            language,
            *count_subset(shape_geo, ~shape_has_geo),
        )

    final_geo = frames["final_geocoded_paragraphs"]
    table.set(
        "paragraphs_after_final_geocoding",
        35,
        "paragraphs",
        "Paragraph rows after final geocoding, including manual overrides, GeoNames, and geocoder-cache matches.",
        language,
        *count_subset(final_geo),
    )
    if not final_geo.empty:
        final_has_geo = has_geo(final_geo)
        table.set(
            "paragraphs_geocoded_final",
            36,
            "paragraphs",
            "Paragraphs with point or polygon geometry after final geocoding.",
            language,
            *count_subset(final_geo, final_has_geo),
        )
        table.set(
            "paragraphs_not_geocoded_final",
            37,
            "paragraphs",
            "Paragraphs without point or polygon geometry after final geocoding.",
            language,
            *count_subset(final_geo, ~final_has_geo),
        )
        if "geo_match_type" in final_geo.columns:
            ignored = final_geo["geo_match_type"].fillna("").astype(str).str.lower().eq("ignored_location")
            table.set(
                "paragraphs_ignored_by_geocoding_override",
                38,
                "paragraphs",
                "Paragraphs explicitly ignored by a manual geocoding override.",
                language,
                *count_subset(final_geo, ignored),
            )
        if "geo_source" in final_geo.columns:
            source = final_geo["geo_source"].fillna("").astype(str).str.lower()
            source_rows = [
                ("manual_override", "paragraphs_geocoded_manual_override", 39),
                ("geonames", "paragraphs_geocoded_geonames", 40),
            ]
            for source_name, metric, step in source_rows:
                source_mask = source.eq(source_name)
                table.set(
                    metric,
                    step,
                    "paragraphs",
                    f"Final geocoded paragraphs whose geo_source is {source_name}.",
                    language,
                    *count_subset(final_geo, source_mask),
                )
            cache_mask = source.str.endswith("_cache")
            table.set(
                "paragraphs_geocoded_geocoder_cache",
                41,
                "paragraphs",
                "Final geocoded paragraphs matched from a geocoder cache.",
                language,
                *count_subset(final_geo, cache_mask),
            )

    unmatched = read_csv_selected(unmatched_path(output_root, language), ["row_count"])
    table.set(
        "geocoding_unmatched_unique_locations",
        42,
        "locations",
        "Unique unmatched location strings in the final geocoding unmatched report.",
        language,
        len(unmatched) if not unmatched.empty else 0,
        None,
    )
    if "row_count" in unmatched.columns:
        table.set(
            "geocoding_unmatched_paragraph_rows",
            43,
            "paragraphs",
            "Paragraph rows represented by the final geocoding unmatched report.",
            language,
            numeric_sum(unmatched["row_count"]),
            None,
        )

    sentences = frames["sentences"]
    table.set(
        "sentences_after_split",
        44,
        "sentences",
        "Sentences produced from final geocoded paragraphs.",
        language,
        *count_subset(sentences),
    )
    if not sentences.empty:
        sentence_has_geo = has_geo(sentences)
        table.set(
            "sentences_with_geo_after_split",
            45,
            "sentences",
            "Sentences with point or polygon geometry after sentence splitting.",
            language,
            *count_subset(sentences, sentence_has_geo),
        )
        table.set(
            "sentences_without_geo_after_split",
            46,
            "sentences",
            "Sentences without point or polygon geometry after sentence splitting.",
            language,
            *count_subset(sentences, ~sentence_has_geo),
        )

    frame_sentences = frames["frames"]
    table.set(
        "sentences_after_frame_keyword_filter",
        47,
        "sentences",
        "Sentences retained after frame keyword matching.",
        language,
        *count_subset(frame_sentences),
    )
    removed_frames = anti_join(sentences, frame_sentences, "sentence_uid")
    table.set(
        "sentences_removed_by_frame_keyword_filter",
        48,
        "sentences",
        "Sentences removed because no frame keyword matched.",
        language,
        *count_subset(removed_frames),
    )
    frame_ids = set(non_empty(frame_sentences["sentence_uid"])) if "sentence_uid" in frame_sentences.columns else set()
    if "sentence_uid" in sentences.columns and frame_ids:
        matched_sentence_inputs = sentences[
            sentences["sentence_uid"].fillna("").astype(str).isin(frame_ids)
        ].copy()
        duplicate_frame_inputs = matched_sentence_inputs[
            matched_sentence_inputs["sentence_uid"].fillna("").astype(str).duplicated(keep="first")
        ].copy()
    else:
        duplicate_frame_inputs = pd.DataFrame()
    table.set(
        "sentences_dropped_during_frame_deduplication",
        49,
        "sentences",
        "Frame-matched input sentence rows dropped when duplicate sentence_uid values were collapsed.",
        language,
        *count_subset(duplicate_frame_inputs),
    )

    sentiment_df = frames["sentiment"]
    table.set(
        "sentences_sent_to_sentiment",
        50,
        "sentences",
        "Frame-matched sentences processed by the sentiment LLM.",
        language,
        *count_subset(sentiment_df),
    )
    if "sentiment_status" in sentiment_df.columns:
        for label, step in [("ok", 51), ("empty", 52), ("error", 53)]:
            mask = sentiment_df["sentiment_status"].fillna("").astype(str).str.lower().eq(label)
            table.set(
                f"sentences_sentiment_status_{label}",
                step,
                "sentences",
                f"Sentiment rows with sentiment_status={label}.",
                language,
                *count_subset(sentiment_df, mask),
            )
    sentiment_col = "sentiment_norm" if "sentiment_norm" in sentiment_df.columns else "sentiment"
    if sentiment_col in sentiment_df.columns:
        sentiment_values = sentiment_df[sentiment_col].fillna("").astype(str).str.lower()
        for label, step in [("negative", 54), ("neutral", 55), ("positive", 56)]:
            mask = sentiment_values.eq(label)
            table.set(
                f"sentences_classified_sentiment_{label}",
                step,
                "sentences",
                f"Sentences classified as {label} sentiment.",
                language,
                *count_subset(sentiment_df, mask),
            )

    categories = frames["categories"]
    table.set(
        "sentences_after_category_filter",
        57,
        "sentences",
        "Sentences retained after final category keyword matching.",
        language,
        *count_subset(categories),
    )
    removed_categories = anti_join(sentiment_df, categories, "sentence_uid")
    table.set(
        "sentences_removed_by_category_filter",
        58,
        "sentences",
        "Sentences removed by the final category keyword matching stage.",
        language,
        *count_subset(removed_categories),
    )

    admin = frames["admin"]
    table.set(
        "sentences_after_admin_aggregation",
        59,
        "sentences",
        "Sentence rows after administrative-area aggregation.",
        language,
        *count_subset(admin),
    )


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
