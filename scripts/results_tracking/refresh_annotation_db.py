#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sqlite3
import zlib
from pathlib import Path

import pandas as pd


LEGACY_STAGE_MAP = {
    "geothermal_relevance": "paragraph",
    "location_extraction": "paragraph_location",
    "paragraph_location_extraction": "paragraph_location",
    "frame_identification": "sentence_frame",
    "frame_matching": "sentence_frame",
    "sentiment_classification": "sentence_sentiment",
}
TASK_BUCKET_MOD = 100


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Refresh selected annotation task queues in the SQLite app database from an annotation CSV."
    )
    ap.add_argument("--annotation-csv", type=Path, default=Path("annotation/dutch/sentences_for_annotation.csv"))
    ap.add_argument("--db", type=Path, default=Path("annotation/annotations.db"))
    ap.add_argument("--language", type=str, default="dutch")
    ap.add_argument("--stages", nargs="+", default=["sentence_frame"])
    ap.add_argument(
        "--delete-annotated",
        action="store_true",
        help="Also delete existing annotations for the selected stages. By default annotated tasks are preserved.",
    )
    return ap.parse_args()


def canonical_evaluation_stage(value: object) -> str:
    stage = str(value or "").strip().lower()
    return LEGACY_STAGE_MAP.get(stage, stage)


def stable_bucket(value: object) -> int:
    return zlib.crc32(str(value).encode("utf-8")) % TASK_BUCKET_MOD


def parse_listish(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    return [item.strip() for item in text.split(";") if item.strip()]


def clean_value(value: object):
    return None if pd.isna(value) else value


def dataframe_for_stage(df: pd.DataFrame, language: str, stage: str) -> pd.DataFrame:
    out = df.copy()
    if "language" in out.columns:
        out = out[out["language"].fillna("").astype(str).str.strip().eq(language)].copy()
    else:
        out.insert(0, "language", language)
    if "evaluation_stage" not in out.columns:
        out["evaluation_stage"] = "paragraph"
    out["evaluation_stage"] = out["evaluation_stage"].map(canonical_evaluation_stage)
    return out[out["evaluation_stage"].eq(stage)].copy()


def row_to_task_values(row: pd.Series, columns: list[str], language: str) -> dict:
    uid_col = (
        "evaluation_uid"
        if "evaluation_uid" in columns
        else ("sentence_uid" if "sentence_uid" in columns else ("paragraph_uid" if "paragraph_uid" in columns else "uid"))
    )
    text_col = (
        "evaluation_text"
        if "evaluation_text" in columns
        else ("sentence_text" if "sentence_text" in columns else "paragraph_text")
    )
    task_uid = str(row[uid_col])
    item_type = str(row.get("item_type") or ("sentence" if "sentence_text" in columns else "paragraph")).strip().lower()
    predicted_frames = parse_listish(row.get("matched_categories_str")) if item_type == "sentence" else []
    sentiment_value = row.get("predicted_sentiment") or row.get("sentiment_norm") or row.get("sentiment")

    meta = {
        column: clean_value(row[column])
        for column in columns
        if column not in {uid_col, text_col, "sentiment"}
    }
    meta["task_uid_col"] = uid_col
    meta["task_text_col"] = text_col
    meta["predicted_frames"] = predicted_frames
    meta["predicted_location"] = row.get("llm_location")
    meta["matched_location"] = row.get("geo_name_matched")
    meta["paragraph_text"] = row.get("paragraph_text")
    meta["sentence_text"] = row.get("sentence_text")
    meta["item_type"] = item_type
    meta["evaluation_stage"] = canonical_evaluation_stage(meta.get("evaluation_stage"))
    meta["language"] = str(row.get("language") or language).strip()

    return {
        "paragraph_uid": task_uid,
        "paragraph_text": str(row[text_col]),
        "aspect_pred": ";".join(predicted_frames) if predicted_frames else None,
        "sentiment_pred": None if item_type != "sentence" or pd.isna(sentiment_value) else str(sentiment_value),
        "language": meta["language"],
        "evaluation_stage": meta["evaluation_stage"],
        "split_bucket": int(stable_bucket(task_uid)),
        "meta_json": json.dumps(meta, ensure_ascii=False),
    }


def refresh_stage(conn: sqlite3.Connection, df: pd.DataFrame, language: str, stage: str, delete_annotated: bool) -> dict:
    stage_df = dataframe_for_stage(df, language, stage)
    task_uids = set(stage_df["evaluation_uid"].astype(str)) if "evaluation_uid" in stage_df.columns else set()

    before = conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE language = ? AND evaluation_stage = ?",
        (language, stage),
    ).fetchone()[0]

    if delete_annotated:
        conn.execute(
            """
            DELETE FROM annotations
            WHERE paragraph_uid IN (
                SELECT paragraph_uid
                FROM tasks
                WHERE language = ? AND evaluation_stage = ?
            )
            """,
            (language, stage),
        )
        conn.execute(
            "DELETE FROM tasks WHERE language = ? AND evaluation_stage = ?",
            (language, stage),
        )
    else:
        conn.execute(
            """
            DELETE FROM tasks
            WHERE language = ?
              AND evaluation_stage = ?
              AND paragraph_uid NOT IN (SELECT paragraph_uid FROM annotations)
            """,
            (language, stage),
        )

    inserted_or_updated = 0
    for _, row in stage_df.iterrows():
        values = row_to_task_values(row, stage_df.columns.tolist(), language)
        conn.execute(
            """
            INSERT INTO tasks(
                paragraph_uid, paragraph_text, aspect_pred, sentiment_pred,
                language, evaluation_stage, split_bucket, meta_json
            )
            VALUES(
                :paragraph_uid, :paragraph_text, :aspect_pred, :sentiment_pred,
                :language, :evaluation_stage, :split_bucket, :meta_json
            )
            ON CONFLICT(paragraph_uid) DO UPDATE SET
                paragraph_text = excluded.paragraph_text,
                aspect_pred = excluded.aspect_pred,
                sentiment_pred = excluded.sentiment_pred,
                language = excluded.language,
                evaluation_stage = excluded.evaluation_stage,
                split_bucket = excluded.split_bucket,
                meta_json = excluded.meta_json
            """,
            values,
        )
        inserted_or_updated += 1

    after = conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE language = ? AND evaluation_stage = ?",
        (language, stage),
    ).fetchone()[0]
    annotated = conn.execute(
        """
        SELECT COUNT(*)
        FROM annotations a
        JOIN tasks t ON t.paragraph_uid = a.paragraph_uid
        WHERE t.language = ? AND t.evaluation_stage = ?
        """,
        (language, stage),
    ).fetchone()[0]

    return {
        "stage": stage,
        "csv_rows": len(stage_df),
        "csv_unique_uids": len(task_uids),
        "before": int(before),
        "inserted_or_updated": inserted_or_updated,
        "after": int(after),
        "annotated_preserved": int(annotated),
    }


def main() -> None:
    args = parse_args()
    if not args.annotation_csv.exists():
        raise FileNotFoundError(f"Annotation CSV not found: {args.annotation_csv}")
    if not args.db.exists():
        raise FileNotFoundError(f"Annotation database not found: {args.db}")

    df = pd.read_csv(args.annotation_csv)
    stages = [canonical_evaluation_stage(stage) for stage in args.stages]

    with sqlite3.connect(args.db) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        summaries = [
            refresh_stage(conn, df, args.language, stage, args.delete_annotated)
            for stage in stages
        ]

    for summary in summaries:
        print(
            " | ".join(
                f"{key}={value}"
                for key, value in summary.items()
            )
        )


if __name__ == "__main__":
    main()
