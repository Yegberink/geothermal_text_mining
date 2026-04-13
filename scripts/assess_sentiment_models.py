#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("annotation/sentiment_model_assessment")
DEFAULT_INPUT_CANDIDATES = [
    Path("output/dutch/text/sentence_sentiment_llm.csv"),
    Path("output/dutch/text/sentences_with_frames_long.csv"),
]
DEFAULT_MODELS = [
    {
        "model_id": "DTAI-KULeuven/robbert-v2-dutch-sentiment",
        "model_slug": "robbert_v2_dutch",
        "display_name": "RobBERT v2 Dutch",
        "language_scope": "dutch",
    },
    {
        "model_id": "DTAI-KULeuven/robbertje-merged-dutch-sentiment",
        "model_slug": "robbertje_dutch",
        "display_name": "RobBERTje Dutch",
        "language_scope": "dutch",
    },
    {
        "model_id": "nlptown/bert-base-multilingual-uncased-sentiment",
        "model_slug": "nlptown_multilingual",
        "display_name": "NLP Town Multilingual BERT",
        "language_scope": "multilingual",
    },
    {
        "model_id": "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        "model_slug": "lxyuan_distilbert_multilingual",
        "display_name": "LXYuan Multilingual DistilBERT",
        "language_scope": "multilingual",
    },
    {
        "model_id": "tabularisai/multilingual-sentiment-analysis",
        "model_slug": "tabularisai_multilingual",
        "display_name": "TabularisAI Multilingual",
        "language_scope": "multilingual",
    },
]
SENTIMENTS = ["negative", "neutral", "positive"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Sample Dutch sentences, assign them across annotators with overlap, and run sentiment models."
    )
    ap.add_argument("--project-dir", type=str, default=str(DEFAULT_PROJECT_DIR))
    ap.add_argument("--input-csv", type=str, default="")
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--annotations-per-annotator", type=int, default=150)
    ap.add_argument("--overlap-sentences", type=int, default=30)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    return ap.parse_args()


def resolve_input_csv(explicit_path: str) -> Path:
    if explicit_path:
        candidate = Path(explicit_path)
        if not candidate.exists():
            raise FileNotFoundError(f"Input CSV not found: {candidate}")
        return candidate

    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate

    tried = ", ".join(str(path) for path in DEFAULT_INPUT_CANDIDATES)
    raise FileNotFoundError(f"Could not find an input CSV. Tried: {tried}")


def normalize_sentiment_label(label: str, model_id: str) -> str:
    value = str(label or "").strip().lower()
    collapsed = re.sub(r"[\s_\-/]+", " ", value)

    direct_map = {
        "negative": "negative",
        "neg": "negative",
        "label 0": "negative",
        "1 star": "negative",
        "1 stars": "negative",
        "2 star": "negative",
        "2 stars": "negative",
        "neutral": "neutral",
        "neu": "neutral",
        "label 1": "neutral",
        "3 star": "neutral",
        "3 stars": "neutral",
        "positive": "positive",
        "pos": "positive",
        "label 2": "positive",
        "4 star": "positive",
        "4 stars": "positive",
        "5 star": "positive",
        "5 stars": "positive",
    }
    if collapsed in direct_map:
        return direct_map[collapsed]

    star_match = re.search(r"([1-5])", collapsed)
    if star_match:
        stars = int(star_match.group(1))
        if stars <= 2:
            return "negative"
        if stars == 3:
            return "neutral"
        return "positive"

    if "neg" in collapsed:
        return "negative"
    if "neu" in collapsed:
        return "neutral"
    if "pos" in collapsed:
        return "positive"

    if collapsed.startswith("label "):
        if collapsed.endswith("0"):
            return "negative"
        if collapsed.endswith("1"):
            return "neutral"
        if collapsed.endswith("2"):
            return "positive"

    raise ValueError(f"Unsupported label '{label}' for model '{model_id}'.")


def make_sample_id(row: pd.Series) -> str:
    sentence_uid = str(row.get("sentence_uid", "")).strip()
    if sentence_uid:
        return sentence_uid

    base = "||".join(
        [
            str(row.get("source", "")),
            str(row.get("document_title", "")),
            str(row.get("publish_date", "")),
            str(row.get("paragraph_uid", "")),
            str(row.get("sentence_id", "")),
            str(row.get("sentence_text", "")),
        ]
    )
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def prepare_source_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.loc[:, ~df.columns.duplicated()].copy()
    if "sentence_text" not in out.columns:
        raise ValueError("Expected 'sentence_text' column in the input CSV.")

    out = out.dropna(subset=["sentence_text"]).copy()
    out["sentence_text"] = out["sentence_text"].astype(str).str.strip()
    out = out[out["sentence_text"] != ""].copy()

    if "paragraph_text" not in out.columns:
        out["paragraph_text"] = out["sentence_text"]
    if "document_title" not in out.columns and "title" in out.columns:
        out["document_title"] = out["title"]
    if "source" not in out.columns and "newspaper" in out.columns:
        out["source"] = out["newspaper"]
    if "publish_date" not in out.columns and "date_parsed" in out.columns:
        out["publish_date"] = out["date_parsed"]

    out["sample_id"] = out.apply(make_sample_id, axis=1)
    out = out.drop_duplicates(subset=["sample_id"]).copy()

    baseline_col = None
    for candidate in ["sentiment_norm", "sentiment"]:
        if candidate in out.columns:
            baseline_col = candidate
            break
    if baseline_col is not None:
        out["baseline_sentiment"] = out[baseline_col].astype(str).str.strip().str.lower()
        out.loc[~out["baseline_sentiment"].isin(SENTIMENTS), "baseline_sentiment"] = "unknown"
    else:
        out["baseline_sentiment"] = "unknown"

    return out


def sample_rows(df: pd.DataFrame, sample_size: int, random_seed: int) -> pd.DataFrame:
    if len(df) < sample_size:
        raise ValueError(f"Requested {sample_size} unique sentences, but only found {len(df)} usable rows.")

    usable = df.copy()
    stratified = usable[usable["baseline_sentiment"].isin(SENTIMENTS)].copy()
    groups = [group for _, group in stratified.groupby("baseline_sentiment", sort=False) if not group.empty]

    sampled_parts: list[pd.DataFrame] = []
    if groups:
        per_group = max(1, sample_size // len(groups))
        for offset, group in enumerate(groups):
            take = min(len(group), per_group)
            sampled_parts.append(group.sample(n=take, random_state=random_seed + offset))

    sampled = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else usable.head(0).copy()
    remaining_n = sample_size - len(sampled)
    if remaining_n > 0:
        already = set(sampled["sample_id"].tolist())
        remainder = usable[~usable["sample_id"].isin(already)].copy()
        sampled = pd.concat(
            [sampled, remainder.sample(n=remaining_n, random_state=random_seed + 999)],
            ignore_index=True,
        )

    sampled = sampled.sample(frac=1.0, random_state=random_seed + 2024).reset_index(drop=True)
    return sampled


def assign_annotators(
    sampled: pd.DataFrame,
    annotations_per_annotator: int,
    overlap_sentences: int,
    random_seed: int,
) -> pd.DataFrame:
    if overlap_sentences > annotations_per_annotator:
        raise ValueError("overlap_sentences cannot exceed annotations_per_annotator.")

    exclusive_per_annotator = annotations_per_annotator - overlap_sentences
    needed_unique = overlap_sentences + (2 * exclusive_per_annotator)
    if len(sampled) != needed_unique:
        raise ValueError(f"Expected {needed_unique} sampled rows, found {len(sampled)}.")

    out = sampled.copy().reset_index(drop=True)
    out["annotation_id"] = range(1, len(out) + 1)
    out["assignment_group"] = None
    out.loc[: overlap_sentences - 1, "assignment_group"] = "overlap"
    out.loc[overlap_sentences : overlap_sentences + exclusive_per_annotator - 1, "assignment_group"] = "dekker_only"
    out.loc[overlap_sentences + exclusive_per_annotator :, "assignment_group"] = "egberink_only"

    out["assigned_to_dekker"] = out["assignment_group"].isin(["overlap", "dekker_only"])
    out["assigned_to_egberink"] = out["assignment_group"].isin(["overlap", "egberink_only"])

    dekker_ids = out.loc[out["assigned_to_dekker"], "sample_id"].sample(
        frac=1.0, random_state=random_seed + 17
    ).tolist()
    egberink_ids = out.loc[out["assigned_to_egberink"], "sample_id"].sample(
        frac=1.0, random_state=random_seed + 29
    ).tolist()

    dekker_order = {sample_id: order for order, sample_id in enumerate(dekker_ids, start=1)}
    egberink_order = {sample_id: order for order, sample_id in enumerate(egberink_ids, start=1)}
    out["annotation_order_dekker"] = out["sample_id"].map(dekker_order)
    out["annotation_order_egberink"] = out["sample_id"].map(egberink_order)
    return out


def load_classifier(model_id: str):
    try:
        return pipeline(task="sentiment-analysis", model=model_id, tokenizer=model_id)
    except Exception as exc:
        message = str(exc)
        if "endswith" in message or "convert_slow_tokenizer" in message:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
            return pipeline(task="sentiment-analysis", model=model_id, tokenizer=tokenizer)
        offline_markers = ["Failed to resolve", "MaxRetryError", "NameResolutionError", "ConnectionError"]
        if any(marker in message for marker in offline_markers):
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_id, local_files_only=True)
            return pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)
        raise


def classify_texts(
    classifier: Any,
    texts: list[str],
    model_id: str,
    batch_size: int,
    max_length: int,
) -> list[dict[str, Any]]:
    raw_predictions = classifier(
        texts,
        batch_size=batch_size,
        truncation=True,
        max_length=max_length,
    )
    rows: list[dict[str, Any]] = []
    for prediction in raw_predictions:
        raw_label = str(prediction.get("label", "") or "").strip()
        score = float(prediction.get("score", 0.0) or 0.0)
        rows.append(
            {
                "predicted_sentiment": normalize_sentiment_label(raw_label, model_id=model_id),
                "raw_label": raw_label,
                "confidence": max(0.0, min(1.0, score)),
            }
        )
    return rows


def build_predictions_long(
    sampled: pd.DataFrame,
    batch_size: int,
    max_length: int,
) -> pd.DataFrame:
    base_cols = [
        "annotation_id",
        "sample_id",
        "sentence_uid",
        "sentence_text",
        "paragraph_text",
        "document_title",
        "source",
        "publish_date",
        "matched_categories_str",
        "matched_keywords_str",
        "baseline_sentiment",
        "assignment_group",
        "assigned_to_dekker",
        "assigned_to_egberink",
        "annotation_order_dekker",
        "annotation_order_egberink",
    ]
    for column in base_cols:
        if column not in sampled.columns:
            sampled[column] = None

    texts = sampled["sentence_text"].astype(str).tolist()
    prediction_frames: list[pd.DataFrame] = []

    for model_cfg in DEFAULT_MODELS:
        print(f"Running {model_cfg['model_id']} ...")
        model_df = sampled[base_cols].copy()
        model_df["model_id"] = model_cfg["model_id"]
        model_df["model_slug"] = model_cfg["model_slug"]
        model_df["model_display_name"] = model_cfg["display_name"]
        model_df["language_scope"] = model_cfg["language_scope"]
        try:
            classifier = load_classifier(model_cfg["model_id"])
            predictions = classify_texts(
                classifier=classifier,
                texts=texts,
                model_id=model_cfg["model_id"],
                batch_size=batch_size,
                max_length=max_length,
            )
            model_df["predicted_sentiment"] = [row["predicted_sentiment"] for row in predictions]
            model_df["raw_label"] = [row["raw_label"] for row in predictions]
            model_df["confidence"] = [row["confidence"] for row in predictions]
            model_df["model_status"] = "ok"
            model_df["model_error"] = None
        except Exception as exc:
            print(f"Model failed: {model_cfg['model_id']} -> {exc!r}")
            model_df["predicted_sentiment"] = None
            model_df["raw_label"] = None
            model_df["confidence"] = None
            model_df["model_status"] = "error"
            model_df["model_error"] = repr(exc)
        prediction_frames.append(model_df)

    return pd.concat(prediction_frames, ignore_index=True)


def build_predictions_wide(sampled: pd.DataFrame, predictions_long: pd.DataFrame) -> pd.DataFrame:
    metadata_cols = [
        "annotation_id",
        "sample_id",
        "sentence_uid",
        "sentence_text",
        "paragraph_text",
        "document_title",
        "source",
        "publish_date",
        "matched_categories_str",
        "matched_keywords_str",
        "baseline_sentiment",
        "assignment_group",
        "assigned_to_dekker",
        "assigned_to_egberink",
        "annotation_order_dekker",
        "annotation_order_egberink",
    ]
    metadata_cols = [column for column in metadata_cols if column in sampled.columns]
    wide = sampled[metadata_cols].copy()

    for model_cfg in DEFAULT_MODELS:
        current = predictions_long[predictions_long["model_slug"] == model_cfg["model_slug"]].copy()
        current = current[
            [
                "sample_id",
                "predicted_sentiment",
                "raw_label",
                "confidence",
                "model_status",
                "model_error",
            ]
        ].rename(
            columns={
                "predicted_sentiment": f"{model_cfg['model_slug']}__sentiment",
                "raw_label": f"{model_cfg['model_slug']}__raw_label",
                "confidence": f"{model_cfg['model_slug']}__confidence",
                "model_status": f"{model_cfg['model_slug']}__status",
                "model_error": f"{model_cfg['model_slug']}__error",
            }
        )
        wide = wide.merge(current, on="sample_id", how="left")

    return wide


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    os.chdir(project_dir)

    if args.overlap_sentences < 0:
        raise ValueError("--overlap-sentences must be >= 0.")
    if args.annotations_per_annotator <= 0:
        raise ValueError("--annotations-per-annotator must be > 0.")

    unique_sample_size = (2 * args.annotations_per_annotator) - args.overlap_sentences
    input_csv = resolve_input_csv(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_df = prepare_source_df(pd.read_csv(input_csv))
    sampled = sample_rows(source_df, sample_size=unique_sample_size, random_seed=args.random_seed)
    sampled["sentence_uid"] = sampled.get("sentence_uid", sampled["sample_id"])
    sampled = assign_annotators(
        sampled=sampled,
        annotations_per_annotator=args.annotations_per_annotator,
        overlap_sentences=args.overlap_sentences,
        random_seed=args.random_seed,
    )

    annotation_path = output_dir / "sentences_for_annotation.csv"
    sampled.to_csv(annotation_path, index=False, encoding="utf-8")
    print(f"Wrote annotation set: {annotation_path} (unique sentences={len(sampled)})")
    print(
        "Assignments: "
        f"Dekker={int(sampled['assigned_to_dekker'].sum())}, "
        f"Egberink={int(sampled['assigned_to_egberink'].sum())}, "
        f"overlap={int((sampled['assignment_group'] == 'overlap').sum())}"
    )

    predictions_long = build_predictions_long(
        sampled=sampled,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    predictions_wide = build_predictions_wide(sampled=sampled, predictions_long=predictions_long)

    long_path = output_dir / "model_predictions_long.csv"
    wide_path = output_dir / "model_predictions_wide.csv"
    models_path = output_dir / "models.json"

    predictions_long.to_csv(long_path, index=False, encoding="utf-8")
    predictions_wide.to_csv(wide_path, index=False, encoding="utf-8")
    models_path.write_text(json.dumps(DEFAULT_MODELS, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote long predictions: {long_path}")
    print(f"Wrote wide predictions: {wide_path}")
    print(f"Wrote model metadata: {models_path}")


if __name__ == "__main__":
    main()
