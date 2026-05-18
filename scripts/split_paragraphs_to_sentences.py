#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path

import pandas as pd

try:
    import spacy
except ImportError:
    spacy = None

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[1]

LANGUAGE_TO_SPACY = {
    "dutch": "nl",
    "nl": "nl",
    "italian": "it",
    "it": "it",
    "german": "de",
    "de": "de",
    "english": "en",
    "en": "en",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=str, default=str(DEFAULT_PROJECT_DIR))
    ap.add_argument("--input-csv", type=str, default="output/text/paragraph_locations_ollama.csv")
    ap.add_argument("--output-csv", type=str, default="output/text/sentence_locations_ollama.csv")
    ap.add_argument("--text-col", type=str, default="paragraph_text")
    ap.add_argument("--uid-col", type=str, default="uid")
    ap.add_argument("--language", type=str, default="nl")
    return ap.parse_args()


def make_sentence_uid(paragraph_uid: str, sentence_id: int, sentence_text: str) -> str:
    base = "||".join([paragraph_uid, str(sentence_id), sentence_text.strip()])
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def build_segmenter(language: str):
    if spacy is None:
        return None
    lang = LANGUAGE_TO_SPACY.get(str(language or "").strip().lower(), "xx")
    nlp = spacy.blank(lang)
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def split_sentences(nlp, text: str) -> list[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return []

    if nlp is None:
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Ý])", cleaned)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        return sentences or [cleaned]

    doc = nlp(cleaned)
    sentences = [str(sent).strip() for sent in doc.sents if str(sent).strip()]
    if not sentences:
        return [cleaned]
    return sentences


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    os.chdir(project_dir)

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    if args.text_col not in df.columns:
        raise ValueError(f"Expected text column '{args.text_col}' in input CSV.")
    if args.uid_col not in df.columns:
        raise ValueError(f"Expected uid column '{args.uid_col}' in input CSV.")

    nlp = build_segmenter(args.language)
    rows: list[dict] = []

    for _, row in df.iterrows():
        paragraph_text = str(row.get(args.text_col, "") or "").strip()
        paragraph_uid = str(row.get(args.uid_col, "") or "").strip()
        sentences = split_sentences(nlp, paragraph_text)

        for sentence_id, sentence_text in enumerate(sentences, start=1):
            out_row = row.to_dict()
            out_row["paragraph_uid"] = paragraph_uid
            out_row["sentence_id"] = sentence_id
            out_row["sentence_count_in_paragraph"] = len(sentences)
            out_row["sentence_text"] = sentence_text
            out_row["sentence_word_count"] = len(sentence_text.split())
            out_row["sentence_char_count"] = len(sentence_text)
            out_row["sentence_uid"] = make_sentence_uid(paragraph_uid, sentence_id, sentence_text)
            rows.append(out_row)

    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Wrote: {output_csv} (rows={len(out)})")
    print(f"[workflow_table] paragraphs_split_to_sentences: {len(df)}")
    print(f"[workflow_table] sentences_after_split: {len(out)}")


if __name__ == "__main__":
    main()
