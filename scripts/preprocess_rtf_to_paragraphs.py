#!/usr/bin/env python3
"""
Standalone newspaper RTF -> paragraph sentiment pipeline (simplified).

What this does:
1) Recursively reads all .rtf files under INPUT_RTF_DIR
2) Converts RTF -> plain text, splits into articles by "End of Document"
3) Extracts metadata (title/newspaper/date/etc.) + body
4) Optional cleaning + optional filtering (all controlled by constants below)
5) Splits body into paragraphs (fixes hard-wrapped newlines)
6) Runs Dutch sentiment (RobBERT) per paragraph (batched)
7) Writes output CSV: OUTPUT_PARAGRAPH_CSV
   (Optional) Writes cleaned-articles CSV: OUTPUT_ARTICLES_CSV

No CLI arguments. Edit the constants in the CONFIG section if needed.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import argparse

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm.auto import tqdm


# =============================
# CONFIG (edit these)
# =============================

# Project root is the parent of the scripts directory.
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Folder containing your .rtf files (recursive).
# Change this to where your RTFs actually live.
INPUT_RTF_DIR = PROJECT_DIR / "input_data"

# Optional mapping file: must contain columns: newspaper, region_name
NEWSPAPER_REGION_CSV = PROJECT_DIR / "data" / "newspaper_regions.csv"

# Outputs
OUTPUT_DIR = PROJECT_DIR / "output" / "text"
OUTPUT_PARAGRAPH_CSV = OUTPUT_DIR / "newspapers_cleaned_paragraphs.csv"
OUTPUT_ARTICLES_CSV = OUTPUT_DIR / "articles_cleaned.csv"
WRITE_ARTICLES_CSV = True

# Cleaning / filtering knobs (keep it simple)
MIN_WORDS_PER_ARTICLE = 100
FILTER_BY_GEO_HITS = False
MIN_GEO_HITS = 3
GEOTHERMAL_PATTERNS = [r"aardwarmte\w*", r"geotherm\w*"]  # regex
KEEP_ONLY_DUTCH = False  # requires langdetect; if missing, will skip

# Paragraph splitting knobs
MIN_WORDS_PER_PARAGRAPH = 20

# Sentiment model
MODEL_NAME = "DTAI-KULeuven/robbert-v2-dutch-sentiment"
BATCH_SIZE = 16
DEVICE = "auto"  # "auto"|"cpu"|"cuda"


# =============================
# RTF -> plain text
# =============================

def rtf_to_text(rtf_content: str) -> str:
    """Convert RTF to plain text using striprtf if available, otherwise a basic fallback."""
    try:
        from striprtf.striprtf import rtf_to_text as _rtf_to_text  # type: ignore
        return _rtf_to_text(rtf_content)
    except Exception:
        # Very rough fallback; good enough to keep pipeline running.
        txt = re.sub(r"{\\.*?}|\\[a-zA-Z]+\d* ?|[{}]", " ", rtf_content)
        txt = re.sub(r"\s+", " ", txt)
        return txt.strip()


def read_text_with_fallbacks(p: Path) -> str:
    raw = p.read_bytes()
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


# =============================
# Article extraction
# =============================

MONTH_TRANSLATIONS = {
    "januari": "January",
    "februari": "February",
    "maart": "March",
    "april": "April",
    "mei": "May",
    "juni": "June",
    "juli": "July",
    "augustus": "August",
    "september": "September",
    "oktober": "October",
    "november": "November",
    "december": "December",
}

_END_DOC_SPLIT_RE = re.compile(r"(?is)\bEnd of Document\b")

# More tolerant than the strict blank-line version:
_BODY_RE = re.compile(
    r"(?is)\bBody\b\s*:?\s*(.*?)(?:\n\s*Load-Date:|\n\s*Copyright|\bEnd of Document\b|$)"
)

_LEN_RE = re.compile(r"(?is)\bLength:\s*(\d+)")
_SECTION_RE = re.compile(r"(?is)\bSection:\s*(.*?);")
_LOAD_DATE_RE = re.compile(r"(?is)\bLoad-Date:\s*(.*)")
_DATE_RE = re.compile(r"(?i)\b(\d{1,2}\s+[A-Za-zÀ-ÿ]+\s+\d{4})\b")  # supports Dutch months too


def translate_dutch_months(s: str) -> str:
    out = str(s)
    for nl, en in MONTH_TRANSLATIONS.items():
        out = re.sub(rf"\b{nl}\b", en, out, flags=re.IGNORECASE)
    return out


def extract_title(article: str) -> str:
    for line in article.splitlines():
        t = line.strip()
        if t:
            return t
    return ""


def extract_date_str(article: str) -> str:
    m = _DATE_RE.search(article)
    return m.group(1).strip() if m else ""


def extract_newspaper(article: str, date_str: str) -> str:
    """
    Best-effort:
    - Find the line containing the date; newspaper often sits on the previous non-empty line.
    - Fallback to the 2nd non-empty line.
    """
    lines = [ln.rstrip() for ln in article.splitlines()]
    if date_str:
        for i, ln in enumerate(lines):
            if date_str in ln:
                # previous non-empty line
                for j in range(i - 1, -1, -1):
                    cand = lines[j].strip()
                    if cand:
                        return cand
                break

    non_empty = [ln.strip() for ln in lines if ln.strip()]
    if len(non_empty) >= 2:
        return non_empty[1]
    return ""


def extract_body(article: str) -> str:
    m = _BODY_RE.search(article)
    if m:
        return m.group(1).strip()

    # Fallback: try everything after "Body" marker until end markers
    m2 = re.search(r"(?is)\bBody\b\s*:?\s*", article)
    if not m2:
        return ""
    rest = article[m2.end():]
    end = re.search(r"(?is)\n\s*(Load-Date:|Copyright|End of Document)\b", rest)
    if end:
        rest = rest[:end.start()]
    return rest.strip()


def load_all_rtf_articles(rtf_dir: Path) -> pd.DataFrame:
    rtf_paths = sorted(rtf_dir.rglob("*.RTF"))
    print(f"[load] Found {len(rtf_paths)} .rtf files under: {rtf_dir}")

    rows: List[Dict[str, object]] = []
    for fp in tqdm(rtf_paths, desc="Reading RTFs"):
        rtf_content = read_text_with_fallbacks(fp)
        plain_text = rtf_to_text(rtf_content)

        # Split into Lexis-like article blocks
        articles = [a for a in _END_DOC_SPLIT_RE.split(plain_text) if a and a.strip()]

        for art in articles:
            title = extract_title(art)
            date_str = extract_date_str(art)
            newspaper = extract_newspaper(art, date_str)
            section = (_SECTION_RE.search(art).group(1).strip() if _SECTION_RE.search(art) else "")
            length_m = _LEN_RE.search(art)
            word_count = int(length_m.group(1)) if length_m else None
            load_date = (_LOAD_DATE_RE.search(art).group(1).strip() if _LOAD_DATE_RE.search(art) else "")
            body = extract_body(art)

            rows.append(
                {
                    "source_file": fp.name,
                    "source_path": str(fp),
                    "title": title,
                    "newspaper": newspaper,
                    "date": date_str,
                    "section": section,
                    "word_count": word_count,
                    "load_date": load_date,
                    "body": body,
                }
            )

    df = pd.DataFrame(rows)
    return df


# =============================
# Cleaning + mapping
# =============================

def normalize_key(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def add_region_name(df: pd.DataFrame, mapping_csv: Path) -> pd.DataFrame:
    df = df.copy()
    df["region_name"] = ""

    if not mapping_csv.exists():
        print(f"[map] Mapping CSV not found, skipping: {mapping_csv}")
        return df

    m = pd.read_csv(mapping_csv)
    if not {"newspaper", "region_name"}.issubset(set(m.columns)):
        print(f"[map] Mapping CSV missing required columns (newspaper, region_name), skipping: {mapping_csv}")
        return df

    map_dict = dict(zip(m["newspaper"].map(normalize_key), m["region_name"].astype(str)))
    df["region_name"] = df["newspaper"].map(lambda x: map_dict.get(normalize_key(x), ""))
    return df


def compute_geo_hits(df: pd.DataFrame, patterns: List[str]) -> pd.Series:
    regs = [re.compile(p, flags=re.IGNORECASE) for p in patterns]

    def hits(text: str) -> int:
        t = text or ""
        return int(sum(len(r.findall(t)) for r in regs))

    return df["body"].fillna("").astype(str).map(hits)


def clean_articles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Basic sanity: ensure columns exist
    for col in ["title", "newspaper", "date", "body"]:
        if col not in df.columns:
            df[col] = ""

    # Parse dates (Dutch -> English -> datetime)
    df["date_translated"] = df["date"].fillna("").astype(str).map(translate_dutch_months)
    df["date_parsed"] = pd.to_datetime(df["date_translated"], errors="coerce", dayfirst=True)

    # Drop rows with no body
    df["body"] = df["body"].fillna("").astype(str)
    df["body_is_empty"] = df["body"].str.strip().eq("")
    empty_rate = float(df["body_is_empty"].mean()) if len(df) else 0.0
    print(f"[clean] Empty-body rate: {empty_rate:.1%} ({df['body_is_empty'].sum()} / {len(df)})")
    df = df[~df["body_is_empty"]].copy()

    # Word count
    if "word_count" not in df.columns:
        df["word_count"] = None
    df["word_count"] = df["word_count"].where(df["word_count"].notna(), df["body"].str.split().str.len())
    df["word_count"] = df["word_count"].fillna(0).astype(int)

    # Geo hits (always compute; optional filter)
    df["geo_hits"] = compute_geo_hits(df, GEOTHERMAL_PATTERNS)

    # Optional filters
    df = df[df["word_count"] >= MIN_WORDS_PER_ARTICLE].copy()
    if FILTER_BY_GEO_HITS:
        df = df[df["geo_hits"] >= MIN_GEO_HITS].copy()

    # Optional language filter
    if KEEP_ONLY_DUTCH:
        try:
            from langdetect import detect  # type: ignore

            def is_dutch(text: str) -> bool:
                try:
                    return detect(text) == "nl"
                except Exception:
                    return False

            df["language"] = df["body"].map(lambda t: "nl" if is_dutch(t) else "other")
            df = df[df["language"] == "nl"].copy()
        except Exception:
            print("[clean] langdetect not available; skipping Dutch-only filter.")

    # Deduplicate by body hash (simpler than set aggregation)
    df["body_hash"] = df["body"].map(lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest())
    before = len(df)
    df = df.sort_values(["date_parsed"], na_position="last").drop_duplicates("body_hash", keep="first")
    print(f"[clean] Deduplicated: {before} -> {len(df)} articles")

    # Downstream-friendly column names
    df["source"] = df["newspaper"].astype(str)
    df["document_title"] = df["title"].astype(str)
    df["publish_date"] = df["date_parsed"].dt.date.astype(str).fillna("")

    return df


# =============================
# Paragraph splitting (hard-wrap aware)
# =============================

def dewrap_hardwrap_lines(text: str) -> str:
    """
    Convert hard-wrapped single newlines into spaces, keep blank lines as paragraph breaks.
    """
    if not text:
        return ""

    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    out_lines: List[str] = []
    buf: List[str] = []

    def flush_buf():
        if not buf:
            return
        out_lines.append(" ".join(buf).strip())
        buf.clear()

    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            flush_buf()
            out_lines.append("")  # paragraph break
            continue

        # If previous line ended with hyphen, merge without space
        if buf and buf[-1].endswith("-") and len(buf[-1]) > 1:
            buf[-1] = buf[-1][:-1] + line
        else:
            buf.append(line)

    flush_buf()

    # collapse multiple blank lines
    cleaned = "\n".join(out_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def split_paragraphs(text: str) -> List[str]:
    text = dewrap_hardwrap_lines(text)
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p and p.strip()]
    paras = [p for p in paras if len(p.split()) >= MIN_WORDS_PER_PARAGRAPH]
    return paras



def make_paragraph_uid(source: str, title: str, publish_date: str, paragraph_id: int, paragraph_text: str) -> str:
    base = "||".join([source, title, publish_date, str(paragraph_id), paragraph_text])
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


# =============================
# Main
# =============================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=str, default=str(PROJECT_DIR))
    ap.add_argument("--input-rtf-dir", type=str, default=str(INPUT_RTF_DIR))
    ap.add_argument("--newspaper-region-csv", type=str, default=str(NEWSPAPER_REGION_CSV))
    ap.add_argument("--output-paragraph-csv", type=str, default=str(OUTPUT_PARAGRAPH_CSV))
    ap.add_argument("--output-articles-csv", type=str, default=str(OUTPUT_ARTICLES_CSV))
    ap.add_argument("--write-articles-csv", action="store_true", default=WRITE_ARTICLES_CSV)
    ap.add_argument("--no-write-articles-csv", action="store_false", dest="write_articles_csv")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    project_dir = Path(args.project_dir).expanduser().resolve()
    input_rtf_dir = Path(args.input_rtf_dir).expanduser().resolve()
    newspaper_region_csv = Path(args.newspaper_region_csv).expanduser().resolve()
    output_paragraph_csv = Path(args.output_paragraph_csv).expanduser().resolve()
    output_articles_csv = Path(args.output_articles_csv).expanduser().resolve()
    output_dir = output_paragraph_csv.parent

    os.chdir(project_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_rtf_dir.exists():
        raise FileNotFoundError(f"INPUT_RTF_DIR does not exist: {input_rtf_dir}")

    # 1) Load raw articles
    df_raw = load_all_rtf_articles(input_rtf_dir)
    print(f"[main] Loaded {len(df_raw)} raw article blocks (after splitting)")

    if df_raw.empty:
        print("[main] No articles found. Check INPUT_RTF_DIR and whether files end with .RTF.")
        output_paragraph_csv.write_text("", encoding="utf-8")
        return

    # Quick diagnostics (very helpful when things go wrong)
    raw_empty_body = float(df_raw["body"].fillna("").astype(str).str.strip().eq("").mean())
    print(f"[main] Raw empty-body rate: {raw_empty_body:.1%}")

    # 2) Clean
    df = clean_articles(df_raw)
    print(f"[main] Articles after cleaning: {len(df)}")

    # 3) Region mapping
    df = add_region_name(df, newspaper_region_csv)

    # Optional articles CSV
    if args.write_articles_csv:
        output_articles_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_articles_csv, index=False, encoding="utf-8")
        print(f"[main] Wrote cleaned articles: {output_articles_csv}")

    # 4) Build paragraph rows
    paragraph_rows: List[Dict[str, object]] = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Splitting paragraphs"):
        paras = split_paragraphs(str(r.get("body", "")))
        if not paras:
            continue

        source = str(r.get("source", ""))
        title = str(r.get("document_title", ""))
        publish_date = str(r.get("publish_date", ""))

        for i, p in enumerate(paras, start=1):
            paragraph_rows.append(
                {
                    **r.to_dict(),
                    "paragraph_id": i,
                    "paragraph_text": p,
                    "uid": make_paragraph_uid(source, title, publish_date, i, p),
                }
            )

    df_paras = pd.DataFrame(paragraph_rows)
    print(f"[main] Paragraph rows: {len(df_paras)}")

    if df_paras.empty:
        print("[main] WARNING: No paragraphs generated. This usually means body extraction failed or everything was filtered out.")
        df_paras.to_csv(output_paragraph_csv, index=False, encoding="utf-8")
        print(f"[main] Wrote empty CSV (for debugging): {output_paragraph_csv}")
        return

    # 6) Write
    df_paras.to_csv(output_paragraph_csv, index=False, encoding="utf-8")
    print(f"[main] Wrote paragraphs+sentiment: {output_paragraph_csv} (rows={len(df_paras)})")


if __name__ == "__main__":
    main()
