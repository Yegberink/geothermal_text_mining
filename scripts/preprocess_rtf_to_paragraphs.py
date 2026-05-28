#!/usr/bin/env python3
"""
Standalone newspaper RTF -> paragraph pipeline.

What this does:
1) Recursively reads all .rtf files under INPUT_RTF_DIR
2) Converts RTF -> plain text via striprtf, splits into articles by "End of Document"
3) Extracts metadata (title/newspaper/date/etc.) + body
4) Optional cleaning + optional filtering (all controlled by constants below)
5) Splits body into paragraphs (fixes hard-wrapped newlines)
6) Writes output CSV: OUTPUT_PARAGRAPH_CSV
   (Optional) Writes cleaned-articles CSV: OUTPUT_ARTICLES_CSV

No CLI arguments needed. Edit the constants in the CONFIG section,
or pass them as CLI flags (see parse_args).
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import argparse

import pandas as pd
import yaml
from striprtf.striprtf import rtf_to_text as _striprtf_to_text
from tqdm.auto import tqdm

from language_resources import load_date_locale, load_geothermal_patterns


# =============================
# CONFIG (edit these)
# =============================

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_RTF_DIR = PROJECT_DIR / "input_data"
NEWSPAPER_REGION_CSV = PROJECT_DIR / "data" / "dutch" / "newspaper_region_mapping.csv"
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "config.yaml"

OUTPUT_DIR = PROJECT_DIR / "output" / "text"
OUTPUT_PARAGRAPH_CSV = OUTPUT_DIR / "newspapers_cleaned_paragraphs.csv"
OUTPUT_ARTICLES_CSV = OUTPUT_DIR / "articles_cleaned.csv"
OUTPUT_RAW_ARTICLES_CSV = OUTPUT_DIR / "raw_articles.csv"
WRITE_ARTICLES_CSV = True

MIN_WORDS_PER_ARTICLE = 100
FILTER_BY_GEO_HITS = False
MIN_GEO_HITS = 3

MIN_WORDS_PER_PARAGRAPH = 40
TARGET_WORDS_PER_PARAGRAPH = 120
MAX_WORDS_PER_PARAGRAPH = 180
MAX_SENTENCES_PER_PARAGRAPH = 6

RAW_ARTICLE_COLUMNS = [
    "source_file",
    "source_path",
    "source_relative_path",
    "source_folder",
    "title",
    "newspaper",
    "date",
    "section",
    "word_count",
    "load_date",
    "body",
]


# =============================
# Precompiled regexes (module-level — compiled once)
# =============================

_END_DOC_SPLIT_RE = re.compile(r"(?is)\bEnd of Document\b")
_BODY_RE = re.compile(
    r"(?is)\bBody\b\s*:?\s*(.*?)(?:\n\s*Load-Date:|\n\s*Copyright|\bEnd of Document\b|$)"
)
_LEN_RE = re.compile(r"(?is)\bLength:\s*(\d+)")
_SECTION_RE = re.compile(r"(?is)\bSection:\s*(.*?);")
_LOAD_DATE_RE = re.compile(r"(?is)\bLoad-Date:\s*(.*?)(?:\n|$)")
_DATE_RE = re.compile(r"(?i)\b(\d{1,2}\.?\s+[A-Za-zÀ-ÿ]+\s+\d{4})\b")
_HEADER_STOP_RE = re.compile(
    r"(?i)^(copyright|section:|length:|byline:|highlight:|body\b|load-date:)"
)
_DATE_LINE_RE = re.compile(r"(?i)^\s*\d{1,2}\.?\s+[A-Za-zÀ-ÿ]+\s+\d{4}(?:\s+[A-Za-zÀ-ÿ])?\s*$")
_CORRUPT_HEX_RE = re.compile(r"\b[0-9A-Fa-f]{16,}\b")
_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"\s+")

# Artifact-stripping regexes
_ARTIFACT_LEADING_DASH_RE = re.compile(r"(?m)^\s*-\d+\s*")
_ARTIFACT_PAGE_OF_RE = re.compile(r"\b-?\d+\s+Page of\s+-?\d+\b", re.IGNORECASE)
_ARTIFACT_CONSECUTIVE_NUM_RE = re.compile(r"\b-?\d+\b(?=\s+-?\d+\b)")
_ARTIFACT_HEX_RE = re.compile(r"\b[0-9A-Fa-f]{32,}\b")
_ARTIFACT_RTF_ESC_RE = re.compile(r"\\'[0-9a-fA-F]{2}")
_ARTIFACT_RTF_STAR_RE = re.compile(r"\\\*")
_ARTIFACT_BEKIJK_RE = re.compile(
    r"\bBekijk de oorspronkelijke pagina:.*$", re.IGNORECASE | re.MULTILINE
)
_ARTIFACT_GRAPHIC_RE = re.compile(r"\bGraphic\b", re.IGNORECASE)


# =============================
# RTF -> plain text
# =============================

def rtf_to_text(rtf_content: str) -> str:
    return _striprtf_to_text(rtf_content)


def read_rtf_file(p: Path) -> str:
    raw = p.read_bytes()
    for enc in ("utf-8", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {p} with utf-8 or cp1252")


def discover_rtf_paths(rtf_dir: Path) -> List[Path]:
    """Find real RTF article exports recursively, including nested German folders."""
    all_paths = sorted(p for p in rtf_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".rtf")
    doclist_paths = [p for p in all_paths if "doclist" in p.stem.lower()]
    if doclist_paths:
        print(f"[load] Skipping Lexis document-list RTF files: {len(doclist_paths)}")
    return [p for p in all_paths if p not in set(doclist_paths)]


def source_path_parts(fp: Path, rtf_root: Path) -> Tuple[str, str]:
    try:
        source_relative_path = str(fp.relative_to(rtf_root))
        source_folder = str(fp.parent.relative_to(rtf_root))
    except ValueError:
        source_relative_path = fp.name
        source_folder = ""
    if source_folder == ".":
        source_folder = ""
    return source_relative_path, source_folder


# =============================
# Article extraction
# =============================

def normalize_article_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).replace("\ufeff", "").replace("\u00a0", " ")
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def load_preprocessing_config(
    config_path: Path,
    project_dir: Path,
    language: str,
) -> Tuple[Dict[str, str], List[str]]:
    date_locale = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        preprocessing = config.get("preprocessing", {}) or {}
        date_locale = preprocessing.get("date_locale", {}) or {}
    return load_date_locale(project_dir, language, date_locale)


def load_workflow_language(config_path: Path) -> str:
    if not config_path.exists():
        return "dutch"
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return str(config.get("language") or "dutch")


def translate_month_names(s: str, month_translations: Dict[str, str]) -> str:
    out = str(s)
    for source_name, english_name in month_translations.items():
        out = re.sub(rf"\b{re.escape(source_name)}\b", english_name, out, flags=re.IGNORECASE)
    return out


def cleanup_date_text(s: str, month_translations: Dict[str, str], weekday_names: List[str]) -> str:
    text = translate_month_names(s, month_translations)
    weekday_pattern = r"\b(" + "|".join(sorted(re.escape(d) for d in weekday_names)) + r")\b"
    text = re.sub(weekday_pattern, "", text, flags=re.IGNORECASE)
    text = text.replace(",", " ")
    text = re.sub(r"\b(\d{1,2})\.\s+", r"\1 ", text)
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def split_header_body(article: str) -> Tuple[List[str], str]:
    normalized = normalize_article_text(article)
    lines = [ln.strip() for ln in normalized.split("\n")]

    body_idx = next(
        (i for i, line in enumerate(lines) if re.match(r"(?i)^body\b", line.strip())),
        None,
    )

    if body_idx is None:
        return [ln for ln in lines if ln.strip()], extract_body(normalized)

    header_lines = [ln for ln in lines[:body_idx] if ln.strip()]
    body_lines: List[str] = []
    for line in lines[body_idx + 1:]:
        if re.match(r"(?i)^load-date:", line.strip()):
            break
        body_lines.append(line)

    return header_lines, "\n".join(body_lines).strip()


def detect_date_line(
    header_lines: List[str],
    month_translations: Dict[str, str],
    weekday_names: List[str],
) -> str:
    for line in header_lines:
        if _DATE_LINE_RE.match(line):
            return line.strip()
    for line in header_lines:
        if _HEADER_STOP_RE.match(line):
            break
        if line.lower().startswith("load-date:"):
            continue
        normalized = cleanup_date_text(line, month_translations, weekday_names)
        if _DATE_RE.search(normalized):
            return line.strip()
    return ""


def parse_header_metadata(
    article: str,
    month_translations: Dict[str, str],
    weekday_names: List[str],
) -> Dict[str, object]:
    header_lines, body = split_header_body(article)

    title = header_lines[0] if header_lines else ""
    date_line = detect_date_line(header_lines, month_translations, weekday_names)
    date_norm = cleanup_date_text(date_line, month_translations, weekday_names)
    date_match = _DATE_RE.search(date_norm)
    date_str = date_match.group(1).strip() if date_match else ""

    newspaper = ""
    if date_line and date_line in header_lines:
        date_idx = header_lines.index(date_line)
        for cand in reversed(header_lines[:date_idx]):
            if not cand or cand == title or _HEADER_STOP_RE.match(cand):
                continue
            newspaper = cand
            break

    if not newspaper:
        for cand in header_lines[1:]:
            if _HEADER_STOP_RE.match(cand):
                break
            normalized = cleanup_date_text(cand, month_translations, weekday_names)
            if _DATE_RE.search(normalized):
                continue
            newspaper = cand
            break

    # Each regex searched once; result reused
    section_m = _SECTION_RE.search(article)
    length_m = _LEN_RE.search(article)
    load_date_m = _LOAD_DATE_RE.search(article)

    return {
        "title": title,
        "newspaper": newspaper,
        "date": date_str,
        "section": section_m.group(1).strip() if section_m else "",
        "word_count": int(length_m.group(1)) if length_m else None,
        "load_date": load_date_m.group(1).strip() if load_date_m else "",
        "body": body if body else extract_body(article),
    }


def extract_body(article: str) -> str:
    article = normalize_article_text(article)
    m = _BODY_RE.search(article)
    if m:
        return m.group(1).strip()

    m2 = re.search(r"(?is)\bBody\b\s*:?\s*", article)
    if not m2:
        return ""
    rest = article[m2.end():]
    end = re.search(r"(?is)\n\s*(Load-Date:|Copyright|End of Document)\b", rest)
    if end:
        rest = rest[:end.start()]
    return rest.strip()


def load_rtf_articles_from_paths(
    rtf_paths: Iterable[Path],
    rtf_root: Path,
    month_translations: Dict[str, str],
    weekday_names: List[str],
) -> pd.DataFrame:
    rtf_paths = list(rtf_paths)
    print(f"[load] Reading {len(rtf_paths)} .rtf files")

    rows: List[Dict[str, object]] = []
    for fp in tqdm(rtf_paths, desc="Reading RTFs"):
        rtf_content = read_rtf_file(fp)
        plain_text = normalize_article_text(rtf_to_text(rtf_content))
        articles = [a for a in _END_DOC_SPLIT_RE.split(plain_text) if a and a.strip()]
        source_relative_path, source_folder = source_path_parts(fp, rtf_root)

        for art in articles:
            meta = parse_header_metadata(art, month_translations, weekday_names)
            rows.append({
                "source_file": fp.name,
                "source_path": str(fp),
                "source_relative_path": source_relative_path,
                "source_folder": source_folder,
                **meta,
            })

    return pd.DataFrame(rows, columns=RAW_ARTICLE_COLUMNS)


def load_all_rtf_articles(
    rtf_dir: Path,
    month_translations: Dict[str, str],
    weekday_names: List[str],
) -> pd.DataFrame:
    rtf_paths = discover_rtf_paths(rtf_dir)
    print(f"[load] Found {len(rtf_paths)} .rtf files under: {rtf_dir}")
    return load_rtf_articles_from_paths(rtf_paths, rtf_dir, month_translations, weekday_names)


def load_single_rtf_articles(
    rtf_file: Path,
    rtf_root: Path,
    month_translations: Dict[str, str],
    weekday_names: List[str],
) -> pd.DataFrame:
    if rtf_file.stem.lower().find("doclist") >= 0:
        print(f"[load] Skipping Lexis document-list RTF file: {rtf_file}")
        return pd.DataFrame(columns=RAW_ARTICLE_COLUMNS)
    return load_rtf_articles_from_paths([rtf_file], rtf_root, month_translations, weekday_names)


def load_raw_article_csvs(raw_article_csvs: List[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for csv_path in raw_article_csvs:
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            continue
        frames.append(pd.read_csv(csv_path))
    if not frames:
        return pd.DataFrame(columns=RAW_ARTICLE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


# =============================
# Cleaning + mapping
# =============================

def normalize_key(s: str) -> str:
    return _MULTI_SPACE_RE.sub(" ", str(s).strip().lower())


def strip_extraction_artifacts(text: str) -> str:
    text = str(text or "")
    text = _ARTIFACT_LEADING_DASH_RE.sub(" ", text)
    text = _ARTIFACT_PAGE_OF_RE.sub(" ", text)
    text = _ARTIFACT_CONSECUTIVE_NUM_RE.sub(" ", text)
    text = _ARTIFACT_HEX_RE.sub(" ", text)
    text = _ARTIFACT_RTF_ESC_RE.sub(" ", text)
    text = _ARTIFACT_RTF_STAR_RE.sub(" ", text)
    text = _ARTIFACT_BEKIJK_RE.sub(" ", text)
    text = _ARTIFACT_GRAPHIC_RE.sub(" ", text)
    lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r" *\n *", "\n", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def looks_corrupt_article(row: pd.Series) -> bool:
    title = str(row.get("title", "") or "")
    newspaper = str(row.get("newspaper", "") or "")
    body = str(row.get("body", "") or "")

    if not newspaper.strip() or not title.strip():
        return True
    if title.startswith("Times New Roman") or "Page of" in title:
        return True
    if len(_CORRUPT_HEX_RE.findall(title + " " + body)) > 2:
        return True
    return False


def add_region_name(df: pd.DataFrame, mapping_csv: Path) -> pd.DataFrame:
    df = df.copy()
    df["region_name"] = ""

    if not mapping_csv.exists():
        print(f"[map] Mapping CSV not found, skipping: {mapping_csv}")
        return df

    m = pd.read_csv(mapping_csv)
    if not {"newspaper", "region_name"}.issubset(set(m.columns)):
        raise ValueError(
            f"Mapping CSV missing required columns (newspaper, region_name): {mapping_csv}"
        )

    map_dict = dict(zip(m["newspaper"].map(normalize_key), m["region_name"].astype(str)))
    df["region_name"] = df["newspaper"].map(lambda x: map_dict.get(normalize_key(x), ""))
    return df


def compute_geo_hits(df: pd.DataFrame, patterns: List[str]) -> pd.Series:
    # Combine all patterns into one regex — single pass per row
    combined = "|".join(f"(?:{p})" for p in patterns)
    geo_re = re.compile(combined, flags=re.IGNORECASE)
    return df["body"].fillna("").astype(str).map(lambda t: len(geo_re.findall(t)))


def clean_articles(
    df: pd.DataFrame,
    month_translations: Dict[str, str],
    weekday_names: List[str],
    geothermal_patterns: List[str],
) -> pd.DataFrame:
    df = df.copy()

    for col in ["title", "newspaper", "date", "body"]:
        if col not in df.columns:
            df[col] = ""

    for col in ["title", "newspaper", "section", "load_date", "body"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).map(strip_extraction_artifacts)

    corrupt_mask = df.apply(looks_corrupt_article, axis=1)
    if corrupt_mask.any():
        print(f"[clean] Dropping likely corrupt article blocks: {int(corrupt_mask.sum())}")
        df = df.loc[~corrupt_mask].copy()

    df["date_translated"] = df["date"].fillna("").astype(str).map(
        lambda x: cleanup_date_text(x, month_translations, weekday_names)
    )
    df["date_parsed"] = pd.to_datetime(df["date_translated"], errors="coerce", dayfirst=True)

    df["body"] = df["body"].fillna("").astype(str)
    empty_mask = df["body"].str.strip().eq("")
    print(f"[clean] Empty-body rate: {empty_mask.mean():.1%} ({empty_mask.sum()} / {len(df)})")
    df = df[~empty_mask].copy()

    if "word_count" not in df.columns:
        df["word_count"] = None
    df["word_count"] = df["word_count"].where(
        df["word_count"].notna(), df["body"].str.split().str.len()
    )
    df["word_count"] = df["word_count"].fillna(0).astype(int)

    df["geo_hits"] = compute_geo_hits(df, geothermal_patterns)
    df = df[df["word_count"] >= MIN_WORDS_PER_ARTICLE].copy()

    if FILTER_BY_GEO_HITS:
        df = df[df["geo_hits"] >= MIN_GEO_HITS].copy()

    df["body_hash"] = df["body"].map(lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest())
    before = len(df)
    df = df.sort_values(["date_parsed"], na_position="last").drop_duplicates(
        "body_hash", keep="first"
    )
    print(f"[clean] Deduplicated: {before} -> {len(df)} articles")

    df["source"] = df["newspaper"].astype(str)
    df["document_title"] = df["title"].astype(str)
    df["publish_date"] = df["date_parsed"].dt.date.astype(str).fillna("")

    return df


# =============================
# Paragraph splitting (content-based, hard-wrap aware)
# =============================

SentenceSplitFunc = Callable[[object, str], List[str]]


def normalize_layout_text(text: str) -> str:
    """
    Merge single-newline layout wraps while preserving blank lines as soft hints.
    """
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""

    text = re.sub(r"\n\s*\n+", "\n\n", text)
    blocks: List[str] = []

    for block in text.split("\n\n"):
        lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in block.split("\n")]
        lines = [line for line in lines if line]

        merged: List[str] = []
        for line in lines:
            if merged and merged[-1].endswith("-") and len(merged[-1]) > 1:
                merged[-1] = merged[-1][:-1] + line
            else:
                merged.append(line)

        block_text = " ".join(merged).strip()
        if block_text:
            blocks.append(block_text)

    return "\n\n".join(blocks)


def fallback_split_sentences(text: str) -> List[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return []
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Ý])", cleaned)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences or [cleaned]


def build_paragraph_sentence_segmenter(
    language: str,
) -> Tuple[object, Optional[SentenceSplitFunc]]:
    try:
        from split_paragraphs_to_sentences import build_segmenter, split_sentences

        return build_segmenter(language), split_sentences
    except Exception as exc:
        print(
            "[main] WARNING: Falling back to regex sentence splitting for paragraphs: "
            f"{exc!r}"
        )
        return None, None


def split_sentences_multilingual(
    text: str,
    language: str,
    nlp: object = None,
    split_func: Optional[SentenceSplitFunc] = None,
) -> List[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return []

    if split_func is None:
        return fallback_split_sentences(cleaned)

    try:
        sentences = split_func(nlp, cleaned)
    except Exception:
        return fallback_split_sentences(cleaned)
    return [sentence.strip() for sentence in sentences if sentence and sentence.strip()]


def is_probable_heading(text: str) -> bool:
    words = str(text or "").split()
    if not words or len(words) > 12:
        return False
    if str(text).strip().endswith((".", "!", "?", ";", ":")):
        return False
    return True


def group_sentences_into_paragraphs(
    sentences: List[str],
    min_words: int = MIN_WORDS_PER_PARAGRAPH,
    target_words: int = TARGET_WORDS_PER_PARAGRAPH,
    max_words: int = MAX_WORDS_PER_PARAGRAPH,
    max_sentences: int = MAX_SENTENCES_PER_PARAGRAPH,
) -> List[str]:
    paragraph_sentence_groups: List[List[str]] = []
    current: List[str] = []
    current_words = 0

    for sentence in sentences:
        sentence = str(sentence or "").strip()
        if not sentence:
            continue

        sentence_words = len(sentence.split())
        should_flush = (
            current
            and (
                current_words + sentence_words > max_words
                or len(current) >= max_sentences
                or current_words >= target_words
            )
        )

        if should_flush:
            paragraph_sentence_groups.append(current)
            current = []
            current_words = 0

        current.append(sentence)
        current_words += sentence_words

    if current:
        paragraph_sentence_groups.append(current)

    merged_groups: List[List[str]] = []
    carry: List[str] = []
    for group in paragraph_sentence_groups:
        if carry:
            candidate = carry + group
            candidate_words = sum(len(sentence.split()) for sentence in candidate)
            if candidate_words <= max_words and len(candidate) <= max_sentences:
                group = candidate
                carry = []
            else:
                merged_groups.append(carry)
                carry = []

        group_words = sum(len(sentence.split()) for sentence in group)
        if group_words < min_words:
            carry = group
        else:
            merged_groups.append(group)

    if carry:
        if merged_groups:
            candidate = merged_groups[-1] + carry
            candidate_words = sum(len(sentence.split()) for sentence in candidate)
            if candidate_words <= max_words and len(candidate) <= max_sentences:
                merged_groups[-1] = candidate
            else:
                merged_groups.append(carry)
        else:
            merged_groups.append(carry)

    return [" ".join(group).strip() for group in merged_groups]


def split_paragraphs_content_based(
    text: str,
    language: str,
    nlp: object = None,
    split_func: Optional[SentenceSplitFunc] = None,
) -> List[str]:
    normalized = normalize_layout_text(text)
    if not normalized:
        return []

    output: List[str] = []
    heading_carry = ""

    for block in normalized.split("\n\n"):
        block = block.strip()
        if not block:
            continue

        if is_probable_heading(block):
            heading_carry = f"{heading_carry} {block}".strip()
            continue

        sentences = split_sentences_multilingual(block, language, nlp, split_func)
        paragraphs = group_sentences_into_paragraphs(sentences)
        if heading_carry and paragraphs:
            paragraphs[0] = f"{heading_carry} {paragraphs[0]}".strip()
            heading_carry = ""
        output.extend(paragraphs)

    if heading_carry:
        if output:
            last_with_heading = f"{output[-1]} {heading_carry}".strip()
            sentence_count = len(
                split_sentences_multilingual(last_with_heading, language, nlp, split_func)
            )
            word_count = len(last_with_heading.split())
            if sentence_count <= MAX_SENTENCES_PER_PARAGRAPH and word_count <= MAX_WORDS_PER_PARAGRAPH:
                output[-1] = last_with_heading
        else:
            output.append(heading_carry)

    return [p for p in output if len(p.split()) >= MIN_WORDS_PER_PARAGRAPH]


def split_paragraphs(text: str, language: str = "nl") -> List[str]:
    return split_paragraphs_content_based(text, language)


def estimate_sentence_count(text: str) -> int:
    cleaned = str(text or "").strip()
    if not cleaned:
        return 0
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Ý])", cleaned)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return len(sentences) if sentences else 1


def format_numeric_stats(values: pd.Series) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return "n=0"
    return (
        f"n={len(values)}, mean={values.mean():.2f}, median={values.median():.2f}, "
        f"p90={values.quantile(0.90):.2f}, p95={values.quantile(0.95):.2f}, "
        f"p99={values.quantile(0.99):.2f}, max={values.max():.0f}"
    )


def sentence_counts_for_paragraphs(paragraph_text: pd.Series, language: str) -> pd.Series:
    try:
        from split_paragraphs_to_sentences import build_segmenter, split_sentences

        nlp = build_segmenter(language)
        return paragraph_text.map(lambda value: len(split_sentences(nlp, value)))
    except Exception as exc:
        print(
            "[main] WARNING: Falling back to regex sentence counts for paragraph split stats: "
            f"{exc!r}"
        )
        return paragraph_text.map(estimate_sentence_count)


def print_paragraph_split_stats(df_paras: pd.DataFrame, language: str) -> None:
    if df_paras.empty:
        return

    doc_key = "body_hash" if "body_hash" in df_paras.columns else None
    if doc_key is not None:
        paragraphs_per_doc = df_paras.groupby(doc_key).size()
        print("[main] Paragraphs per document:", format_numeric_stats(paragraphs_per_doc))
        print(f"[workflow_table] paragraphs_per_document_mean: {paragraphs_per_doc.mean():.2f}")
        print(f"[workflow_table] paragraphs_per_document_median: {paragraphs_per_doc.median():.2f}")
        print(f"[workflow_table] paragraphs_per_document_p95: {paragraphs_per_doc.quantile(0.95):.2f}")
        print(f"[workflow_table] paragraphs_per_document_max: {int(paragraphs_per_doc.max())}")

        outlier_threshold = max(20, paragraphs_per_doc.quantile(0.99))
        outliers = paragraphs_per_doc[paragraphs_per_doc >= outlier_threshold].sort_values(ascending=False)
        print(
            f"[main] Documents with many paragraphs: "
            f"{len(outliers)} documents with >= {outlier_threshold:.0f} paragraphs"
        )
        print(f"[workflow_table] paragraph_heavy_documents_threshold: {outlier_threshold:.0f}")
        print(f"[workflow_table] paragraph_heavy_documents_count: {len(outliers)}")

        if not outliers.empty:
            meta_cols = [c for c in ["source", "document_title", "publish_date"] if c in df_paras.columns]
            doc_meta = df_paras.drop_duplicates(doc_key).set_index(doc_key)
            print("[main] Top documents by paragraph count:")
            for body_hash, n_paragraphs in outliers.head(10).items():
                meta = doc_meta.loc[body_hash, meta_cols].to_dict() if meta_cols else {}
                title = str(meta.get("document_title", ""))[:90]
                source = str(meta.get("source", ""))
                date = str(meta.get("publish_date", ""))
                print(
                    f"  - paragraphs={int(n_paragraphs)} | "
                    f"source={source} | date={date} | title={title}"
                )

    paragraph_text = df_paras["paragraph_text"].fillna("").astype(str)
    paragraph_word_counts = paragraph_text.map(lambda value: len(value.split()))
    paragraph_sentence_counts = sentence_counts_for_paragraphs(paragraph_text, language)

    print("[main] Sentences per paragraph:", format_numeric_stats(paragraph_sentence_counts))
    print("[main] Words per paragraph:", format_numeric_stats(paragraph_word_counts))
    print(f"[workflow_table] sentences_per_paragraph_mean: {paragraph_sentence_counts.mean():.2f}")
    print(f"[workflow_table] sentences_per_paragraph_median: {paragraph_sentence_counts.median():.2f}")
    print(f"[workflow_table] sentences_per_paragraph_p95: {paragraph_sentence_counts.quantile(0.95):.2f}")
    print(f"[workflow_table] sentences_per_paragraph_max: {int(paragraph_sentence_counts.max())}")
    print(f"[workflow_table] words_per_paragraph_mean: {paragraph_word_counts.mean():.2f}")
    print(f"[workflow_table] words_per_paragraph_median: {paragraph_word_counts.median():.2f}")
    print(f"[workflow_table] words_per_paragraph_p95: {paragraph_word_counts.quantile(0.95):.2f}")
    print(f"[workflow_table] words_per_paragraph_max: {int(paragraph_word_counts.max())}")


def make_paragraph_uid(
    source: str, title: str, publish_date: str, paragraph_id: int, paragraph_text: str
) -> str:
    base = "||".join([source, title, publish_date, str(paragraph_id), paragraph_text])
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


# =============================
# CLI
# =============================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=str, default=str(PROJECT_DIR))
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--input-rtf-dir", type=str, default=str(INPUT_RTF_DIR))
    ap.add_argument("--input-rtf-file", type=str, default="")
    ap.add_argument("--input-raw-articles-csv", type=str, nargs="+", default=[])
    ap.add_argument("--output-raw-articles-csv", type=str, default=str(OUTPUT_RAW_ARTICLES_CSV))
    ap.add_argument("--raw-only", action="store_true")
    ap.add_argument("--newspaper-region-csv", type=str, default=str(NEWSPAPER_REGION_CSV))
    ap.add_argument("--output-paragraph-csv", type=str, default=str(OUTPUT_PARAGRAPH_CSV))
    ap.add_argument("--output-articles-csv", type=str, default=str(OUTPUT_ARTICLES_CSV))
    ap.add_argument("--write-articles-csv", action="store_true", default=WRITE_ARTICLES_CSV)
    ap.add_argument("--no-write-articles-csv", action="store_false", dest="write_articles_csv")
    return ap.parse_args()


# =============================
# Main
# =============================

def main() -> None:
    args = parse_args()

    project_dir = Path(args.project_dir).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    input_rtf_dir = Path(args.input_rtf_dir).expanduser().resolve()
    input_rtf_file = Path(args.input_rtf_file).expanduser().resolve() if args.input_rtf_file else None
    input_raw_article_csvs = [
        Path(path).expanduser().resolve() for path in args.input_raw_articles_csv
    ]
    newspaper_region_csv = Path(args.newspaper_region_csv).expanduser().resolve()
    output_paragraph_csv = Path(args.output_paragraph_csv).expanduser().resolve()
    output_articles_csv = Path(args.output_articles_csv).expanduser().resolve()
    output_raw_articles_csv = Path(args.output_raw_articles_csv).expanduser().resolve()

    os.chdir(project_dir)
    output_paragraph_csv.parent.mkdir(parents=True, exist_ok=True)

    workflow_language = load_workflow_language(config_path)
    month_translations, weekday_names = load_preprocessing_config(
        config_path,
        project_dir,
        workflow_language,
    )
    geothermal_patterns = load_geothermal_patterns(project_dir, workflow_language)
    paragraph_nlp, paragraph_split_func = build_paragraph_sentence_segmenter(workflow_language)

    if input_raw_article_csvs:
        missing_raw_csvs = [path for path in input_raw_article_csvs if not path.exists()]
        if missing_raw_csvs:
            raise FileNotFoundError(
                "Input raw article CSV(s) do not exist: "
                + ", ".join(str(path) for path in missing_raw_csvs)
            )
    elif input_rtf_file is not None and not input_rtf_file.exists():
        raise FileNotFoundError(f"Input RTF file does not exist: {input_rtf_file}")
    elif not input_rtf_dir.exists():
        raise FileNotFoundError(f"INPUT_RTF_DIR does not exist: {input_rtf_dir}")

    # 1) Load raw articles
    if input_raw_article_csvs:
        df_raw = load_raw_article_csvs(input_raw_article_csvs)
        print(f"[main] Loaded raw article chunks: {len(input_raw_article_csvs)} CSV files")
    elif input_rtf_file is not None:
        df_raw = load_single_rtf_articles(
            input_rtf_file,
            input_rtf_dir,
            month_translations,
            weekday_names,
        )
    else:
        df_raw = load_all_rtf_articles(input_rtf_dir, month_translations, weekday_names)
    print(f"[main] Loaded {len(df_raw)} raw article blocks")
    print(f"[workflow_table] raw_article_blocks: {len(df_raw)}")

    if args.raw_only:
        output_raw_articles_csv.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_csv(output_raw_articles_csv, index=False, encoding="utf-8")
        print(f"[main] Wrote raw articles: {output_raw_articles_csv} (rows={len(df_raw)})")
        return

    if df_raw.empty:
        print("[main] No articles found. Check INPUT_RTF_DIR and that files end with .rtf/.RTF")
        output_paragraph_csv.write_text("", encoding="utf-8")
        return

    raw_empty_body = df_raw["body"].fillna("").astype(str).str.strip().eq("").mean()
    print(f"[main] Raw empty-body rate: {raw_empty_body:.1%}")

    # 2) Clean
    df = clean_articles(df_raw, month_translations, weekday_names, geothermal_patterns)
    print(f"[main] Articles after cleaning: {len(df)}")
    print(f"[workflow_table] cleaned_articles: {len(df)}")

    # 3) Region mapping
    df = add_region_name(df, newspaper_region_csv)

    # 4) Optionally write articles CSV
    if args.write_articles_csv:
        output_articles_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_articles_csv, index=False, encoding="utf-8")
        print(f"[main] Wrote cleaned articles: {output_articles_csv}")

    # 5) Build paragraph rows
    paragraph_rows: List[Dict[str, object]] = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Splitting paragraphs"):
        paras = split_paragraphs_content_based(
            str(r.get("body", "")),
            workflow_language,
            paragraph_nlp,
            paragraph_split_func,
        )
        if not paras:
            continue

        source = str(r.get("source", ""))
        title = str(r.get("document_title", ""))
        publish_date = str(r.get("publish_date", ""))

        for i, p in enumerate(paras, start=1):
            paragraph_rows.append({
                **r.to_dict(),
                "paragraph_id": i,
                "paragraph_text": p,
                "uid": make_paragraph_uid(source, title, publish_date, i, p),
            })

    df_paras = pd.DataFrame(paragraph_rows)
    print(f"[main] Paragraph rows: {len(df_paras)}")
    print(f"[workflow_table] paragraphs_after_split_filter: {len(df_paras)}")
    if not df_paras.empty and "body_hash" in df_paras.columns:
        paragraphs_per_article = df_paras.groupby("body_hash").size()
        articles_with_multiple_paragraphs = int((paragraphs_per_article > 1).sum())
        print(
            "[main] Paragraphs per article: "
            f"mean={paragraphs_per_article.mean():.2f}, "
            f"max={int(paragraphs_per_article.max())}, "
            f"multi_paragraph_articles={articles_with_multiple_paragraphs}/{len(paragraphs_per_article)}"
        )
        print(f"[workflow_table] articles_with_paragraphs: {len(paragraphs_per_article)}")
        print(f"[workflow_table] articles_with_multiple_paragraphs: {articles_with_multiple_paragraphs}")
        print(f"[workflow_table] mean_paragraphs_per_article: {paragraphs_per_article.mean():.2f}")

    print_paragraph_split_stats(df_paras, workflow_language)

    if df_paras.empty:
        print(
            "[main] WARNING: No paragraphs generated. "
            "Body extraction may have failed or everything was filtered out."
        )

    df_paras.to_csv(output_paragraph_csv, index=False, encoding="utf-8")
    print(f"[main] Wrote paragraphs CSV: {output_paragraph_csv} (rows={len(df_paras)})")


if __name__ == "__main__":
    main()
