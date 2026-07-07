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
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import warnings

import pandas as pd
import yaml

from helpers.country_scope import country_scope_from_config
from striprtf.striprtf import rtf_to_text as _striprtf_to_text
from tqdm.auto import tqdm

from helpers.language_resources import load_date_locale, load_geothermal_patterns, normalize_language


# =============================
# CONFIG (edit these)
# =============================

PROJECT_DIR = Path(__file__).resolve().parents[2]
INPUT_RTF_DIR = PROJECT_DIR / "data" / "text_data"
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "config.yaml"

OUTPUT_DIR = PROJECT_DIR / "output" / "workflow"
OUTPUT_PARAGRAPH_CSV = OUTPUT_DIR / "newspapers_cleaned_paragraphs.csv"
OUTPUT_ARTICLES_CSV = OUTPUT_DIR / "articles_cleaned.csv"
OUTPUT_RAW_ARTICLES_CSV = OUTPUT_DIR / "raw_articles.csv"
OUTPUT_PARAGRAPH_SAMPLE_CSV = OUTPUT_DIR / "paragraph_split_sample.csv"
OUTPUT_PARAGRAPH_SAMPLE_MD = OUTPUT_DIR / "paragraph_split_sample.md"
OUTPUT_LAYOUT_DEBUG_MD = OUTPUT_DIR / "layout_debug_sample.md"
WRITE_ARTICLES_CSV = True

MIN_WORDS_PER_ARTICLE = 100
FILTER_BY_GEO_HITS = False
MIN_GEO_HITS = 3

MIN_WORDS_PER_PARAGRAPH = 40
TARGET_WORDS_PER_PARAGRAPH = 120
MAX_WORDS_PER_PARAGRAPH = 180
MAX_SENTENCES_PER_PARAGRAPH = 6
ENABLE_CONTENT_AWARE_PARAGRAPHS = True
CONTENT_AWARE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CONTENT_SHIFT_THRESHOLD = 0.48
MICRO_SEGMENT_SENTENCES = 2
MAX_WORDS_PER_CONTENT_PARAGRAPH = MAX_WORDS_PER_PARAGRAPH
PARAGRAPH_SAMPLE_RANDOM_SEED = 42
PARAGRAPH_EMBEDDING_BACKEND = "transformers"
PRESERVE_SINGLE_NEWLINE_PARAGRAPHS = True
SEMANTIC_SPLIT_MIN_WORDS = MAX_WORDS_PER_PARAGRAPH
ENABLE_SEMANTIC_SPLIT_FOR_NORMAL_BLOCKS = False
MIN_WORDS_FOR_STANDALONE_LAYOUT_PARAGRAPH = 25
MIN_WORDS_FOR_ATTACHED_HEADING = 1
LAYOUT_DEBUG_PREVIEW_CHARS = 1200
VERBOSE = False

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


def log(message: str) -> None:
    if VERBOSE:
        print(message)


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
_WORD_RE = re.compile(r"\b[\wÀ-ÿ]+(?:[-'][\wÀ-ÿ]+)*\b", re.UNICODE)
_SENTENCE_FINAL_RE = re.compile(r"""[.!?][)"'\]»”’]*$""")
_PARAGRAPH_START_RE = re.compile(r"""^[("'\[«“‘]*[A-ZÀ-ÖØ-Þ0-9]""")
_URL_OR_EMAIL_RE = re.compile(r"(https?://|www\.|\S+@\S+)", re.IGNORECASE)
_LAYOUT_ARTIFACT_LINE_RE = re.compile(
    r"(?i)^\s*(?:"
    r"end of document|load-date:|copyright|all rights reserved|"
    r"pdf-datei dieses dokuments|pdf file of this document|"
    r"graphic|bekijk de oorspronkelijke pagina:|"
    r"https?://\S+|www\.\S+"
    r")"
)

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
_ARTIFACT_PDF_DOC_RE = re.compile(
    r"(?im)^\s*(?:PDF-Datei dieses Dokuments|PDF file of this document).*$"
)
_ARTIFACT_LEXIS_URL_RE = re.compile(r"(?im)^\s*(?:https?://|www\.)\S+.*$")


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
        log(f"[load] Skipping Lexis document-list RTF files: {len(doclist_paths)}")
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


def load_workflow_country(config_path: Path, language: str) -> str:
    if not config_path.exists():
        return language
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    scope = country_scope_from_config(config, language)
    return scope.label or language


def resolve_workflow_language(config_path: Path, cli_language: str = "") -> str:
    if str(cli_language or "").strip():
        return normalize_language(cli_language)
    return normalize_language(load_workflow_language(config_path))


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
    log(f"[load] Reading {len(rtf_paths)} .rtf files")

    rows: List[Dict[str, object]] = []
    for fp in tqdm(rtf_paths, desc="Reading RTFs", disable=not VERBOSE):
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
    log(f"[load] Found {len(rtf_paths)} .rtf files under: {rtf_dir}")
    return load_rtf_articles_from_paths(rtf_paths, rtf_dir, month_translations, weekday_names)


def load_single_rtf_articles(
    rtf_file: Path,
    rtf_root: Path,
    month_translations: Dict[str, str],
    weekday_names: List[str],
) -> pd.DataFrame:
    if rtf_file.stem.lower().find("doclist") >= 0:
        log(f"[load] Skipping Lexis document-list RTF file: {rtf_file}")
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


def add_region_name(df: pd.DataFrame, mapping_csv: Path | None = None, default_region: str = "") -> pd.DataFrame:
    df = df.copy()
    df["region_name"] = default_region

    if mapping_csv is None or not mapping_csv.exists():
        if mapping_csv is not None:
            log(f"[map] Mapping CSV not found, skipping: {mapping_csv}")
        return df

    m = pd.read_csv(mapping_csv)
    if not {"newspaper", "region_name"}.issubset(set(m.columns)):
        raise ValueError(
            f"Mapping CSV missing required columns (newspaper, region_name): {mapping_csv}"
        )

    map_dict = dict(zip(m["newspaper"].map(normalize_key), m["region_name"].astype(str)))
    df["region_name"] = df["newspaper"].map(lambda x: map_dict.get(normalize_key(x), default_region))
    return df


def compute_geo_hits(df: pd.DataFrame, patterns: List[str]) -> pd.Series:
    # Combine all patterns into one regex — single pass per row
    combined = "|".join(f"(?:{p})" for p in patterns)
    geo_re = re.compile(combined, flags=re.IGNORECASE)
    return df["body"].fillna("").astype(str).map(lambda t: len(geo_re.findall(t)))


@dataclass
class CleanupStats:
    removed_bylines: int = 0
    removed_captions: int = 0
    removed_pull_quotes: int = 0
    removed_graphic_artifacts: int = 0
    removed_url_document_artifacts: int = 0
    removed_lines: int = 0

    def add(self, other: "CleanupStats") -> None:
        self.removed_bylines += other.removed_bylines
        self.removed_captions += other.removed_captions
        self.removed_pull_quotes += other.removed_pull_quotes
        self.removed_graphic_artifacts += other.removed_graphic_artifacts
        self.removed_url_document_artifacts += other.removed_url_document_artifacts
        self.removed_lines += other.removed_lines

    def as_dict(self) -> Dict[str, int]:
        return {
            "removed_bylines": self.removed_bylines,
            "removed_captions": self.removed_captions,
            "removed_pull_quotes": self.removed_pull_quotes,
            "removed_graphic_artifacts": self.removed_graphic_artifacts,
            "removed_url_document_artifacts": self.removed_url_document_artifacts,
            "removed_lines": self.removed_lines,
        }


BYLINE_START_PATTERNS = {
    "dutch": [r"door\b", r"van onze redactie\b", r"redactie\b"],
    "german": [r"von\b"],
    "italian": [r"di\b"],
    "english": [r"by\b"],
}
CAPTION_START_PATTERNS = [
    r"foto\b",
    r"photo\b",
    r"bild\b",
    r"afbeelding\b",
    r"caption\b",
    r"didascalia\b",
    r"immagine\b",
    r"grafiek\b",
    r"graphic\b",
    r"illustration\b",
]
GRAPHIC_ARTIFACT_RE = re.compile(
    r"(?i)^\s*(graphic|foto|photo|bild|afbeelding|didascalia|immagine)\s*:?\s*$"
)
DOCUMENT_ARTIFACT_RE = re.compile(
    r"(?i)^\s*(pdf-datei dieses dokuments|pdf file of this document|"
    r"bekijk de oorspronkelijke pagina:?|https?://\S+|www\.\S+)\s*.*$"
)
EMAIL_ONLY_RE = re.compile(r"^\s*[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\s*$")
QUOTE_EDGE_RE = re.compile(r"""^\s*["'“”‘’„»«].*["'“”‘’„»«]\s*$""")


def line_word_count(text: str) -> int:
    return len(_WORD_RE.findall(str(text or "")))


def attribution_patterns_for_language(language: str) -> List[str]:
    lang = normalize_language(language)
    patterns = list(BYLINE_START_PATTERNS.get(lang, []))
    patterns.extend(BYLINE_START_PATTERNS["english"])
    return list(dict.fromkeys(patterns))


def is_probable_byline_line(line: str, line_idx: int, language: str) -> bool:
    text = str(line or "").strip()
    if not text:
        return False
    if line_idx > 5:
        return False
    if EMAIL_ONLY_RE.match(text):
        return True
    if line_word_count(text) > 12:
        return False
    if _SENTENCE_FINAL_RE.search(text) and line_word_count(text) > 4:
        return False
    for pattern in attribution_patterns_for_language(language):
        match = re.match(rf"(?i)^({pattern})(?:\s|:)(.+)", text)
        if not match:
            continue
        rest = match.group(2).strip()
        pattern_text = pattern.replace(r"\b", "").lower()
        if pattern_text in {"redactie", "van onze redactie"}:
            return True
        if rest and (
            rest[0].isupper()
            or EMAIL_ONLY_RE.search(rest)
            or re.match(r"(?i)^(onze|redactie|ap|dpa|ansa)\b", rest)
        ):
            return True
    return False


def is_probable_caption_line(line: str, after_artifact_marker: bool = False) -> bool:
    text = str(line or "").strip()
    if not text:
        return False
    words = line_word_count(text)
    if words > 20:
        return False
    if any(re.match(rf"(?i)^{pattern}\s*:?", text) for pattern in CAPTION_START_PATTERNS):
        return True
    if after_artifact_marker and words <= 16 and not _SENTENCE_FINAL_RE.search(text):
        return True
    return False


def normalize_pull_quote_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[\"'“”‘’„»«]", "", text)
    text = re.sub(r"[^\wÀ-ÿ]+", " ", text)
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def is_duplicated_pull_quote_line(line: str, other_text: str) -> bool:
    text = str(line or "").strip()
    words = line_word_count(text)
    if words < 4 or words > 30:
        return False
    if not QUOTE_EDGE_RE.match(text):
        return False
    normalized = normalize_pull_quote_text(text)
    if len(normalized) < 20:
        return False
    return normalized in normalize_pull_quote_text(other_text)


def clean_body_before_paragraph_split(
    text: str,
    language: str,
) -> Tuple[str, CleanupStats]:
    lines = str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    stats = CleanupStats()
    kept: List[str] = []
    skip_next_caption = False

    for idx, line in enumerate(lines):
        stripped = _WHITESPACE_RE.sub(" ", line).strip()
        if not stripped:
            kept.append("")
            skip_next_caption = False
            continue

        if GRAPHIC_ARTIFACT_RE.match(stripped):
            stats.removed_graphic_artifacts += 1
            stats.removed_lines += 1
            skip_next_caption = True
            continue

        if DOCUMENT_ARTIFACT_RE.match(stripped):
            stats.removed_url_document_artifacts += 1
            stats.removed_lines += 1
            skip_next_caption = True
            continue

        if is_probable_caption_line(stripped, after_artifact_marker=skip_next_caption):
            stats.removed_captions += 1
            stats.removed_lines += 1
            skip_next_caption = False
            continue

        if is_probable_byline_line(stripped, idx, language):
            stats.removed_bylines += 1
            stats.removed_lines += 1
            skip_next_caption = False
            continue

        other_text = "\n".join(lines[:idx] + lines[idx + 1:])
        if is_duplicated_pull_quote_line(stripped, other_text):
            stats.removed_pull_quotes += 1
            stats.removed_lines += 1
            skip_next_caption = False
            continue

        kept.append(stripped)
        skip_next_caption = False

    cleaned = "\n".join(kept)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = _MULTI_NEWLINE_RE.sub("\n\n", cleaned)
    return cleaned.strip(), stats


def clean_articles(
    df: pd.DataFrame,
    month_translations: Dict[str, str],
    weekday_names: List[str],
    geothermal_patterns: List[str],
    summary_stats: Optional[Dict[str, int]] = None,
    language: str = "dutch",
) -> pd.DataFrame:
    df = df.copy()

    for col in ["title", "newspaper", "date", "body"]:
        if col not in df.columns:
            df[col] = ""

    for col in ["title", "newspaper", "section", "load_date", "body"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).map(strip_extraction_artifacts)

    cleanup_stats = CleanupStats()
    cleaned_bodies: List[str] = []
    for body in df["body"].fillna("").astype(str):
        cleaned_body, row_stats = clean_body_before_paragraph_split(body, language)
        cleanup_stats.add(row_stats)
        cleaned_bodies.append(cleaned_body)
    df["body"] = cleaned_bodies
    if cleanup_stats.removed_lines:
        log(
            "[clean] Body artifact cleanup: "
            + ", ".join(f"{key}={value}" for key, value in cleanup_stats.as_dict().items())
        )

    corrupt_mask = df.apply(looks_corrupt_article, axis=1)
    if corrupt_mask.any():
        log(f"[clean] Dropping likely corrupt article blocks: {int(corrupt_mask.sum())}")
        df = df.loc[~corrupt_mask].copy()

    df["date_translated"] = df["date"].fillna("").astype(str).map(
        lambda x: cleanup_date_text(x, month_translations, weekday_names)
    )
    df["date_parsed"] = pd.to_datetime(df["date_translated"], errors="coerce", dayfirst=True)

    df["body"] = df["body"].fillna("").astype(str)
    empty_mask = df["body"].str.strip().eq("")
    log(f"[clean] Empty-body rate: {empty_mask.mean():.1%} ({empty_mask.sum()} / {len(df)})")
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
    if summary_stats is not None:
        summary_stats["unique_documents"] = len(df)
    log(f"[clean] Deduplicated: {before} -> {len(df)} articles")

    df["source"] = df["newspaper"].astype(str)
    df["document_title"] = df["title"].astype(str)
    df["publish_date"] = df["date_parsed"].dt.date.astype(str).fillna("")

    return df


# =============================
# Paragraph splitting (content-based, hard-wrap aware)
# =============================

SentenceSplitFunc = Callable[[object, str], List[str]]


@dataclass
class LayoutBlock:
    text: str
    block_id: int
    boundary_reason: str
    original_text: str = ""
    hard_wrap_joins: int = 0


@dataclass
class ParagraphSplitRecord:
    text: str
    split_reason: str
    semantic_similarity_before: Optional[float] = None
    semantic_similarity_after: Optional[float] = None
    layout_block_id: Optional[int] = None
    layout_block_word_count: Optional[int] = None
    original_layout_block_text: str = ""


@dataclass
class ParagraphSplitStats:
    rtf_or_layout_boundaries: int = 0
    single_newline_layout_boundaries: int = 0
    hard_wrap_joins: int = 0
    layout_boundaries: int = 0
    preserved_layout_blocks: int = 0
    semantic_topic_shifts: int = 0
    max_length_constraints: int = 0
    fallback_length_grouping: int = 0
    heading_attachments: int = 0
    short_merges: int = 0
    embedding_unavailable: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "rtf_or_layout_boundaries": self.rtf_or_layout_boundaries,
            "single_newline_layout_boundaries": self.single_newline_layout_boundaries,
            "hard_wrap_joins": self.hard_wrap_joins,
            "layout_boundaries": self.layout_boundaries,
            "preserved_layout_blocks": self.preserved_layout_blocks,
            "semantic_topic_shifts": self.semantic_topic_shifts,
            "max_length_constraints": self.max_length_constraints,
            "fallback_length_grouping": self.fallback_length_grouping,
            "heading_attachments": self.heading_attachments,
            "short_merges": self.short_merges,
            "embedding_unavailable": self.embedding_unavailable,
        }


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(str(text or "")))


def preview_text(text: str, max_chars: int = LAYOUT_DEBUG_PREVIEW_CHARS) -> str:
    cleaned = str(text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + " ..."


def is_artifact_layout_line(line: str) -> bool:
    text = str(line or "").strip()
    if not text:
        return False
    if _LAYOUT_ARTIFACT_LINE_RE.search(text):
        return True
    if _CORRUPT_HEX_RE.search(text):
        return True
    return False


def is_continuation_fragment(line: str) -> bool:
    text = str(line or "").strip()
    if not text:
        return True
    if text[0].islower():
        return True
    if text.startswith((",", ";", ":", ")", "]")):
        return True
    if _URL_OR_EMAIL_RE.search(text):
        return True
    return False


def is_likely_layout_paragraph_break(prev_line: str, next_line: str) -> bool:
    prev = str(prev_line or "").strip()
    nxt = str(next_line or "").strip()
    if not prev or not nxt:
        return True
    if _URL_OR_EMAIL_RE.search(prev) or _URL_OR_EMAIL_RE.search(nxt):
        return False
    if is_artifact_layout_line(prev) or is_artifact_layout_line(nxt):
        return True
    if prev.endswith("-"):
        return False
    if is_continuation_fragment(nxt):
        return False
    if is_probable_heading(prev):
        return True
    if _SENTENCE_FINAL_RE.search(prev) and _PARAGRAPH_START_RE.search(nxt):
        return True
    return False


def make_layout_block(
    lines: List[str],
    block_id: int,
    boundary_reason: str,
    hard_wrap_joins: int = 0,
) -> Optional[LayoutBlock]:
    cleaned_lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in lines]
    cleaned_lines = [line for line in cleaned_lines if line and not is_artifact_layout_line(line)]
    if not cleaned_lines:
        return None

    merged = cleaned_lines[0]
    joins = hard_wrap_joins
    for line in cleaned_lines[1:]:
        if merged.endswith("-") and len(merged) > 1:
            merged = merged[:-1] + line
        else:
            merged = f"{merged} {line}".strip()
        joins += 1

    return LayoutBlock(
        text=merged.strip(),
        block_id=block_id,
        boundary_reason=boundary_reason,
        original_text="\n".join(cleaned_lines),
        hard_wrap_joins=joins,
    )


def normalize_layout_blocks_preserving_paragraphs(text: str) -> List[LayoutBlock]:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return []

    raw = re.sub(r"(\w)-[ \t]*\n[ \t]*(\w)", r"\1\2", raw)
    lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in raw.split("\n")]
    blocks: List[LayoutBlock] = []
    current: List[str] = []
    current_reason = "layout_boundary"
    current_hard_wrap_joins = 0

    def flush(reason_for_current: Optional[str] = None) -> None:
        nonlocal current, current_reason, current_hard_wrap_joins
        if not current:
            return
        block = make_layout_block(
            current,
            len(blocks) + 1,
            reason_for_current or current_reason,
            current_hard_wrap_joins,
        )
        if block is not None:
            blocks.append(block)
        current = []
        current_reason = "layout_boundary"
        current_hard_wrap_joins = 0

    for idx, line in enumerate(lines):
        if not line:
            flush("layout_boundary")
            current_reason = "layout_boundary"
            continue
        if is_artifact_layout_line(line):
            flush("layout_boundary")
            current_reason = "layout_boundary"
            continue

        if not current:
            current = [line]
            continue

        if (
            PRESERVE_SINGLE_NEWLINE_PARAGRAPHS
            and is_likely_layout_paragraph_break(current[-1], line)
        ):
            flush(current_reason)
            current = [line]
            current_reason = "single_newline_layout_boundary"
            continue

        current.append(line)
        current_hard_wrap_joins += 1

    flush(current_reason)

    for idx, block in enumerate(blocks, start=1):
        block.block_id = idx
    return blocks


def normalize_layout_text(text: str) -> str:
    """
    Merge single-newline layout wraps while preserving blank lines as paragraph hints.
    """
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""

    text = re.sub(r"(\w)-[ \t]*\n[ \t]*(\w)", r"\1\2", text)
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


class TransformersParagraphEmbeddingModel:
    def __init__(self, model_name: str, max_length: int = 256, batch_size: int = 32) -> None:
        from transformers import AutoModel, AutoTokenizer
        import torch

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode(
        self,
        texts: List[str],
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> List[object]:
        del show_progress_bar
        if isinstance(texts, str):
            texts = [texts]

        embeddings: List[object] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [str(text or "") for text in texts[start:start + self.batch_size]]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with self.torch.no_grad():
                model_output = self.model(**encoded)

            token_embeddings = model_output.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(
                token_embeddings.size()
            ).float()
            pooled = (token_embeddings * attention_mask).sum(dim=1)
            pooled = pooled / attention_mask.sum(dim=1).clamp(min=1e-9)

            if convert_to_numpy:
                embeddings.extend(list(pooled.cpu().numpy()))
            else:
                embeddings.extend(pooled.cpu().tolist())

        return embeddings


def build_paragraph_embedding_model(
    enabled: bool = ENABLE_CONTENT_AWARE_PARAGRAPHS,
    model_name: str = CONTENT_AWARE_MODEL,
    backend: str = PARAGRAPH_EMBEDDING_BACKEND,
) -> object:
    if not enabled:
        log("[main] Content-aware paragraph splitting disabled; using sentence/length fallback.")
        return None

    backend = str(backend or "").strip().lower()

    if backend in {"transformers", "auto"}:
        try:
            model = TransformersParagraphEmbeddingModel(model_name)
            log(f"[main] Loaded paragraph embedding model once via transformers: {model_name}")
            return model
        except Exception as exc:
            log(
                "[main] WARNING: Could not load paragraph embedding model via transformers "
                f"{model_name!r}; using sentence/length paragraph fallback: {exc!r}"
            )
            return None

    if backend in {"sentence-transformers", "sentence_transformers"}:
        log(
            "[main] WARNING: sentence-transformers backend is disabled by default in this "
            "environment because importing it can trigger duplicate OpenMP runtime aborts. "
            "Using sentence/length paragraph fallback."
        )
        return None

    log(
        "[main] WARNING: Unknown paragraph embedding backend "
        f"{backend!r}; using sentence/length paragraph fallback."
    )
    return None


def fallback_split_sentences(text: str) -> List[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return []
    abbreviation_pattern = (
        r"\b(?:dhr|dr|prof|mr|mevr|mw|ir|ing|st|nr|ca|vgl|bijv|etc|"
        r"z\.B|bzw|bspw|d\.h|u\.a|e\.g|i\.e|vs)\."
    )
    protected = re.sub(
        abbreviation_pattern,
        lambda match: match.group(0).replace(".", "<DOT>"),
        cleaned,
        flags=re.IGNORECASE,
    )
    sentences = re.split(
        r"(?<=[.!?])(?:[\"')\]]+)?\s+(?=[A-ZÀ-ÖØ-Ý0-9])",
        protected,
    )
    sentences = [sentence.replace("<DOT>", ".").strip() for sentence in sentences if sentence.strip()]
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences or [cleaned]


def build_paragraph_sentence_segmenter(
    language: str,
) -> Tuple[object, Optional[SentenceSplitFunc]]:
    try:
        from split_paragraphs_to_sentences import build_segmenter, split_sentences

        return build_segmenter(language), split_sentences
    except Exception as exc:
        log(
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


def join_reasons(*reasons: str) -> str:
    ordered: List[str] = []
    for reason in reasons:
        for part in str(reason or "").split("|"):
            part = part.strip()
            if part and part not in ordered:
                ordered.append(part)
    return "|".join(ordered) if ordered else "unknown"


def combine_paragraph_records(
    left: ParagraphSplitRecord,
    right: ParagraphSplitRecord,
) -> ParagraphSplitRecord:
    return ParagraphSplitRecord(
        text=f"{left.text} {right.text}".strip(),
        split_reason=join_reasons(left.split_reason, right.split_reason, "merged_short"),
        semantic_similarity_before=left.semantic_similarity_before,
        semantic_similarity_after=right.semantic_similarity_after,
        layout_block_id=left.layout_block_id,
        layout_block_word_count=left.layout_block_word_count,
        original_layout_block_text=left.original_layout_block_text or right.original_layout_block_text,
    )


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


def make_micro_segments(
    sentences: List[str],
    window_size: int = MICRO_SEGMENT_SENTENCES,
) -> List[List[str]]:
    window_size = max(1, int(window_size))
    return [
        sentences[i:i + window_size]
        for i in range(0, len(sentences), window_size)
        if sentences[i:i + window_size]
    ]


def cosine_similarity(left: object, right: object) -> float:
    if hasattr(left, "tolist"):
        left = left.tolist()
    if hasattr(right, "tolist"):
        right = right.tolist()
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    numerator = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = sum(a * a for a in left_values) ** 0.5
    right_norm = sum(b * b for b in right_values) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def encode_paragraph_segments(embedding_model: object, segment_texts: List[str]) -> List[object]:
    try:
        return list(
            embedding_model.encode(
                segment_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        )
    except TypeError:
        return list(embedding_model.encode(segment_texts))


def split_by_content_shifts(
    sentences: List[str],
    embedding_model: object,
    base_reason: str = "layout_boundary",
    threshold: float = CONTENT_SHIFT_THRESHOLD,
    micro_segment_sentences: int = MICRO_SEGMENT_SENTENCES,
    min_words: int = MIN_WORDS_PER_PARAGRAPH,
) -> List[ParagraphSplitRecord]:
    micro_segments = make_micro_segments(sentences, micro_segment_sentences)
    if len(micro_segments) <= 1:
        return [ParagraphSplitRecord(" ".join(sentences).strip(), base_reason)]

    segment_texts = [" ".join(segment).strip() for segment in micro_segments]
    embeddings = encode_paragraph_segments(embedding_model, segment_texts)
    similarities = [
        cosine_similarity(embeddings[i - 1], embeddings[i])
        for i in range(1, len(embeddings))
    ]

    records: List[ParagraphSplitRecord] = []
    current: List[str] = []
    current_similarity_before: Optional[float] = None

    for idx, segment in enumerate(micro_segments):
        similarity_before = similarities[idx - 1] if idx > 0 else None
        current_words = word_count(" ".join(current))
        remaining_words = word_count(
            " ".join(sentence for future in micro_segments[idx:] for sentence in future)
        )
        should_split = (
            idx > 0
            and similarity_before is not None
            and similarity_before < threshold
            and current_words >= min_words
            and remaining_words >= max(1, min_words // 2)
        )

        if should_split:
            records.append(
                ParagraphSplitRecord(
                    text=" ".join(current).strip(),
                    split_reason=base_reason if not records else "semantic_topic_shift",
                    semantic_similarity_before=current_similarity_before,
                    semantic_similarity_after=similarity_before,
                )
            )
            current = []
            current_similarity_before = similarity_before

        current.extend(segment)

    if current:
        records.append(
            ParagraphSplitRecord(
                text=" ".join(current).strip(),
                split_reason="semantic_topic_shift" if records else base_reason,
                semantic_similarity_before=current_similarity_before,
                semantic_similarity_after=None,
            )
        )

    return records


def group_sentences_into_paragraph_records(
    sentences: List[str],
    reason: str,
    stats: Optional[ParagraphSplitStats] = None,
) -> List[ParagraphSplitRecord]:
    paragraphs = group_sentences_into_paragraphs(sentences)
    if stats is not None and len(paragraphs) > 1:
        stats.fallback_length_grouping += len(paragraphs) - 1
    return [
        ParagraphSplitRecord(text=paragraph, split_reason=reason)
        for paragraph in paragraphs
        if paragraph
    ]


def add_layout_metadata(
    records: List[ParagraphSplitRecord],
    layout_block: LayoutBlock,
) -> List[ParagraphSplitRecord]:
    for record in records:
        record.layout_block_id = layout_block.block_id
        record.layout_block_word_count = word_count(layout_block.text)
        record.original_layout_block_text = preview_text(layout_block.original_text)
    return records


def should_preserve_short_layout_record(record: ParagraphSplitRecord, total_records: int) -> bool:
    words = word_count(record.text)
    if total_records <= 1:
        return True
    if "heading_attached" in record.split_reason:
        return True
    if words >= MIN_WORDS_FOR_STANDALONE_LAYOUT_PARAGRAPH and (
        "layout_boundary" in record.split_reason
        or "single_newline_layout_boundary" in record.split_reason
    ):
        return True
    return False


def split_oversized_or_shifted_block(
    block: object,
    language: str,
    nlp: object = None,
    split_func: Optional[SentenceSplitFunc] = None,
    embedding_model: object = None,
    stats: Optional[ParagraphSplitStats] = None,
) -> List[ParagraphSplitRecord]:
    if isinstance(block, LayoutBlock):
        layout_block = block
    else:
        layout_block = LayoutBlock(
            text=str(block or "").strip(),
            block_id=0,
            boundary_reason="layout_boundary",
            original_text=str(block or "").strip(),
        )

    block_text = layout_block.text.strip()
    base_reason = layout_block.boundary_reason or "layout_boundary"
    if not block_text:
        return []

    sentences = split_sentences_multilingual(block_text, language, nlp, split_func)
    sentence_count = len(sentences) if sentences else estimate_sentence_count(block_text)
    block_words = word_count(block_text)

    should_preserve = (
        block_words < SEMANTIC_SPLIT_MIN_WORDS
        and (
            block_words <= MAX_WORDS_PER_CONTENT_PARAGRAPH
            or not ENABLE_SEMANTIC_SPLIT_FOR_NORMAL_BLOCKS
        )
    )
    if should_preserve and sentence_count <= MAX_SENTENCES_PER_PARAGRAPH:
        if stats is not None:
            stats.preserved_layout_blocks += 1
        return add_layout_metadata(
            [ParagraphSplitRecord(block_text, base_reason)],
            layout_block,
        )

    if stats is not None:
        stats.max_length_constraints += 1

    if not sentences:
        return add_layout_metadata([ParagraphSplitRecord(block_text, base_reason)], layout_block)

    if embedding_model is None:
        return add_layout_metadata(
            group_sentences_into_paragraph_records(
                sentences,
                join_reasons(base_reason, "fallback_length_grouping"),
                stats,
            ),
            layout_block,
        )

    try:
        shifted_records = split_by_content_shifts(sentences, embedding_model, base_reason=base_reason)
    except Exception as exc:
        if stats is not None:
            stats.embedding_unavailable += 1
        log(
            "[main] WARNING: Paragraph embedding split failed for one block; "
            f"using sentence/length fallback: {exc!r}"
        )
        return add_layout_metadata(
            group_sentences_into_paragraph_records(
                sentences,
                join_reasons(base_reason, "fallback_length_grouping"),
                stats,
            ),
            layout_block,
        )

    if stats is not None:
        stats.semantic_topic_shifts += max(0, len(shifted_records) - 1)

    constrained: List[ParagraphSplitRecord] = []
    for record in shifted_records:
        record_sentences = split_sentences_multilingual(record.text, language, nlp, split_func)
        if (
            word_count(record.text) <= MAX_WORDS_PER_CONTENT_PARAGRAPH
            and len(record_sentences) <= MAX_SENTENCES_PER_PARAGRAPH
        ):
            constrained.append(record)
            continue

        if stats is not None:
            stats.max_length_constraints += 1
        for paragraph in group_sentences_into_paragraphs(record_sentences):
            constrained.append(
                ParagraphSplitRecord(
                    text=paragraph,
                    split_reason=join_reasons(record.split_reason, "max_length_constraint"),
                    semantic_similarity_before=record.semantic_similarity_before,
                    semantic_similarity_after=record.semantic_similarity_after,
                    layout_block_id=record.layout_block_id,
                    layout_block_word_count=record.layout_block_word_count,
                    original_layout_block_text=record.original_layout_block_text,
                )
            )

    return add_layout_metadata(constrained, layout_block)


def apply_paragraph_size_constraints(
    records: List[ParagraphSplitRecord],
    language: str,
    nlp: object = None,
    split_func: Optional[SentenceSplitFunc] = None,
    stats: Optional[ParagraphSplitStats] = None,
) -> List[ParagraphSplitRecord]:
    records = [record for record in records if record.text.strip()]
    if len(records) <= 1:
        return records

    merged: List[ParagraphSplitRecord] = []
    i = 0
    while i < len(records):
        record = records[i]
        if (
            word_count(record.text) >= MIN_WORDS_PER_PARAGRAPH
            or should_preserve_short_layout_record(record, len(records))
        ):
            merged.append(record)
            i += 1
            continue

        if i + 1 < len(records):
            candidate = combine_paragraph_records(record, records[i + 1])
            candidate_sentences = split_sentences_multilingual(
                candidate.text,
                language,
                nlp,
                split_func,
            )
            if (
                word_count(candidate.text) <= MAX_WORDS_PER_CONTENT_PARAGRAPH
                and len(candidate_sentences) <= MAX_SENTENCES_PER_PARAGRAPH
            ):
                records[i + 1] = candidate
                if stats is not None:
                    stats.short_merges += 1
                i += 1
                continue

        if merged:
            candidate = combine_paragraph_records(merged[-1], record)
            candidate_sentences = split_sentences_multilingual(
                candidate.text,
                language,
                nlp,
                split_func,
            )
            if (
                word_count(candidate.text) <= MAX_WORDS_PER_CONTENT_PARAGRAPH
                and len(candidate_sentences) <= MAX_SENTENCES_PER_PARAGRAPH
            ):
                merged[-1] = candidate
                if stats is not None:
                    stats.short_merges += 1
                i += 1
                continue

        merged.append(record)
        i += 1

    constrained: List[ParagraphSplitRecord] = []
    for record in merged:
        sentences = split_sentences_multilingual(record.text, language, nlp, split_func)
        if (
            word_count(record.text) <= MAX_WORDS_PER_CONTENT_PARAGRAPH
            and len(sentences) <= MAX_SENTENCES_PER_PARAGRAPH
        ):
            constrained.append(record)
            continue

        if stats is not None:
            stats.max_length_constraints += 1
        for paragraph in group_sentences_into_paragraphs(sentences):
            constrained.append(
                ParagraphSplitRecord(
                    text=paragraph,
                    split_reason=join_reasons(record.split_reason, "max_length_constraint"),
                    semantic_similarity_before=record.semantic_similarity_before,
                    semantic_similarity_after=record.semantic_similarity_after,
                    layout_block_id=record.layout_block_id,
                    layout_block_word_count=record.layout_block_word_count,
                    original_layout_block_text=record.original_layout_block_text,
                )
            )

    return constrained


def split_paragraphs_with_metadata(
    text: str,
    language: str,
    nlp: object = None,
    split_func: Optional[SentenceSplitFunc] = None,
    embedding_model: object = None,
    stats: Optional[ParagraphSplitStats] = None,
) -> List[ParagraphSplitRecord]:
    layout_blocks = normalize_layout_blocks_preserving_paragraphs(text)
    if not layout_blocks:
        return []

    output: List[ParagraphSplitRecord] = []
    heading_carry = ""
    heading_block: Optional[LayoutBlock] = None

    for layout_block in layout_blocks:
        block_text = layout_block.text.strip()
        if not block_text:
            continue

        if stats is not None:
            stats.layout_boundaries += 1
            if layout_block.boundary_reason == "single_newline_layout_boundary":
                stats.single_newline_layout_boundaries += 1
            else:
                stats.rtf_or_layout_boundaries += 1
            stats.hard_wrap_joins += layout_block.hard_wrap_joins

        if (
            word_count(block_text) >= MIN_WORDS_FOR_ATTACHED_HEADING
            and is_probable_heading(block_text)
        ):
            heading_carry = f"{heading_carry} {block_text}".strip()
            heading_block = layout_block
            continue

        paragraphs = split_oversized_or_shifted_block(
            layout_block,
            language,
            nlp,
            split_func,
            embedding_model,
            stats,
        )
        if heading_carry and paragraphs:
            paragraphs[0] = ParagraphSplitRecord(
                text=f"{heading_carry} {paragraphs[0].text}".strip(),
                split_reason=join_reasons("heading_attached", paragraphs[0].split_reason),
                semantic_similarity_before=paragraphs[0].semantic_similarity_before,
                semantic_similarity_after=paragraphs[0].semantic_similarity_after,
                layout_block_id=paragraphs[0].layout_block_id,
                layout_block_word_count=paragraphs[0].layout_block_word_count,
                original_layout_block_text=preview_text(
                    "\n".join(
                        part
                        for part in [
                            heading_block.original_text if heading_block is not None else "",
                            paragraphs[0].original_layout_block_text,
                        ]
                        if part
                    )
                ),
            )
            if stats is not None:
                stats.heading_attachments += 1
            heading_carry = ""
            heading_block = None
        output.extend(paragraphs)

    if heading_carry:
        if output:
            last_with_heading = f"{output[-1].text} {heading_carry}".strip()
            sentence_count = len(
                split_sentences_multilingual(last_with_heading, language, nlp, split_func)
            )
            if sentence_count <= MAX_SENTENCES_PER_PARAGRAPH and word_count(last_with_heading) <= MAX_WORDS_PER_PARAGRAPH:
                output[-1] = ParagraphSplitRecord(
                    text=last_with_heading,
                    split_reason=join_reasons(output[-1].split_reason, "heading_attached"),
                    semantic_similarity_before=output[-1].semantic_similarity_before,
                    semantic_similarity_after=output[-1].semantic_similarity_after,
                    layout_block_id=output[-1].layout_block_id,
                    layout_block_word_count=output[-1].layout_block_word_count,
                    original_layout_block_text=preview_text(
                        "\n".join(
                            part
                            for part in [
                                output[-1].original_layout_block_text,
                                heading_block.original_text if heading_block is not None else "",
                            ]
                            if part
                        )
                    ),
                )
                if stats is not None:
                    stats.heading_attachments += 1
        else:
            output.append(
                ParagraphSplitRecord(
                    heading_carry,
                    "heading_only",
                    layout_block_id=heading_block.block_id if heading_block is not None else None,
                    layout_block_word_count=(
                        word_count(heading_block.text) if heading_block is not None else None
                    ),
                    original_layout_block_text=(
                        preview_text(heading_block.original_text)
                        if heading_block is not None
                        else heading_carry
                    ),
                )
            )

    return apply_paragraph_size_constraints(output, language, nlp, split_func, stats)


def split_paragraphs_content_based(
    text: str,
    language: str,
    nlp: object = None,
    split_func: Optional[SentenceSplitFunc] = None,
    embedding_model: object = None,
    stats: Optional[ParagraphSplitStats] = None,
) -> List[str]:
    records = split_paragraphs_with_metadata(
        text,
        language,
        nlp,
        split_func,
        embedding_model,
        stats,
    )
    return [record.text for record in records if record.text.strip()]


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
        log(
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
        log("[main] Paragraphs per document: " + format_numeric_stats(paragraphs_per_doc))
        log(f"[workflow_table] paragraphs_per_document_mean: {paragraphs_per_doc.mean():.2f}")
        log(f"[workflow_table] paragraphs_per_document_median: {paragraphs_per_doc.median():.2f}")
        log(f"[workflow_table] paragraphs_per_document_p95: {paragraphs_per_doc.quantile(0.95):.2f}")
        log(f"[workflow_table] paragraphs_per_document_max: {int(paragraphs_per_doc.max())}")

        outlier_threshold = max(20, paragraphs_per_doc.quantile(0.99))
        outliers = paragraphs_per_doc[paragraphs_per_doc >= outlier_threshold].sort_values(ascending=False)
        log(
            f"[main] Documents with many paragraphs: "
            f"{len(outliers)} documents with >= {outlier_threshold:.0f} paragraphs"
        )
        log(f"[workflow_table] paragraph_heavy_documents_threshold: {outlier_threshold:.0f}")
        log(f"[workflow_table] paragraph_heavy_documents_count: {len(outliers)}")

        if not outliers.empty:
            meta_cols = [c for c in ["source", "document_title", "publish_date"] if c in df_paras.columns]
            doc_meta = df_paras.drop_duplicates(doc_key).set_index(doc_key)
            log("[main] Top documents by paragraph count:")
            for body_hash, n_paragraphs in outliers.head(10).items():
                meta = doc_meta.loc[body_hash, meta_cols].to_dict() if meta_cols else {}
                title = str(meta.get("document_title", ""))[:90]
                source = str(meta.get("source", ""))
                date = str(meta.get("publish_date", ""))
                log(
                    f"  - paragraphs={int(n_paragraphs)} | "
                    f"source={source} | date={date} | title={title}"
                )

    paragraph_text = df_paras["paragraph_text"].fillna("").astype(str)
    paragraph_word_counts = paragraph_text.map(lambda value: len(value.split()))
    paragraph_sentence_counts = sentence_counts_for_paragraphs(paragraph_text, language)

    log("[main] Sentences per paragraph: " + format_numeric_stats(paragraph_sentence_counts))
    log("[main] Words per paragraph: " + format_numeric_stats(paragraph_word_counts))
    log(f"[workflow_table] sentences_per_paragraph_mean: {paragraph_sentence_counts.mean():.2f}")
    log(f"[workflow_table] sentences_per_paragraph_median: {paragraph_sentence_counts.median():.2f}")
    log(f"[workflow_table] sentences_per_paragraph_p95: {paragraph_sentence_counts.quantile(0.95):.2f}")
    log(f"[workflow_table] sentences_per_paragraph_max: {int(paragraph_sentence_counts.max())}")
    log(f"[workflow_table] words_per_paragraph_mean: {paragraph_word_counts.mean():.2f}")
    log(f"[workflow_table] words_per_paragraph_median: {paragraph_word_counts.median():.2f}")
    log(f"[workflow_table] words_per_paragraph_p95: {paragraph_word_counts.quantile(0.95):.2f}")
    log(f"[workflow_table] words_per_paragraph_max: {int(paragraph_word_counts.max())}")


def make_paragraph_uid(
    source: str, title: str, publish_date: str, paragraph_id: int, paragraph_text: str
) -> str:
    base = "||".join([source, title, publish_date, str(paragraph_id), paragraph_text])
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def write_paragraph_split_sample(
    df_paras: pd.DataFrame,
    sample_size: int,
    output_csv: Path,
    output_md: Path,
    random_seed: int = PARAGRAPH_SAMPLE_RANDOM_SEED,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    if df_paras.empty:
        pd.DataFrame().to_csv(output_csv, index=False, encoding="utf-8")
        output_md.write_text("# Paragraph Split Sample\n\nNo paragraphs generated.\n", encoding="utf-8")
        log(f"[main] Wrote empty paragraph split sample: {output_csv}")
        log(f"[main] Wrote empty paragraph split sample Markdown: {output_md}")
        return

    doc_key = "body_hash" if "body_hash" in df_paras.columns else "uid"
    unique_docs = pd.Series(df_paras[doc_key].dropna().astype(str).unique()).sort_values()
    n_docs = min(max(0, int(sample_size)), len(unique_docs))
    if n_docs == 0:
        sample_docs = unique_docs.iloc[0:0]
    else:
        sample_docs = unique_docs.sample(n=n_docs, random_state=random_seed)

    sample = df_paras[df_paras[doc_key].astype(str).isin(set(sample_docs))].copy()
    sort_cols = [
        col
        for col in ["source", "publish_date", "document_title", doc_key, "paragraph_id"]
        if col in sample.columns
    ]
    if sort_cols:
        sample = sample.sort_values(sort_cols)

    sample["previous_paragraph_text"] = sample.groupby(doc_key)["paragraph_text"].shift(1).fillna("")
    sample["next_paragraph_text"] = sample.groupby(doc_key)["paragraph_text"].shift(-1).fillna("")

    wanted_cols = [
        "source",
        "document_title",
        "publish_date",
        "body_hash",
        "paragraph_id",
        "paragraph_text",
        "layout_block_id",
        "layout_block_word_count",
        "original_layout_block_text",
        "paragraph_word_count",
        "paragraph_sentence_count",
        "split_reason",
        "previous_paragraph_text",
        "next_paragraph_text",
        "semantic_similarity_before",
        "semantic_similarity_after",
    ]
    sample_cols = [col for col in wanted_cols if col in sample.columns]
    sample[sample_cols].to_csv(output_csv, index=False, encoding="utf-8")

    lines: List[str] = ["# Paragraph Split Sample", ""]
    for _, article in sample.groupby(doc_key, sort=False):
        first = article.iloc[0]
        lines.extend([
            f"TITLE: {first.get('document_title', '')}",
            f"SOURCE: {first.get('source', '')}",
            f"DATE: {first.get('publish_date', '')}",
            f"ARTICLE HASH: {first.get('body_hash', first.get(doc_key, ''))}",
            "",
        ])

        for _, row in article.sort_values("paragraph_id").iterrows():
            lines.append(
                "[Paragraph "
                f"{row.get('paragraph_id', '')} | "
                f"block={row.get('layout_block_id', '')} | "
                f"words={row.get('paragraph_word_count', '')} | "
                f"sentences={row.get('paragraph_sentence_count', '')} | "
                f"reason={row.get('split_reason', '')}]"
            )
            lines.append(str(row.get("paragraph_text", "")).strip())
            lines.append("")
        lines.append("")

    output_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    log(f"[main] Wrote paragraph split sample CSV: {output_csv} (rows={len(sample)})")
    log(f"[main] Wrote paragraph split sample Markdown: {output_md}")


def visible_newlines(text: str, max_chars: int = LAYOUT_DEBUG_PREVIEW_CHARS) -> str:
    return preview_text(text, max_chars=max_chars).replace("\n", "\\n\n")


def write_layout_debug_sample(
    df_articles: pd.DataFrame,
    sample_size: int,
    output_md: Path,
    random_seed: int = PARAGRAPH_SAMPLE_RANDOM_SEED,
) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)

    if df_articles.empty:
        output_md.write_text("# Layout Debug Sample\n\nNo articles available.\n", encoding="utf-8")
        log(f"[main] Wrote empty layout debug sample Markdown: {output_md}")
        return

    doc_key = "body_hash" if "body_hash" in df_articles.columns else None
    if doc_key is not None:
        unique_docs = pd.Series(df_articles[doc_key].dropna().astype(str).unique()).sort_values()
        n_docs = min(max(0, int(sample_size)), len(unique_docs))
        sample_docs = (
            unique_docs.iloc[0:0]
            if n_docs == 0
            else unique_docs.sample(n=n_docs, random_state=random_seed)
        )
        sample = df_articles[df_articles[doc_key].astype(str).isin(set(sample_docs))].copy()
    else:
        n_docs = min(max(0, int(sample_size)), len(df_articles))
        sample = df_articles.sample(n=n_docs, random_state=random_seed) if n_docs else df_articles.iloc[0:0]

    sort_cols = [
        col
        for col in ["source", "publish_date", "document_title", "body_hash"]
        if col in sample.columns
    ]
    if sort_cols:
        sample = sample.sort_values(sort_cols)

    lines: List[str] = ["# Layout Debug Sample", ""]
    for _, row in sample.iterrows():
        body = str(row.get("body", "") or "")
        blocks = normalize_layout_blocks_preserving_paragraphs(body)
        lines.extend([
            f"TITLE: {row.get('document_title', row.get('title', ''))}",
            f"SOURCE: {row.get('source', row.get('newspaper', ''))}",
            f"DATE: {row.get('publish_date', row.get('date', ''))}",
            f"BODY HASH: {row.get('body_hash', '')}",
            "",
            "[Raw/extracted body preview with visible newline markers]",
            visible_newlines(body),
            "",
        ])

        for block in blocks:
            lines.append(
                "[Detected layout block "
                f"{block.block_id} | "
                f"words={word_count(block.text)} | "
                f"reason={block.boundary_reason} | "
                f"hard_wrap_joins={block.hard_wrap_joins}]"
            )
            lines.append(visible_newlines(block.original_text))
            lines.append("")
        lines.append("")

    output_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    log(f"[main] Wrote layout debug sample Markdown: {output_md}")


# =============================
# CLI
# =============================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--language", type=str, default="")
    ap.add_argument("--project-dir", type=str, default=str(PROJECT_DIR))
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--input-rtf-dir", type=str, default=str(INPUT_RTF_DIR))
    ap.add_argument("--input-rtf-file", type=str, default="")
    ap.add_argument("--input-raw-articles-csv", type=str, nargs="+", default=[])
    ap.add_argument("--output-raw-articles-csv", type=str, default=str(OUTPUT_RAW_ARTICLES_CSV))
    ap.add_argument("--raw-only", action="store_true")
    ap.add_argument("--newspaper-region-csv", type=str, default="")
    ap.add_argument("--output-paragraph-csv", type=str, default=str(OUTPUT_PARAGRAPH_CSV))
    ap.add_argument("--output-articles-csv", type=str, default=str(OUTPUT_ARTICLES_CSV))
    ap.add_argument("--write-articles-csv", action="store_true", default=WRITE_ARTICLES_CSV)
    ap.add_argument("--no-write-articles-csv", action="store_false", dest="write_articles_csv")
    ap.add_argument(
        "--enable-content-aware-paragraphs",
        action="store_true",
        default=ENABLE_CONTENT_AWARE_PARAGRAPHS,
    )
    ap.add_argument(
        "--disable-content-aware-paragraphs",
        action="store_false",
        dest="enable_content_aware_paragraphs",
    )
    ap.add_argument("--content-aware-model", type=str, default=CONTENT_AWARE_MODEL)
    ap.add_argument(
        "--paragraph-embedding-backend",
        type=str,
        choices=["transformers", "sentence-transformers"],
        default=PARAGRAPH_EMBEDDING_BACKEND,
    )
    ap.add_argument("--write-paragraph-sample", action="store_true")
    ap.add_argument("--paragraph-sample-only", action="store_true")
    ap.add_argument("--paragraph-sample-size", type=int, default=50)
    ap.add_argument("--debug-layout-sample", action="store_true")
    ap.add_argument("--debug-layout-sample-size", type=int, default=10)
    ap.add_argument(
        "--output-paragraph-sample-csv",
        type=str,
        default=str(OUTPUT_PARAGRAPH_SAMPLE_CSV),
    )
    ap.add_argument(
        "--output-paragraph-sample-md",
        type=str,
        default=str(OUTPUT_PARAGRAPH_SAMPLE_MD),
    )
    ap.add_argument(
        "--output-layout-debug-md",
        type=str,
        default=str(OUTPUT_LAYOUT_DEBUG_MD),
    )
    return ap.parse_args()


def print_summary(language: str, summary_stats: Dict[str, int]) -> None:
    language = str(language or "unknown")
    print(f'{language} loaded documents: "{summary_stats.get("loaded_documents", 0)}"')
    print(f'{language} unique documents: "{summary_stats.get("unique_documents", 0)}"')
    print(
        f'{language} documents after cleaning: '
        f'"{summary_stats.get("documents_after_cleaning", 0)}"'
    )
    print(f'{language} paragraphs: "{summary_stats.get("paragraphs", 0)}"')


# =============================
# Main
# =============================

def main() -> None:
    global VERBOSE
    args = parse_args()
    VERBOSE = bool(args.verbose)
    if not VERBOSE:
        warnings.filterwarnings("ignore")

    project_dir = Path(args.project_dir).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    input_rtf_dir = Path(args.input_rtf_dir).expanduser().resolve()
    input_rtf_file = Path(args.input_rtf_file).expanduser().resolve() if args.input_rtf_file else None
    input_raw_article_csvs = [
        Path(path).expanduser().resolve() for path in args.input_raw_articles_csv
    ]
    newspaper_region_csv = Path(args.newspaper_region_csv).expanduser().resolve() if args.newspaper_region_csv else None
    output_paragraph_csv = Path(args.output_paragraph_csv).expanduser().resolve()
    output_articles_csv = Path(args.output_articles_csv).expanduser().resolve()
    output_raw_articles_csv = Path(args.output_raw_articles_csv).expanduser().resolve()
    output_paragraph_sample_csv = Path(args.output_paragraph_sample_csv).expanduser().resolve()
    output_paragraph_sample_md = Path(args.output_paragraph_sample_md).expanduser().resolve()
    output_layout_debug_md = Path(args.output_layout_debug_md).expanduser().resolve()

    os.chdir(project_dir)
    output_paragraph_csv.parent.mkdir(parents=True, exist_ok=True)

    workflow_language = resolve_workflow_language(config_path, args.language)
    workflow_country = load_workflow_country(config_path, workflow_language)
    month_translations, weekday_names = load_preprocessing_config(
        config_path,
        project_dir,
        workflow_language,
    )
    geothermal_patterns = load_geothermal_patterns(project_dir, workflow_language)
    paragraph_nlp, paragraph_split_func = build_paragraph_sentence_segmenter(workflow_language)
    summary_stats: Dict[str, int] = {
        "loaded_documents": 0,
        "unique_documents": 0,
        "documents_after_cleaning": 0,
        "paragraphs": 0,
    }

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
        log(f"[main] Loaded raw article chunks: {len(input_raw_article_csvs)} CSV files")
    elif input_rtf_file is not None:
        df_raw = load_single_rtf_articles(
            input_rtf_file,
            input_rtf_dir,
            month_translations,
            weekday_names,
        )
    else:
        df_raw = load_all_rtf_articles(input_rtf_dir, month_translations, weekday_names)
    summary_stats["loaded_documents"] = len(df_raw)
    log(f"[main] Loaded {len(df_raw)} raw article blocks")
    log(f"[workflow_table] raw_article_blocks: {len(df_raw)}")

    if args.raw_only:
        output_raw_articles_csv.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_csv(output_raw_articles_csv, index=False, encoding="utf-8")
        log(f"[main] Wrote raw articles: {output_raw_articles_csv} (rows={len(df_raw)})")
        return

    if df_raw.empty:
        log("[main] No articles found. Check INPUT_RTF_DIR and that files end with .rtf/.RTF")
        output_paragraph_csv.write_text("", encoding="utf-8")
        print_summary(workflow_language, summary_stats)
        return

    raw_empty_body = df_raw["body"].fillna("").astype(str).str.strip().eq("").mean()
    log(f"[main] Raw empty-body rate: {raw_empty_body:.1%}")

    # 2) Clean
    df = clean_articles(
        df_raw,
        month_translations,
        weekday_names,
        geothermal_patterns,
        summary_stats=summary_stats,
        language=workflow_language,
    )
    summary_stats["documents_after_cleaning"] = len(df)
    if not summary_stats["unique_documents"]:
        summary_stats["unique_documents"] = len(df)
    log(f"[main] Articles after cleaning: {len(df)}")
    log(f"[workflow_table] cleaned_articles: {len(df)}")

    # 3) Region mapping
    df = add_region_name(df, newspaper_region_csv, default_region=workflow_country)

    # 4) Optionally write articles CSV
    if args.write_articles_csv:
        output_articles_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_articles_csv, index=False, encoding="utf-8")
        log(f"[main] Wrote cleaned articles: {output_articles_csv}")

    if args.debug_layout_sample:
        write_layout_debug_sample(
            df,
            args.debug_layout_sample_size,
            output_layout_debug_md,
        )

    # 5) Build paragraph rows
    paragraph_rows: List[Dict[str, object]] = []
    paragraph_split_stats = ParagraphSplitStats()
    paragraph_embedding_model = build_paragraph_embedding_model(
        enabled=args.enable_content_aware_paragraphs,
        model_name=args.content_aware_model,
        backend=args.paragraph_embedding_backend,
    )
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Splitting paragraphs", disable=not VERBOSE):
        paragraph_records = split_paragraphs_with_metadata(
            str(r.get("body", "")),
            workflow_language,
            paragraph_nlp,
            paragraph_split_func,
            paragraph_embedding_model,
            paragraph_split_stats,
        )
        if not paragraph_records:
            continue

        source = str(r.get("source", ""))
        title = str(r.get("document_title", ""))
        publish_date = str(r.get("publish_date", ""))

        for i, record in enumerate(paragraph_records, start=1):
            paragraph_text = record.text
            paragraph_rows.append({
                **r.to_dict(),
                "paragraph_id": i,
                "paragraph_text": paragraph_text,
                "paragraph_word_count": word_count(paragraph_text),
                "paragraph_sentence_count": len(
                    split_sentences_multilingual(
                        paragraph_text,
                        workflow_language,
                        paragraph_nlp,
                        paragraph_split_func,
                    )
                ),
                "split_reason": record.split_reason,
                "layout_block_id": record.layout_block_id,
                "layout_block_word_count": record.layout_block_word_count,
                "original_layout_block_text": record.original_layout_block_text,
                "semantic_similarity_before": record.semantic_similarity_before,
                "semantic_similarity_after": record.semantic_similarity_after,
                "uid": make_paragraph_uid(source, title, publish_date, i, paragraph_text),
            })

    df_paras = pd.DataFrame(paragraph_rows)
    summary_stats["paragraphs"] = len(df_paras)
    log(f"[main] Paragraph rows: {len(df_paras)}")
    log(f"[workflow_table] paragraphs_after_split_filter: {len(df_paras)}")
    log("[main] Paragraph split reason counters:")
    for key, value in paragraph_split_stats.as_dict().items():
        log(f"  - {key}: {value}")
        log(f"[workflow_table] paragraph_split_{key}: {value}")
    if VERBOSE and not df_paras.empty and "split_reason" in df_paras.columns:
        log("[main] Paragraph split reasons:")
        for reason, count in df_paras["split_reason"].value_counts().items():
            log(f"  - {reason}: {int(count)}")
    if not df_paras.empty and "body_hash" in df_paras.columns:
        paragraphs_per_article = df_paras.groupby("body_hash").size()
        articles_with_multiple_paragraphs = int((paragraphs_per_article > 1).sum())
        log(
            "[main] Paragraphs per article: "
            f"mean={paragraphs_per_article.mean():.2f}, "
            f"max={int(paragraphs_per_article.max())}, "
            f"multi_paragraph_articles={articles_with_multiple_paragraphs}/{len(paragraphs_per_article)}"
        )
        log(f"[workflow_table] articles_with_paragraphs: {len(paragraphs_per_article)}")
        log(f"[workflow_table] articles_with_multiple_paragraphs: {articles_with_multiple_paragraphs}")
        log(f"[workflow_table] mean_paragraphs_per_article: {paragraphs_per_article.mean():.2f}")

    if VERBOSE:
        print_paragraph_split_stats(df_paras, workflow_language)

    if df_paras.empty:
        log(
            "[main] WARNING: No paragraphs generated. "
            "Body extraction may have failed or everything was filtered out."
        )

    if args.write_paragraph_sample or args.paragraph_sample_only:
        write_paragraph_split_sample(
            df_paras,
            args.paragraph_sample_size,
            output_paragraph_sample_csv,
            output_paragraph_sample_md,
        )

    if args.paragraph_sample_only:
        log("[main] Paragraph sample-only mode: skipping full paragraph CSV write.")
        print_summary(workflow_language, summary_stats)
        return

    df_paras.to_csv(output_paragraph_csv, index=False, encoding="utf-8")
    log(f"[main] Wrote paragraphs CSV: {output_paragraph_csv} (rows={len(df_paras)})")
    print_summary(workflow_language, summary_stats)


if __name__ == "__main__":
    main()
