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
import yaml
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
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "config.yaml"

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
        return rtf_to_text_fallback(rtf_content)


def rtf_to_text_fallback(rtf_content: str) -> str:
    """Best-effort RTF reader that strips formatting groups and keeps visible text."""
    destinations = {
        "aftncn", "aftnsep", "aftnsepc", "annotation", "author", "background",
        "bkmkend", "bkmkstart", "blipuid", "buptim", "category", "colorschememapping",
        "colortbl", "comment", "company", "creatim", "datafield", "datastore",
        "defchp", "defpap", "do", "doccomm", "docvar", "dptxbxtext", "ebcend",
        "ebcstart", "factoidname", "falt", "fchars", "ffdeftext", "ffentrymcr",
        "ffexitmcr", "ffformat", "ffhelptext", "ffl", "ffname", "ffstattext",
        "file", "filetbl", "fldinst", "fldtype", "fname", "fontemb", "fontfile",
        "fonttbl", "footer", "footerf", "footerl", "footerr", "footnote", "formfield",
        "ftncn", "ftnsep", "ftnsepc", "g", "generator", "gridtbl", "header", "headerf",
        "headerl", "headerr", "hl", "hlfr", "hlinkbase", "htmltag", "info", "keycode",
        "keywords", "latentstyles", "lchars", "levelnumbers", "leveltext", "list",
        "listlevel", "listname", "listoverride", "listoverridetable", "listpicture",
        "liststylename", "listtable", "manager", "margPr", "mhtmltag", "mmath",
        "moMath", "moMathPara", "objalias", "objclass", "objdata", "object", "objname",
        "objsect", "objtime", "oldcprops", "oldpprops", "oldsprops", "oldtprops",
        "oleclsid", "operator", "panose", "password", "passwordhash", "pgp", "pgptbl",
        "picprop", "pict", "pn", "pnseclvl", "pntext", "propname", "protend", "protstart",
        "protusertbl", "pxe", "result", "revtbl", "rsidtbl", "rxe", "shp", "shpgrp",
        "shpinst", "shppict", "shprslt", "shptxt", "sn", "sp", "staticval", "stylesheet",
        "subject", "sv", "themedata", "title", "txe", "ud", "upr", "userprops",
        "wgrffmtfilter", "xmlattrname", "xmlattrvalue", "xmlclose", "xmlname",
        "xmlnstbl", "xmlopen",
    }
    specialchars = {
        "par": "\n",
        "line": "\n",
        "tab": "\t",
        "emdash": "\u2014",
        "endash": "\u2013",
        "emspace": "\u2003",
        "enspace": "\u2002",
        "qmspace": "\u2005",
        "bullet": "\u2022",
        "lquote": "\u2018",
        "rquote": "\u2019",
        "ldblquote": "\u201c",
        "rdblquote": "\u201d",
    }

    stack: List[Tuple[int, bool]] = []
    ignorable = False
    ucskip = 1
    curskip = 0
    out: List[str] = []
    i = 0
    text = rtf_content

    while i < len(text):
        ch = text[i]

        if curskip > 0:
            curskip -= 1
            i += 1
            continue

        if ch == "{":
            stack.append((ucskip, ignorable))
            i += 1
            continue

        if ch == "}":
            if stack:
                ucskip, ignorable = stack.pop()
            i += 1
            continue

        if ch == "\\":
            i += 1
            if i >= len(text):
                break

            ch = text[i]
            if ch in "\\{}":
                if not ignorable:
                    out.append(ch)
                i += 1
                continue

            if ch == "*":
                ignorable = True
                i += 1
                continue

            if ch == "'":
                if i + 2 < len(text):
                    hexcode = text[i + 1:i + 3]
                    try:
                        decoded = bytes.fromhex(hexcode).decode("cp1252")
                    except Exception:
                        decoded = ""
                    if not ignorable:
                        out.append(decoded)
                    i += 3
                    continue

            match = re.match(r"([a-zA-Z]+)(-?\d+)? ?", text[i:])
            if match:
                word = match.group(1)
                arg = match.group(2)
                i += len(match.group(0))

                if word in destinations:
                    ignorable = True
                elif word == "uc":
                    ucskip = int(arg) if arg else 1
                elif word == "u":
                    if arg and not ignorable:
                        codepoint = int(arg)
                        if codepoint < 0:
                            codepoint += 65536
                        out.append(chr(codepoint))
                    curskip = ucskip
                elif word in specialchars and not ignorable:
                    out.append(specialchars[word])
                continue

            if ch in "~_-":
                if not ignorable:
                    out.append(" " if ch == "~" else "\n" if ch == "_" else "-")
                i += 1
                continue

            i += 1
            continue

        if not ignorable:
            out.append(ch)
        i += 1

    txt = "".join(out)
    txt = re.sub(r"[^\S\n]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
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

DEFAULT_MONTH_TRANSLATIONS = {
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
    "gennaio": "January",
    "febbraio": "February",
    "marzo": "March",
    "aprile": "April",
    "maggio": "May",
    "giugno": "June",
    "luglio": "July",
    "agosto": "August",
    "settembre": "September",
    "ottobre": "October",
    "novembre": "November",
    "dicembre": "December",
}

DEFAULT_WEEKDAY_NAMES = {
    "maandag", "dinsdag", "woensdag", "donderdag", "vrijdag", "zaterdag", "zondag",
    "lunedì", "lunedi", "martedì", "martedi", "mercoledì", "mercoledi", "giovedì",
    "giovedi", "venerdì", "venerdi", "sabato", "domenica",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}

_END_DOC_SPLIT_RE = re.compile(r"(?is)\bEnd of Document\b")

# More tolerant than the strict blank-line version:
_BODY_RE = re.compile(
    r"(?is)\bBody\b\s*:?\s*(.*?)(?:\n\s*Load-Date:|\n\s*Copyright|\bEnd of Document\b|$)"
)

_LEN_RE = re.compile(r"(?is)\bLength:\s*(\d+)")
_SECTION_RE = re.compile(r"(?is)\bSection:\s*(.*?);")
_LOAD_DATE_RE = re.compile(r"(?is)\bLoad-Date:\s*(.*?)(?:\n|$)")
_DATE_RE = re.compile(r"(?i)\b(\d{1,2}\s+[A-Za-zÀ-ÿ]+\s+\d{4})\b")  # supports Dutch months too
_HEADER_STOP_RE = re.compile(
    r"(?i)^(copyright|section:|length:|byline:|highlight:|body\b|load-date:)"
)
_DATE_LINE_RE = re.compile(r"(?i)^\s*\d{1,2}\s+[A-Za-zÀ-ÿ]+\s+\d{4}(?:\s+[A-Za-zÀ-ÿ]+)?\s*$")


def normalize_article_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).replace("\ufeff", "")
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def article_lines(article: str) -> List[str]:
    return [ln.strip() for ln in normalize_article_text(article).split("\n") if ln.strip()]


def load_preprocessing_config(config_path: Path) -> Tuple[Dict[str, str], List[str]]:
    month_translations = dict(DEFAULT_MONTH_TRANSLATIONS)
    weekday_names = list(DEFAULT_WEEKDAY_NAMES)

    if not config_path.exists():
        return month_translations, weekday_names

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    preprocessing = config.get("preprocessing", {}) or {}
    date_locale = preprocessing.get("date_locale", {}) or {}

    extra_months = date_locale.get("month_translations", {}) or {}
    if isinstance(extra_months, dict):
        month_translations.update({str(k): str(v) for k, v in extra_months.items()})

    extra_weekdays = date_locale.get("weekday_names", []) or []
    if isinstance(extra_weekdays, list):
        weekday_names.extend(str(day) for day in extra_weekdays)

    weekday_names = list(dict.fromkeys(weekday_names))
    return month_translations, weekday_names


def translate_month_names(s: str, month_translations: Dict[str, str]) -> str:
    out = str(s)
    for source_name, english_name in month_translations.items():
        out = re.sub(rf"\b{re.escape(source_name)}\b", english_name, out, flags=re.IGNORECASE)
    return out


def cleanup_date_text(s: str, month_translations: Dict[str, str], weekday_names: List[str]) -> str:
    text = translate_month_names(s, month_translations)
    weekday_pattern = r"\b(" + "|".join(sorted(re.escape(day) for day in weekday_names)) + r")\b"
    text = re.sub(weekday_pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_header_body(article: str) -> Tuple[List[str], str]:
    normalized = normalize_article_text(article)
    lines = [ln.strip() for ln in normalized.split("\n")]

    body_idx = None
    for i, line in enumerate(lines):
        if re.match(r"(?i)^body\b", line.strip()):
            body_idx = i
            break

    if body_idx is None:
        return [ln for ln in lines if ln.strip()], extract_body(normalized)

    header_lines = [ln.strip() for ln in lines[:body_idx] if ln.strip()]
    body_lines: List[str] = []
    for line in lines[body_idx + 1:]:
        stripped = line.strip()
        if re.match(r"(?i)^load-date:", stripped):
            break
        body_lines.append(line)

    body = "\n".join(body_lines).strip()
    return header_lines, body


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

    section = (_SECTION_RE.search(article).group(1).strip() if _SECTION_RE.search(article) else "")
    length_m = _LEN_RE.search(article)
    word_count = int(length_m.group(1)) if length_m else None
    load_date = (_LOAD_DATE_RE.search(article).group(1).strip() if _LOAD_DATE_RE.search(article) else "")

    return {
        "title": title,
        "newspaper": newspaper,
        "date": date_str,
        "section": section,
        "word_count": word_count,
        "load_date": load_date,
        "body": body if body else extract_body(article),
    }


def extract_title(article: str) -> str:
    return str(parse_header_metadata(article, DEFAULT_MONTH_TRANSLATIONS, list(DEFAULT_WEEKDAY_NAMES)).get("title", ""))


def extract_date_str(article: str) -> str:
    return str(parse_header_metadata(article, DEFAULT_MONTH_TRANSLATIONS, list(DEFAULT_WEEKDAY_NAMES)).get("date", ""))


def extract_newspaper(article: str, date_str: str) -> str:
    return str(parse_header_metadata(article, DEFAULT_MONTH_TRANSLATIONS, list(DEFAULT_WEEKDAY_NAMES)).get("newspaper", ""))


def extract_body(article: str) -> str:
    article = normalize_article_text(article)
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


def load_all_rtf_articles(
    rtf_dir: Path,
    month_translations: Dict[str, str],
    weekday_names: List[str],
) -> pd.DataFrame:
    rtf_paths = sorted(rtf_dir.rglob("*.RTF"))
    print(f"[load] Found {len(rtf_paths)} .rtf files under: {rtf_dir}")

    rows: List[Dict[str, object]] = []
    for fp in tqdm(rtf_paths, desc="Reading RTFs"):
        rtf_content = read_text_with_fallbacks(fp)
        plain_text = normalize_article_text(rtf_to_text(rtf_content))

        # Split into Lexis-like article blocks
        articles = [a for a in _END_DOC_SPLIT_RE.split(plain_text) if a and a.strip()]

        for art in articles:
            meta = parse_header_metadata(art, month_translations, weekday_names)

            rows.append(
                {
                    "source_file": fp.name,
                    "source_path": str(fp),
                    "title": meta["title"],
                    "newspaper": meta["newspaper"],
                    "date": meta["date"],
                    "section": meta["section"],
                    "word_count": meta["word_count"],
                    "load_date": meta["load_date"],
                    "body": meta["body"],
                }
            )

    df = pd.DataFrame(rows)
    return df


# =============================
# Cleaning + mapping
# =============================

def normalize_key(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def strip_extraction_artifacts(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"(?m)^\s*-\d+\s*", " ", text)
    text = re.sub(r"\b-?\d+\s+Page of\s+-?\d+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b-?\d+\b(?=\s+-?\d+\b)", " ", text)
    text = re.sub(r"\b[0-9A-Fa-f]{32,}\b", " ", text)
    text = re.sub(r"\\'[0-9a-fA-F]{2}", " ", text)
    text = re.sub(r"\\\*", " ", text)
    text = re.sub(r"\bBekijk de oorspronkelijke pagina:.*$", " ", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"\bGraphic\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def looks_corrupt_article(row: pd.Series) -> bool:
    title = str(row.get("title", "") or "")
    newspaper = str(row.get("newspaper", "") or "")
    body = str(row.get("body", "") or "")

    if not newspaper.strip():
        return True
    if not title.strip():
        return True
    if title.startswith("Times New Roman"):
        return True
    if "Page of" in title:
        return True
    if len(re.findall(r"\b[0-9A-Fa-f]{16,}\b", title + " " + body)) > 2:
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


def clean_articles(
    df: pd.DataFrame,
    month_translations: Dict[str, str],
    weekday_names: List[str],
) -> pd.DataFrame:
    df = df.copy()

    # Basic sanity: ensure columns exist
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

    # Parse dates via structure-driven extraction + month-name normalization
    df["date_translated"] = df["date"].fillna("").astype(str).map(
        lambda x: cleanup_date_text(x, month_translations, weekday_names)
    )
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
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
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
    config_path = Path(args.config).expanduser().resolve()
    input_rtf_dir = Path(args.input_rtf_dir).expanduser().resolve()
    newspaper_region_csv = Path(args.newspaper_region_csv).expanduser().resolve()
    output_paragraph_csv = Path(args.output_paragraph_csv).expanduser().resolve()
    output_articles_csv = Path(args.output_articles_csv).expanduser().resolve()
    output_dir = output_paragraph_csv.parent

    os.chdir(project_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    month_translations, weekday_names = load_preprocessing_config(config_path)

    if not input_rtf_dir.exists():
        raise FileNotFoundError(f"INPUT_RTF_DIR does not exist: {input_rtf_dir}")

    # 1) Load raw articles
    df_raw = load_all_rtf_articles(input_rtf_dir, month_translations, weekday_names)
    print(f"[main] Loaded {len(df_raw)} raw article blocks (after splitting)")

    if df_raw.empty:
        print("[main] No articles found. Check INPUT_RTF_DIR and whether files end with .RTF.")
        output_paragraph_csv.write_text("", encoding="utf-8")
        return

    # Quick diagnostics (very helpful when things go wrong)
    raw_empty_body = float(df_raw["body"].fillna("").astype(str).str.strip().eq("").mean())
    print(f"[main] Raw empty-body rate: {raw_empty_body:.1%}")

    # 2) Clean
    df = clean_articles(df_raw, month_translations, weekday_names)
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
