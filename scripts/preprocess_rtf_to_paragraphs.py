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
from typing import Dict, List, Tuple
import argparse

import pandas as pd
import yaml
from striprtf.striprtf import rtf_to_text as _striprtf_to_text
from tqdm.auto import tqdm


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
WRITE_ARTICLES_CSV = True

MIN_WORDS_PER_ARTICLE = 100
FILTER_BY_GEO_HITS = False
MIN_GEO_HITS = 3
GEOTHERMAL_PATTERNS = [r"aardwarmte\w*", r"geotherm\w*"]

MIN_WORDS_PER_PARAGRAPH = 20


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
_DATE_RE = re.compile(r"(?i)\b(\d{1,2}\s+[A-Za-zÀ-ÿ]+\s+\d{4})\b")
_HEADER_STOP_RE = re.compile(
    r"(?i)^(copyright|section:|length:|byline:|highlight:|body\b|load-date:)"
)
_DATE_LINE_RE = re.compile(r"(?i)^\s*\d{1,2}\s+[A-Za-zÀ-ÿ]+\s+\d{4}(?:\s+[A-Za-zÀ-ÿ])?\s*$")
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


# =============================
# Article extraction
# =============================

DEFAULT_MONTH_TRANSLATIONS = {
    "januari": "January", "februari": "February", "maart": "March",
    "april": "April", "mei": "May", "juni": "June", "juli": "July",
    "augustus": "August", "september": "September", "oktober": "October",
    "november": "November", "december": "December",
    "gennaio": "January", "febbraio": "February", "marzo": "March",
    "aprile": "April", "maggio": "May", "giugno": "June", "luglio": "July",
    "agosto": "August", "settembre": "September", "ottobre": "October",
    "novembre": "November", "dicembre": "December",
}

DEFAULT_WEEKDAY_NAMES = {
    "maandag", "dinsdag", "woensdag", "donderdag", "vrijdag", "zaterdag", "zondag",
    "lunedì", "lunedi", "martedì", "martedi", "mercoledì", "mercoledi", "giovedì",
    "giovedi", "venerdì", "venerdi", "sabato", "domenica",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}


def normalize_article_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).replace("\ufeff", "").replace("\u00a0", " ")
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


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

    return month_translations, list(dict.fromkeys(weekday_names))


def translate_month_names(s: str, month_translations: Dict[str, str]) -> str:
    out = str(s)
    for source_name, english_name in month_translations.items():
        out = re.sub(rf"\b{re.escape(source_name)}\b", english_name, out, flags=re.IGNORECASE)
    return out


def cleanup_date_text(s: str, month_translations: Dict[str, str], weekday_names: List[str]) -> str:
    text = translate_month_names(s, month_translations)
    weekday_pattern = r"\b(" + "|".join(sorted(re.escape(d) for d in weekday_names)) + r")\b"
    text = re.sub(weekday_pattern, "", text, flags=re.IGNORECASE)
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


def load_all_rtf_articles(
    rtf_dir: Path,
    month_translations: Dict[str, str],
    weekday_names: List[str],
) -> pd.DataFrame:
    rtf_paths = sorted(rtf_dir.rglob("*.RTF"))
    print(f"[load] Found {len(rtf_paths)} .rtf files under: {rtf_dir}")

    rows: List[Dict[str, object]] = []
    for fp in tqdm(rtf_paths, desc="Reading RTFs"):
        rtf_content = read_rtf_file(fp)
        plain_text = normalize_article_text(rtf_to_text(rtf_content))
        articles = [a for a in _END_DOC_SPLIT_RE.split(plain_text) if a and a.strip()]

        for art in articles:
            meta = parse_header_metadata(art, month_translations, weekday_names)
            rows.append({
                "source_file": fp.name,
                "source_path": str(fp),
                **meta,
            })

    return pd.DataFrame(rows)


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
    return _MULTI_SPACE_RE.sub(" ", text).strip()


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

    df["geo_hits"] = compute_geo_hits(df, GEOTHERMAL_PATTERNS)
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
# Paragraph splitting (hard-wrap aware)
# =============================

def dewrap_hardwrap_lines(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    out_lines: List[str] = []
    buf: List[str] = []

    def flush_buf() -> None:
        if buf:
            out_lines.append(" ".join(buf).strip())
            buf.clear()

    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            flush_buf()
            out_lines.append("")
            continue
        if buf and buf[-1].endswith("-") and len(buf[-1]) > 1:
            buf[-1] = buf[-1][:-1] + line
        else:
            buf.append(line)

    flush_buf()
    return _MULTI_NEWLINE_RE.sub("\n\n", "\n".join(out_lines)).strip()


def split_paragraphs(text: str) -> List[str]:
    text = dewrap_hardwrap_lines(text)
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p and p.strip()]
    return [p for p in paras if len(p.split()) >= MIN_WORDS_PER_PARAGRAPH]


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
    newspaper_region_csv = Path(args.newspaper_region_csv).expanduser().resolve()
    output_paragraph_csv = Path(args.output_paragraph_csv).expanduser().resolve()
    output_articles_csv = Path(args.output_articles_csv).expanduser().resolve()

    os.chdir(project_dir)
    output_paragraph_csv.parent.mkdir(parents=True, exist_ok=True)

    month_translations, weekday_names = load_preprocessing_config(config_path)

    if not input_rtf_dir.exists():
        raise FileNotFoundError(f"INPUT_RTF_DIR does not exist: {input_rtf_dir}")

    # 1) Load raw articles
    df_raw = load_all_rtf_articles(input_rtf_dir, month_translations, weekday_names)
    print(f"[main] Loaded {len(df_raw)} raw article blocks")

    if df_raw.empty:
        print("[main] No articles found. Check INPUT_RTF_DIR and that files end with .RTF")
        output_paragraph_csv.write_text("", encoding="utf-8")
        return

    raw_empty_body = df_raw["body"].fillna("").astype(str).str.strip().eq("").mean()
    print(f"[main] Raw empty-body rate: {raw_empty_body:.1%}")

    # 2) Clean
    df = clean_articles(df_raw, month_translations, weekday_names)
    print(f"[main] Articles after cleaning: {len(df)}")

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
        paras = split_paragraphs(str(r.get("body", "")))
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

    if df_paras.empty:
        print(
            "[main] WARNING: No paragraphs generated. "
            "Body extraction may have failed or everything was filtered out."
        )

    df_paras.to_csv(output_paragraph_csv, index=False, encoding="utf-8")
    print(f"[main] Wrote paragraphs CSV: {output_paragraph_csv} (rows={len(df_paras)})")


if __name__ == "__main__":
    main()
