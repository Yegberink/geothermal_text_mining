#!/usr/bin/env python3
"""
geothermal_classifier_ollama.py
Resumable paragraph -> geothermal relevance classification via Ollama.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence

import pandas as pd
import requests
import yaml
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm.auto import tqdm

from helpers.country_scope import country_scope_from_args
from helpers.language_resources import normalize_language, vocab_dir

SYSTEM = (
    "You are a careful text classification assistant. "
    "You decide whether a newspaper paragraph is mainly about geothermal energy."
)

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[2]
ROW_UID_COL = "_row_uid"
GEOTHERMAL_LEXICON_CATEGORIES = ("strong", "contextual", "competing")


def load_geothermal_lexicon(project_dir: Path, language: str | None) -> dict[str, list[str]]:
    lang = normalize_language(language)
    path = vocab_dir(project_dir, lang) / "geo_keywords.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing geothermal keyword file: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, Mapping):
        raise ValueError(f"Malformed geothermal vocabulary {path}: document must be a mapping.")

    for category in GEOTHERMAL_LEXICON_CATEGORIES:
        if category not in data:
            raise ValueError(
                f"Malformed geothermal vocabulary {path}: missing category {category!r}."
            )

    for category in data:
        if category not in GEOTHERMAL_LEXICON_CATEGORIES:
            raise ValueError(
                f"Malformed geothermal vocabulary {path}: invalid category {category!r}."
            )

    lexicon: dict[str, list[str]] = {}
    for category in GEOTHERMAL_LEXICON_CATEGORIES:
        values = data[category]
        if not isinstance(values, list):
            raise ValueError(
                f"Malformed geothermal vocabulary {path}: category {category!r} must be a list."
            )

        cleaned: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not isinstance(value, str):
                raise ValueError(
                    f"Malformed geothermal vocabulary {path}: "
                    f"category {category!r} contains a non-string entry."
                )
            term = value.strip()
            if not term or term in seen:
                continue
            seen.add(term)
            cleaned.append(term)
        lexicon[category] = cleaned

    return lexicon


def _format_terms(terms: Sequence[str]) -> str:
    return ", ".join(str(term).strip() for term in terms if str(term).strip()) or "(none)"


def _lexicon_fingerprint_text(geothermal_lexicon: Mapping[str, Sequence[str]]) -> str:
    payload = {
        category: list(geothermal_lexicon.get(category, ()))
        for category in GEOTHERMAL_LEXICON_CATEGORIES
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _sanitize_evidence(value: object) -> str:
    evidence = "" if value is None else str(value).strip()
    words = evidence.split()
    if len(words) > 20:
        evidence = " ".join(words[:20])
    elif words:
        evidence = " ".join(words)
    return evidence or "No usable evidence."


def _fingerprint(
    text: str,
    country: str,
    language: str,
    geothermal_lexicon: Mapping[str, Sequence[str]],
) -> str:
    h = hashlib.sha256()
    h.update((country or "").encode("utf-8"))
    h.update(b"\n")
    h.update((language or "").encode("utf-8"))
    h.update(b"\n")
    h.update(_lexicon_fingerprint_text(geothermal_lexicon).encode("utf-8"))
    h.update(b"\n")
    h.update((text or "").encode("utf-8"))
    return h.hexdigest()


def make_uid(row: Dict[str, object]) -> str:
    base = "||".join([
        str(row.get("source", "")),
        str(row.get("document_title", "")),
        str(row.get("publish_date", "")),
        str(row.get("paragraph_id", "")),
        str(row.get("paragraph_text", "")),
    ])
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def set_unique_row_index(df: pd.DataFrame, uid_col: str = "uid") -> pd.DataFrame:
    """Keep content uid as data, but use a unique key for pandas alignment."""
    out = df.copy()
    if uid_col not in out.columns:
        if out.index.name == uid_col:
            out[uid_col] = out.index
        else:
            out[uid_col] = out.apply(lambda r: make_uid(r.to_dict()), axis=1)
    if out.index.name == uid_col:
        out = out.reset_index(drop=True)

    out[uid_col] = out[uid_col].astype(str)
    duplicate_mask = out[uid_col].duplicated(keep=False)
    occurrence = out.groupby(uid_col, sort=False).cumcount().astype(str)
    out[ROW_UID_COL] = out[uid_col]
    out.loc[duplicate_mask, ROW_UID_COL] = (
        out.loc[duplicate_mask, uid_col] + "__dup" + occurrence.loc[duplicate_mask]
    )
    return out.set_index(ROW_UID_COL, drop=True)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(
        (requests.Timeout, requests.ConnectionError, requests.HTTPError)
    ),
)
def llm_is_geothermal(
    text: str,
    country: str,
    ollama_url: str,
    model: str,
    language: str,
    geothermal_lexicon: Mapping[str, Sequence[str]],
) -> dict:
    strong_terms = _format_terms(geothermal_lexicon.get("strong", ()))
    contextual_terms = _format_terms(geothermal_lexicon.get("contextual", ()))
    competing_terms = _format_terms(geothermal_lexicon.get("competing", ()))

    prompt = f"""
Task: Decide whether this paragraph is MAINLY about geothermal energy.

Country: {country}
Language: {language}

Language-specific terms:
- Strong geothermal: {strong_terms}
- Contextual: {contextual_terms}
- Competing topics: {competing_terms}

Labels:
- YES: geothermal energy is clearly the main or substantial subject.
- NO: geothermal is absent, incidental, or another subject clearly dominates.
- MAYBE: geothermal is plausible or relevant, but it is unclear whether it is the main subject.

Rules:
- Judge the overall meaning, not keyword count.
- Strong terms are evidence, not an automatic YES.
- Contextual terms count only with clear underground-heat or geothermal-system context.
- Competing terms suggest alternative topics but do not automatically mean NO.
- A passing mention or inclusion in a list of renewables is NO.
- Prefer MAYBE over YES when geothermal is not clearly central.
- Return valid JSON only.
- Keys: is_geothermal, confidence, evidence_short.
- is_geothermal must be one of: YES, NO, MAYBE.
- confidence is a number from 0 to 1.
- evidence_short must be at most 20 words.

Paragraph:
{text}
""".strip()

    payload = {
        "model": model,
        "system": SYSTEM,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 120},
    }

    r = requests.post(ollama_url, json=payload, timeout=120)
    r.raise_for_status()
    out = (r.json().get("response") or "").strip()

    start = out.find("{")
    end = out.rfind("}")
    if start == -1 or end == -1:
        return {
            "is_geothermal": "MAYBE",
            "confidence": 0.0,
            "evidence_short": "No JSON returned.",
        }

    try:
        obj = json.loads(out[start:end + 1])
    except json.JSONDecodeError:
        return {
            "is_geothermal": "MAYBE",
            "confidence": 0.0,
            "evidence_short": "Invalid JSON.",
        }
    if not isinstance(obj, Mapping):
        return {
            "is_geothermal": "MAYBE",
            "confidence": 0.0,
            "evidence_short": "Invalid JSON.",
        }

    label = str(obj.get("is_geothermal") or "MAYBE").strip().upper()
    if label not in ("YES", "NO", "MAYBE"):
        label = "MAYBE"

    conf = obj.get("confidence", 0.0) or 0.0
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    ev = _sanitize_evidence(obj.get("evidence_short"))
    return {"is_geothermal": label, "confidence": conf, "evidence_short": ev}


def incomplete_count(out: pd.DataFrame) -> int:
    if "llm_status" not in out.columns:
        return len(out)
    return int((~out["llm_status"].isin(["ok", "empty"])).sum())


def restart_error_rows(out: pd.DataFrame) -> int:
    if "llm_status" not in out.columns:
        return 0
    error_mask = out["llm_status"].eq("error")
    restarted = int(error_mask.sum())
    if restarted:
        reset_values = {
            "llm_is_geothermal": None,
            "llm_geo_confidence": 0.0,
            "llm_geo_evidence_short": None,
            "llm_status": None,
            "llm_error": None,
        }
        for col, value in reset_values.items():
            out.loc[error_mask, col] = value
    return restarted


def write_final_csv_atomic(out: pd.DataFrame, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_csv.with_suffix(out_csv.suffix + ".tmp")
    out.to_csv(tmp_path, index=False, encoding="utf-8")
    tmp_path.replace(out_csv)


def batch_geothermal_resumable(
    df: pd.DataFrame,
    text_col: str,
    cache_path: Path,
    sleep_s: float,
    ollama_url: str,
    model: str,
    country: str,
    language: str,
    geothermal_lexicon: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    out = df.copy()
    out["llm_is_geothermal"] = None
    out["llm_geo_confidence"] = 0.0
    out["llm_geo_evidence_short"] = None
    out["llm_status"] = None
    out["llm_error"] = None

    cache: Dict[str, dict] = {}
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    cache[obj["key"]] = obj["value"]
                except Exception:
                    pass

    def cache_get(key: str):
        return cache.get(key)

    def cache_put(key: str, value: dict):
        if key in cache:
            return
        cache[key] = value
        with cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")

    def is_done(row) -> bool:
        st = row.get("llm_status")
        return st in ("ok", "empty")

    todo_idx = [i for i, r in out.iterrows() if not is_done(r)]
    pbar = tqdm(todo_idx, desc="Geothermal classification", unit="row")

    ok = int((out.get("llm_status") == "ok").sum()) if "llm_status" in out.columns else 0
    err = int((out.get("llm_status") == "error").sum()) if "llm_status" in out.columns else 0

    try:
        for i in pbar:
            text = str(out.at[i, text_col] if text_col in out.columns else "") or ""

            if not text.strip():
                out.at[i, "llm_status"] = "empty"
                out.at[i, "llm_is_geothermal"] = "MAYBE"
                out.at[i, "llm_geo_confidence"] = 0.0
                out.at[i, "llm_geo_evidence_short"] = "Empty paragraph."
                out.at[i, "llm_error"] = None
                continue

            key = _fingerprint(text, country, language, geothermal_lexicon)
            cached = cache_get(key)

            try:
                res = cached if cached is not None else llm_is_geothermal(
                    text=text,
                    country=country,
                    ollama_url=ollama_url,
                    model=model,
                    language=language,
                    geothermal_lexicon=geothermal_lexicon
                )
                if cached is None:
                    cache_put(key, res)

                out.at[i, "llm_is_geothermal"] = res.get("is_geothermal")
                out.at[i, "llm_geo_confidence"] = float(res.get("confidence") or 0.0)
                out.at[i, "llm_geo_evidence_short"] = res.get("evidence_short")
                out.at[i, "llm_status"] = "ok"
                out.at[i, "llm_error"] = None
                ok += 1
            except Exception as e:
                out.at[i, "llm_status"] = "error"
                out.at[i, "llm_error"] = repr(e)
                err += 1

            pbar.set_postfix({"ok": ok, "error": err, "cached": cached is not None})

            if sleep_s:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        print(
            f"\nStopped by user. Progress saved to:\n"
            f"- {cache_path}\n"
        )
        raise

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=str, default=str(DEFAULT_PROJECT_DIR))
    ap.add_argument("--input-csv", type=str, default="output/workflow/newspapers_cleaned_paragraphs.csv")
    ap.add_argument("--out-csv", type=str, default="output/workflow/paragraph_geothermal_ollama.csv")

    ap.add_argument("--text-col", type=str, default="paragraph_text")
    ap.add_argument("--region-col", type=str, default="region_name")

    ap.add_argument("--cache", type=str, default="cache/is_geothermal.jsonl")

    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--sleep-s", type=float, default=0.0)

    ap.add_argument("--ollama-url", type=str, default="http://localhost:11434/api/generate")
    ap.add_argument("--model", type=str, default="llama3.1:8b")
    ap.add_argument("--country", type=str, default="Nederland", help="Backward-compatible single-country shorthand.")
    ap.add_argument("--countries", nargs="+", default=None)
    ap.add_argument("--country-scope", type=str, default="")
    ap.add_argument("--language", type=str, default="nl", help="Language for the text analysis.")

    args = ap.parse_args()
    country_scope = country_scope_from_args(
        country=args.country,
        countries=args.countries,
        country_scope=args.country_scope,
    )
    project_dir = Path(args.project_dir).expanduser().resolve()
    os.chdir(project_dir)

    df = pd.read_csv(args.input_csv, encoding="utf-8")

    if "uid" not in df.columns:
        df["uid"] = df.apply(lambda r: make_uid(r.to_dict()), axis=1)

    # Keep uid as the content identifier, but use a unique row key internally.
    df = set_unique_row_index(df)

    df["word_count"] = df[args.text_col].apply(lambda x: len(str(x).split()))
    input_rows = len(df)
    df = df[(df["word_count"] >= 20) & (df["word_count"] < 500)].copy()
    print(f"[workflow_table] paragraphs_before_geothermal_length_filter: {input_rows}")
    print(f"[workflow_table] paragraphs_after_geothermal_length_filter: {len(df)}")

    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    geothermal_lexicon = load_geothermal_lexicon(project_dir, args.language)

    out = batch_geothermal_resumable(
        df=df,
        text_col=args.text_col,
        cache_path=cache_path,
        sleep_s=args.sleep_s,
        ollama_url=args.ollama_url,
        model=args.model,
        country=country_scope.label,
        language=args.language,
        geothermal_lexicon=geothermal_lexicon,
    )

    out_csv = Path(args.out_csv)

    # uid remains a regular column in the final CSV; the internal row key stays as index.
    out = out.copy()

    remaining = incomplete_count(out)
    if remaining:
        raise RuntimeError(
            f"Geothermal classification is incomplete ({remaining} rows unfinished). "
            f"Completed model calls are saved in {cache_path}; "
            "not writing the final Snakemake output CSV."
        )

    write_final_csv_atomic(out, out_csv)
    print(f"Wrote: {args.out_csv}  (rows={len(out)})")
    if "llm_is_geothermal" in out.columns:
        geothermal_yes = int(out["llm_is_geothermal"].astype(str).str.upper().eq("YES").sum())
        geothermal_maybe = int(out["llm_is_geothermal"].astype(str).str.upper().eq("MAYBE").sum())
        geothermal_no = int(out["llm_is_geothermal"].astype(str).str.upper().eq("NO").sum())
        print(f"[workflow_table] paragraphs_classified_geothermal_yes: {geothermal_yes}")
        print(f"[workflow_table] paragraphs_classified_geothermal_maybe: {geothermal_maybe}")
        print(f"[workflow_table] paragraphs_classified_geothermal_no: {geothermal_no}")


if __name__ == "__main__":
    main()
