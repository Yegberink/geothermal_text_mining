#!/usr/bin/env python3
"""
locations_ollama.py
Resumable paragraph -> primary location extraction via Ollama.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm.auto import tqdm

SYSTEM = (
    "You are an assistant that extracts the single primary geographic location "
    "a newspaper paragraph is mainly about."
)

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[1]
ROW_UID_COL = "_row_uid"
NO_LOCATION_VALUES = {"", "none", "nan", "null"}
LOCATION_TRACKING_DEFAULTS = {
    "llm_paragraph_location_raw": None,
    "llm_paragraph_returned_none": None,
    "llm_document_location_raw": None,
    "llm_document_returned_none": None,
    "llm_country_fallback_applied": False,
}


def _fingerprint(text: str, region: str, country: str) -> str:
    h = hashlib.sha256()
    h.update((region or "").encode("utf-8"))
    h.update(b"\n")
    h.update((country or "").encode("utf-8"))
    h.update(b"\n")
    h.update((text or "").encode("utf-8"))
    return h.hexdigest()


def _document_fingerprint(document_key: str, text: str, region: str, country: str) -> str:
    h = hashlib.sha256()
    h.update(b"document_location\n")
    h.update((document_key or "").encode("utf-8"))
    h.update(b"\n")
    h.update((region or "").encode("utf-8"))
    h.update(b"\n")
    h.update((country or "").encode("utf-8"))
    h.update(b"\n")
    h.update((text or "").encode("utf-8"))
    return h.hexdigest()


def is_valid_location(value: object) -> bool:
    return str(value or "").strip().lower() not in NO_LOCATION_VALUES


def normalize_location_key(value: object) -> str:
    return str(value or "").strip().lower()


def normalize_location_result(obj: dict) -> dict:
    loc = obj.get("location", "NONE") or "NONE"
    gran = obj.get("granularity", "none") or "none"
    conf = obj.get("confidence", 0.0) or 0.0
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    reason = obj.get("reasoning_short", "") or ""
    return {
        "location": loc,
        "granularity": gran,
        "confidence": conf,
        "reasoning_short": reason,
    }


def parse_location_response(out: str) -> dict:
    out = str(out or "").strip()
    start = out.find("{")
    end = out.rfind("}")
    if start == -1 or end == -1:
        return {
            "location": "NONE",
            "granularity": "none",
            "confidence": 0.0,
            "reasoning_short": "No JSON returned.",
        }

    try:
        obj = json.loads(out[start:end + 1])
    except json.JSONDecodeError:
        return {
            "location": "NONE",
            "granularity": "none",
            "confidence": 0.0,
            "reasoning_short": "Invalid JSON.",
        }
    return normalize_location_result(obj)


def ensure_location_tracking_columns(out: pd.DataFrame) -> pd.DataFrame:
    for col, default in LOCATION_TRACKING_DEFAULTS.items():
        if col not in out.columns:
            out[col] = default
    return out


def backfill_missing_paragraph_raw_columns(out: pd.DataFrame) -> pd.DataFrame:
    out = ensure_location_tracking_columns(out)
    if "llm_status" in out.columns:
        completed = out["llm_status"].isin(["ok", "empty"])
    else:
        completed = pd.Series(True, index=out.index)
    missing_raw = out["llm_paragraph_location_raw"].isna() & completed
    if missing_raw.any():
        out.loc[missing_raw, "llm_paragraph_location_raw"] = out.loc[missing_raw, "llm_location"]
        out.loc[missing_raw, "llm_paragraph_returned_none"] = ~out.loc[
            missing_raw, "llm_location"
        ].map(is_valid_location)
    return out


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
    retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError, requests.HTTPError)),
)
def llm_primary_location(
    text: str,
    newspaper_region_name: str,
    country: str,
    ollama_url: str,
    model: str,
) -> dict:
    prompt = f"""
Task: Determine the ONE primary geographic location this paragraph is mainly about.

Context:
- The newspaper's coverage region is: {newspaper_region_name}
- The paragraph is about geothermal energy.
- Primary country of interest: {country}

Rules:
- Return exactly ONE location name, or "NONE" if no clear primary location.
- Prefer the most specific location that is clearly the focus (site/city/municipality).
- Do NOT list multiple places.
- Output MUST be valid JSON only, with keys:
  location, granularity, confidence, reasoning_short

Granularity must be one of: site, city, municipality, province, country, none
Confidence must be a number from 0 to 1.

Paragraph:
{text}
""".strip()

    payload = {
        "model": model,
        "system": SYSTEM,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 200},
    }

    r = requests.post(ollama_url, json=payload, timeout=120)
    r.raise_for_status()
    out = (r.json().get("response") or "").strip()
    return parse_location_response(out)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError, requests.HTTPError)),
)
def llm_document_primary_location(
    document_text: str,
    newspaper_region_name: str,
    country: str,
    ollama_url: str,
    model: str,
) -> dict:
    prompt = f"""
Task: Determine the ONE primary geographic location this full newspaper document is mainly about.

Context:
- The newspaper's coverage region is: {newspaper_region_name}
- The document contains one or more paragraphs selected for geothermal-energy processing.
- Primary country of interest: {country}

Rules:
- Return exactly ONE location name, or "NONE" if no clear primary location.
- Prefer the most specific location that is clearly the focus (site/city/municipality).
- Do NOT list multiple places.
- Output MUST be valid JSON only, with keys:
  location, granularity, confidence, reasoning_short

Granularity must be one of: site, city, municipality, province, country, none
Confidence must be a number from 0 to 1.

Document:
{document_text}
""".strip()

    payload = {
        "model": model,
        "system": SYSTEM,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 200},
    }

    r = requests.post(ollama_url, json=payload, timeout=120)
    r.raise_for_status()
    out = (r.json().get("response") or "").strip()
    return parse_location_response(out)


def document_key_for_row(row: pd.Series) -> str:
    body_hash = str(row.get("body_hash", "") or "").strip()
    if body_hash and body_hash.lower() not in NO_LOCATION_VALUES:
        return body_hash
    return "||".join(
        [
            str(row.get("source", "") or ""),
            str(row.get("document_title", "") or ""),
            str(row.get("publish_date", "") or ""),
        ]
    )


def document_text_for_group(group: pd.DataFrame, text_col: str) -> str:
    if "body" in group.columns:
        for value in group["body"].dropna().astype(str):
            text = value.strip()
            if text and text.lower() not in NO_LOCATION_VALUES:
                return text

    ordered = group.copy()
    if "paragraph_id" in ordered.columns:
        ordered = ordered.sort_values("paragraph_id", kind="stable")
    return "\n\n".join(ordered[text_col].fillna("").astype(str).str.strip().tolist()).strip()


def location_payload_from_row(row: pd.Series) -> dict:
    return {
        "location": row.get("llm_location", "NONE") or "NONE",
        "granularity": row.get("llm_granularity", "none") or "none",
        "confidence": row.get("llm_confidence", 0.0) or 0.0,
        "reasoning_short": row.get("llm_reasoning_short", "") or "",
    }


def apply_location_payload(
    out: pd.DataFrame,
    idxs,
    payload: dict,
    source: str,
    review_flag: bool,
    review_reason: Optional[str],
) -> None:
    result = normalize_location_result(payload)
    out.loc[idxs, "llm_location"] = result["location"]
    out.loc[idxs, "llm_granularity"] = result["granularity"]
    out.loc[idxs, "llm_confidence"] = result["confidence"]
    out.loc[idxs, "llm_reasoning_short"] = result["reasoning_short"]
    out.loc[idxs, "llm_status"] = "ok"
    out.loc[idxs, "llm_error"] = None
    out.loc[idxs, "llm_location_source"] = source
    out.loc[idxs, "llm_location_review_flag"] = bool(review_flag)
    out.loc[idxs, "llm_location_review_reason"] = review_reason
    out.loc[idxs, "llm_country_fallback_applied"] = source == "country_fallback"


def country_fallback_payload(country: str) -> dict:
    return {
        "location": country,
        "granularity": "country",
        "confidence": 0.0,
        "reasoning_short": "No clear paragraph- or document-level location; using country fallback.",
    }


def apply_document_location_fallback(
    out: pd.DataFrame,
    text_col: str,
    region_col: str,
    country: str,
    document_location_resolver: Callable[[str, str, str], dict],
) -> pd.DataFrame:
    out = out.copy()
    out = ensure_location_tracking_columns(out)
    for col, default in [
        ("llm_location_source", None),
        ("llm_location_review_flag", False),
        ("llm_location_review_reason", None),
    ]:
        if col not in out.columns:
            out[col] = default

    valid_mask = out["llm_location"].map(is_valid_location)
    out.loc[valid_mask & out["llm_location_source"].isna(), "llm_location_source"] = "paragraph"
    out.loc[valid_mask & out["llm_location_review_reason"].isna(), "llm_location_review_flag"] = False

    if "llm_status" in out.columns:
        completed_mask = out["llm_status"].isin(["ok", "empty"])
    else:
        completed_mask = pd.Series(True, index=out.index)
    missing_mask = ~valid_mask & completed_mask
    if not missing_mask.any():
        return out

    doc_keys = out.apply(document_key_for_row, axis=1)
    fallback_count = 0
    document_llm_count = 0
    country_fallback_count = 0

    for document_key, idxs in doc_keys.groupby(doc_keys).groups.items():
        group = out.loc[list(idxs)]
        if "llm_status" in group.columns:
            group_completed = group["llm_status"].isin(["ok", "empty"])
        else:
            group_completed = pd.Series(True, index=group.index)
        group_missing = group[~group["llm_location"].map(is_valid_location) & group_completed]
        if group_missing.empty:
            continue

        group_valid = group[group["llm_location"].map(is_valid_location)]
        unique_locations = {}
        for _, valid_row in group_valid.iterrows():
            key = normalize_location_key(valid_row.get("llm_location"))
            unique_locations.setdefault(key, location_payload_from_row(valid_row))

        missing_idxs = group_missing.index
        if len(unique_locations) == 1:
            payload = next(iter(unique_locations.values()))
            apply_location_payload(
                out,
                missing_idxs,
                payload,
                source="document_single_location",
                review_flag=False,
                review_reason=None,
            )
            fallback_count += len(missing_idxs)
            continue

        review_reason = "multiple_document_locations" if unique_locations else "no_document_paragraph_locations"
        document_text = document_text_for_group(group, text_col)
        region_name = str(group.iloc[0].get(region_col, "") or country)
        payload = document_location_resolver(document_text, region_name, str(document_key))
        document_result = normalize_location_result(payload)
        document_returned_none = not is_valid_location(document_result.get("location"))
        source = "document_llm"
        if document_returned_none:
            payload = country_fallback_payload(country)
            source = "country_fallback"
            review_reason = f"{review_reason};document_llm_no_location"
            country_fallback_count += len(missing_idxs)
        else:
            document_llm_count += len(missing_idxs)
        apply_location_payload(
            out,
            missing_idxs,
            payload,
            source=source,
            review_flag=True,
            review_reason=review_reason,
        )
        out.loc[missing_idxs, "llm_document_location_raw"] = document_result["location"]
        out.loc[missing_idxs, "llm_document_returned_none"] = bool(document_returned_none)

    if fallback_count or document_llm_count or country_fallback_count:
        print(f"[workflow_table] paragraph_locations_filled_from_document_single_location: {fallback_count}")
        print(f"[workflow_table] paragraph_locations_filled_from_document_llm: {document_llm_count}")
        print(f"[workflow_table] paragraph_locations_filled_from_country_fallback: {country_fallback_count}")
    return out


def save_checkpoint(out: pd.DataFrame, checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    out.to_parquet(tmp_path, index=True)
    tmp_path.replace(checkpoint_path)


def write_partial_csv(out: pd.DataFrame, partial_csv_path: Optional[Path]) -> None:
    if partial_csv_path is None:
        return
    partial_csv_path.parent.mkdir(parents=True, exist_ok=True)
    partial_out = out.copy()
    partial_out[ROW_UID_COL] = partial_out.index
    partial_out.to_csv(partial_csv_path, index=False, encoding="utf-8")


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
            "llm_location": None,
            "llm_granularity": None,
            "llm_confidence": 0.0,
            "llm_reasoning_short": None,
            "llm_status": None,
            "llm_error": None,
            "llm_location_source": None,
            "llm_location_review_flag": False,
            "llm_location_review_reason": None,
            **LOCATION_TRACKING_DEFAULTS,
        }
        for col, value in reset_values.items():
            out.loc[error_mask, col] = value
    return restarted


def write_final_csv_atomic(out: pd.DataFrame, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_csv.with_suffix(out_csv.suffix + ".tmp")
    out.to_csv(tmp_path, index=False, encoding="utf-8")
    tmp_path.replace(out_csv)


def count_true(series: pd.Series) -> int:
    truthy = series.fillna(False).map(lambda value: str(value).strip().lower() in {"1", "true", "yes", "y"})
    return int(truthy.sum())


def batch_primary_locations_resumable(
    df: pd.DataFrame,
    text_col: str,
    region_col: str,
    checkpoint_path: Path,
    cache_path: Path,
    save_every: int,
    sleep_s: float,
    ollama_url: str,
    model: str,
    country: str,
    partial_csv_path: Optional[Path],
) -> pd.DataFrame:
    if checkpoint_path.exists():
        out = pd.read_parquet(checkpoint_path)

        # Defensive handling for older or differently written checkpoints.
        out = set_unique_row_index(out)

        out = out.reindex(df.index)

        for c in df.columns:
            out[c] = df[c]
        restarted = restart_error_rows(out)
        if restarted:
            print(f"[resume] Restarting {restarted} location extraction rows that previously errored.")
    else:
        out = df.copy()
        out["llm_location"] = None
        out["llm_granularity"] = None
        out["llm_confidence"] = 0.0
        out["llm_reasoning_short"] = None
        out["llm_status"] = None
        out["llm_error"] = None
        out = ensure_location_tracking_columns(out)

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

    def resolve_document_location(document_text: str, region_name: str, document_key: str) -> dict:
        if not str(document_text or "").strip():
            return {
                "location": "NONE",
                "granularity": "none",
                "confidence": 0.0,
                "reasoning_short": "Empty document text.",
            }

        key = _document_fingerprint(document_key, document_text, region_name, country)
        cached = cache_get(key)
        if cached is not None:
            return cached

        res = llm_document_primary_location(
            document_text=document_text,
            newspaper_region_name=region_name,
            country=country,
            ollama_url=ollama_url,
            model=model,
        )
        cache_put(key, res)
        if sleep_s:
            time.sleep(sleep_s)
        return res

    def is_done(row) -> bool:
        st = row.get("llm_status")
        return st in ("ok", "empty")

    out = out.copy()
    out = ensure_location_tracking_columns(out)
    if "llm_is_geothermal" in out.columns:
        out = out[out["llm_is_geothermal"].astype(str).str.upper() == "YES"].copy()

    todo_idx = [i for i, r in out.iterrows() if not is_done(r)]
    pbar = tqdm(todo_idx, desc="Geothermal location extraction", unit="row")

    ok = int((out.get("llm_status") == "ok").sum()) if "llm_status" in out.columns else 0
    err = int((out.get("llm_status") == "error").sum()) if "llm_status" in out.columns else 0
    processed_since_save = 0

    try:
        for i in pbar:
            text = str(out.at[i, text_col] if text_col in out.columns else "") or ""
            region_name = str(out.at[i, region_col] if region_col in out.columns else country) or country

            if not text.strip():
                out.at[i, "llm_status"] = "empty"
                out.at[i, "llm_location"] = "NONE"
                out.at[i, "llm_granularity"] = "none"
                out.at[i, "llm_confidence"] = 0.0
                out.at[i, "llm_reasoning_short"] = "Empty paragraph."
                out.at[i, "llm_error"] = None
                out.at[i, "llm_paragraph_location_raw"] = "NONE"
                out.at[i, "llm_paragraph_returned_none"] = None
                processed_since_save += 1
                continue

            key = _fingerprint(text, region_name, country)
            cached = cache_get(key)

            try:
                res = cached if cached is not None else llm_primary_location(
                    text=text,
                    newspaper_region_name=region_name,
                    country=country,
                    ollama_url=ollama_url,
                    model=model,
                )
                if cached is None:
                    cache_put(key, res)

                out.at[i, "llm_location"] = res.get("location")
                out.at[i, "llm_granularity"] = res.get("granularity")
                out.at[i, "llm_confidence"] = float(res.get("confidence") or 0.0)
                out.at[i, "llm_reasoning_short"] = res.get("reasoning_short")
                out.at[i, "llm_status"] = "ok"
                out.at[i, "llm_error"] = None
                out.at[i, "llm_paragraph_location_raw"] = res.get("location")
                out.at[i, "llm_paragraph_returned_none"] = not is_valid_location(res.get("location"))
                ok += 1
            except Exception as e:
                out.at[i, "llm_status"] = "error"
                out.at[i, "llm_error"] = repr(e)
                err += 1

            processed_since_save += 1
            pbar.set_postfix({"ok": ok, "error": err, "cached": cached is not None})

            if sleep_s:
                time.sleep(sleep_s)

            if processed_since_save >= save_every:
                save_checkpoint(out, checkpoint_path)
                processed_since_save = 0

    except KeyboardInterrupt:
        save_checkpoint(out, checkpoint_path)
        write_partial_csv(out, partial_csv_path)
        print(
            f"\nStopped by user. Progress saved to:\n"
            f"- {checkpoint_path}\n"
            + (f"- {partial_csv_path}\n" if partial_csv_path is not None else "")
        )
        raise

    out = backfill_missing_paragraph_raw_columns(out)
    out = apply_document_location_fallback(
        out=out,
        text_col=text_col,
        region_col=region_col,
        country=country,
        document_location_resolver=resolve_document_location,
    )

    save_checkpoint(out, checkpoint_path)
    write_partial_csv(out, partial_csv_path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=str, default=str(DEFAULT_PROJECT_DIR))
    ap.add_argument("--input-csv", type=str, default="output/text/paragraph_geothermal_ollama.csv")
    ap.add_argument("--out-csv", type=str, default="output/text/paragraph_locations_ollama.csv")

    ap.add_argument("--text-col", type=str, default="paragraph_text")
    ap.add_argument("--region-col", type=str, default="region_name")

    ap.add_argument("--checkpoint", type=str, default="cache/geo_checkpoint.parquet")
    ap.add_argument("--cache", type=str, default="cache/geo_cache.jsonl")
    ap.add_argument("--partial-csv", type=str, default="cache/geo_partial_results.csv")

    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--sleep-s", type=float, default=0.0)

    ap.add_argument("--ollama-url", type=str, default="http://localhost:11434/api/generate")
    ap.add_argument("--model", type=str, default="llama3.1:8b")
    ap.add_argument("--country", type=str, default="Nederland")

    args = ap.parse_args()

    project_dir = Path(args.project_dir).expanduser().resolve()
    os.chdir(project_dir)

    df = pd.read_csv(args.input_csv, encoding="utf-8")
    if "uid" not in df.columns:
        df["uid"] = df.apply(lambda r: make_uid(r.to_dict()), axis=1)

    # Keep uid as the content identifier, but align checkpoints on a unique row key.
    df = set_unique_row_index(df)

    df["word_count"] = df[args.text_col].apply(lambda x: len(str(x).split()))
    input_rows = len(df)
    df = df[(df["word_count"] >= 20) & (df["word_count"] < 500)].copy()
    print(f"[workflow_table] paragraphs_before_location_length_filter: {input_rows}")
    print(f"[workflow_table] paragraphs_after_location_length_filter: {len(df)}")

    checkpoint_path = Path(args.checkpoint)
    cache_path = Path(args.cache)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    out = batch_primary_locations_resumable(
        df=df,
        text_col=args.text_col,
        region_col=args.region_col,
        checkpoint_path=checkpoint_path,
        cache_path=cache_path,
        save_every=args.save_every,
        sleep_s=args.sleep_s,
        ollama_url=args.ollama_url,
        model=args.model,
        country=args.country,
        partial_csv_path=Path(args.partial_csv) if args.partial_csv else None,
    )

    out_csv = Path(args.out_csv)

    # uid remains a regular column in the final CSV; the internal row key stays as index.
    out = out.copy()

    remaining = incomplete_count(out)
    if remaining:
        raise RuntimeError(
            f"Location extraction is incomplete ({remaining} rows unfinished). "
            f"Progress is saved in {checkpoint_path} and {args.partial_csv}; "
            "not writing the final Snakemake output CSV."
        )

    write_final_csv_atomic(out, out_csv)
    print(f"Wrote: {args.out_csv}  (rows={len(out)})")
    if "llm_location" in out.columns:
        located = out["llm_location"].fillna("").astype(str).str.strip().str.upper().ne("NONE")
        print(f"[workflow_table] paragraphs_sent_to_location_extraction: {len(out)}")
        print(f"[workflow_table] paragraphs_with_extracted_location: {int(located.sum())}")
        print(f"[workflow_table] paragraphs_without_extracted_location: {int((~located).sum())}")
    if "llm_paragraph_returned_none" in out.columns:
        print(
            "[workflow_table] paragraph_location_llm_none_responses: "
            f"{count_true(out['llm_paragraph_returned_none'])}"
        )
    if "llm_document_returned_none" in out.columns:
        print(
            "[workflow_table] document_location_llm_none_responses: "
            f"{count_true(out['llm_document_returned_none'])}"
        )
    if "llm_country_fallback_applied" in out.columns:
        print(
            "[workflow_table] paragraph_locations_filled_from_country_fallback_final: "
            f"{count_true(out['llm_country_fallback_applied'])}"
        )


if __name__ == "__main__":
    main()
