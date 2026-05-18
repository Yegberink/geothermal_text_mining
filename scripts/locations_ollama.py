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
from typing import Dict, Optional

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm.auto import tqdm

SYSTEM = (
    "You are an assistant that extracts the single primary geographic location "
    "a newspaper paragraph is mainly about."
)

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[1]


def _fingerprint(text: str, region: str, country: str) -> str:
    h = hashlib.sha256()
    h.update((region or "").encode("utf-8"))
    h.update(b"\n")
    h.update((country or "").encode("utf-8"))
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
- If the paragraph is general or national-level, return "{country}".
- If no subnational location is stated but the paragraph clearly refers to the country as a whole, return "{country}".
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


def save_checkpoint(out: pd.DataFrame, checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    out.to_parquet(tmp_path, index=True)
    tmp_path.replace(checkpoint_path)


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
        if "uid" in out.columns and out.index.name != "uid":
            out = out.set_index("uid", drop=True)

        out = out.reindex(df.index)

        for c in df.columns:
            if c not in out.columns:
                out[c] = df[c]
    else:
        out = df.copy()
        out["llm_location"] = None
        out["llm_granularity"] = None
        out["llm_confidence"] = 0.0
        out["llm_reasoning_short"] = None
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

    out = out.copy()
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
        if partial_csv_path is not None:
            partial_csv_path.parent.mkdir(parents=True, exist_ok=True)
            partial_out = out.copy()
            partial_out["uid"] = partial_out.index
            partial_out.to_csv(partial_csv_path, index=False, encoding="utf-8")
        print(
            f"\nStopped by user. Progress saved to:\n"
            f"- {checkpoint_path}\n"
            + (f"- {partial_csv_path}\n" if partial_csv_path is not None else "")
        )
        return out

    save_checkpoint(out, checkpoint_path)
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

    # Important fix:
    # keep uid only as index, not both index and column.
    df = df.set_index("uid", drop=True)

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

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    # Restore uid as a regular column for the final CSV export.
    out = out.copy()
    out["uid"] = out.index

    out.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"Wrote: {args.out_csv}  (rows={len(out)})")
    if "llm_location" in out.columns:
        located = out["llm_location"].fillna("").astype(str).str.strip().str.upper().ne("NONE")
        print(f"[workflow_table] paragraphs_sent_to_location_extraction: {len(out)}")
        print(f"[workflow_table] paragraphs_with_extracted_location: {int(located.sum())}")
        print(f"[workflow_table] paragraphs_without_extracted_location: {int((~located).sum())}")


if __name__ == "__main__":
    main()
