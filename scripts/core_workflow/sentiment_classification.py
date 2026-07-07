#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict

import pandas as pd
from tqdm.auto import tqdm

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_LANGUAGE = "dutch"
DEFAULT_PROMPT_VARIANT = "zero_shot"
ROW_UID_COL = "_row_uid"
SENTIMENTS = ["negative", "neutral", "positive"]
OLLAMA_SYSTEM_PROMPT = (
    "You are a careful sentiment classification assistant for newspaper sentences about geothermal energy. "
    "You must return a single sentiment label and a short rationale in valid JSON only."
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=str, default=str(DEFAULT_PROJECT_DIR))
    ap.add_argument("--input-csv", type=str, default="output/workflow/sentence_locations_ollama.csv")
    ap.add_argument("--output-csv", type=str, default="output/workflow/sentence_sentiment_llm.csv")
    ap.add_argument("--cache", type=str, default="cache/sentiment_classification.jsonl")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--ollama-url", type=str, default=DEFAULT_OLLAMA_URL)
    ap.add_argument("--text-col", type=str, default="sentence_text")
    ap.add_argument("--language", type=str, default=DEFAULT_LANGUAGE)
    ap.add_argument("--prompt-variant", type=str, default=DEFAULT_PROMPT_VARIANT)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--sleep-s", type=float, default=0.0)
    ap.add_argument("--save-every", type=int, default=25)
    return ap.parse_args()


def make_uid(row: Dict[str, object]) -> str:
    if str(row.get("sentence_uid", "")).strip():
        return str(row["sentence_uid"])
    if str(row.get("uid", "")).strip():
        return str(row["uid"])
    base = "||".join(
        [
            str(row.get("source", "")),
            str(row.get("document_title", "")),
            str(row.get("publish_date", "")),
            str(row.get("paragraph_uid", "")),
            str(row.get("sentence_id", "")),
            str(row.get("sentence_text", "")),
            str(row.get("paragraph_id", "")),
            str(row.get("paragraph_text", "")),
        ]
    )
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def set_unique_row_index(df: pd.DataFrame, uid_col: str = "sentence_uid") -> pd.DataFrame:
    """Keep the semantic uid as data, but use a unique key for row alignment."""
    out = df.copy()
    if uid_col not in out.columns:
        if out.index.name == uid_col:
            out[uid_col] = out.index
        else:
            out[uid_col] = out.apply(make_uid, axis=1)
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


def _fingerprint(text: str, model_name: str, prompt_variant: str, language: str) -> str:
    h = hashlib.sha256()
    h.update((model_name or "").encode("utf-8"))
    h.update(b"\n")
    h.update((prompt_variant or "").encode("utf-8"))
    h.update(b"\n")
    h.update((language or "").encode("utf-8"))
    h.update(b"\n")
    h.update((text or "").strip().encode("utf-8"))
    return h.hexdigest()


def _install_interrupt_handlers() -> None:
    def _handle_interrupt(signum, _frame):
        signame = signal.Signals(signum).name
        raise KeyboardInterrupt(f"Received {signame}")

    signal.signal(signal.SIGINT, _handle_interrupt)
    signal.signal(signal.SIGTERM, _handle_interrupt)


def normalize_sentiment_label(label: str) -> str:
    value = str(label or "").strip().lower()
    if value in SENTIMENTS:
        return value
    raise ValueError(f"Unsupported sentiment label: {label!r}")


def build_ollama_prompt(text: str, language: str, prompt_variant: str) -> str:
    variant_line = (
        "Do not use any few-shot examples; rely only on the definitions below."
        if prompt_variant == "zero_shot"
        else "Use the task definitions below."
    )
    return f"""
Task: Classify the sentiment of this newspaper sentence about geothermal energy.

Sentence language: {language}

Use only these labels:
- negative
- neutral
- positive

Interpretation rules:
- negative: emphasizes risk, costs, obstacles, criticism, harm, uncertainty, conflict, or failure
- neutral: mainly factual, procedural, descriptive, or mixed without clear evaluative polarity
- positive: emphasizes benefits, support, progress, feasibility, opportunity, or success

{variant_line}

Return valid JSON only with these keys:
- sentiment
- confidence
- rationale_short

Requirements:
- sentiment must be exactly one of: negative, neutral, positive
- confidence must be a number from 0 to 1
- rationale_short must be <= 20 words

Sentence:
{text}
""".strip()


def extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    return None


def parse_ollama_sentiment_response(raw_response: str) -> dict[str, object]:
    candidate = extract_json_object(raw_response)
    if candidate is not None:
        try:
            obj = json.loads(candidate)
            raw_label = str(obj.get("sentiment", "") or "").strip()
            return {
                "sentiment": normalize_sentiment_label(raw_label),
                "confidence": max(0.0, min(1.0, float(obj.get("confidence", 0.0) or 0.0))),
                "evidence_short": str(obj.get("rationale_short", "") or "").strip(),
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    lowered = raw_response.lower()
    lowered_norm = lowered.replace("’", "'").replace("‘", "'")
    refusal_markers = [
        "can't process text",
        "cannot process text",
        "can't process that text",
        "cannot process that text",
        "can't analyze this text",
        "cannot analyze this text",
        "can't perform sentiment analysis",
        "cannot perform sentiment analysis",
        "can't classify the sentiment",
        "cannot classify the sentiment",
        "doesn't appear to be about geothermal",
        "does not appear to be about geothermal",
        "doesn't mention geothermal",
        "does not mention geothermal",
        "not mention geothermal",
        "not about geothermal",
    ]
    if any(marker in lowered_norm for marker in refusal_markers):
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "evidence_short": "Model declined; treated as neutral.",
        }

    for sentiment in SENTIMENTS:
        if sentiment in lowered_norm:
            return {
                "sentiment": sentiment,
                "confidence": 0.0,
                "evidence_short": "",
            }

    raise ValueError(f"Ollama did not return a parseable sentiment label: {raw_response!r}")


def call_ollama_sentiment(
    text: str,
    model_name: str,
    ollama_url: str,
    language: str,
    prompt_variant: str,
    timeout: int,
) -> dict[str, object]:
    payload = {
        "model": model_name,
        "system": OLLAMA_SYSTEM_PROMPT,
        "prompt": build_ollama_prompt(text=text, language=language, prompt_variant=prompt_variant),
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 120},
    }
    request = urllib.request.Request(
        ollama_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404 and "not found" in detail.lower():
            raise RuntimeError(
                f"Ollama model '{model_name}' was not found at {ollama_url}. "
                f"Pull it first with: ollama pull {model_name}"
            ) from exc
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Ollama at {ollama_url}: {exc.reason}") from exc

    raw_response = str(body.get("response", "") or "").strip()
    return parse_ollama_sentiment_response(raw_response)


def batch_sentiment_resumable(
    df: pd.DataFrame,
    text_col: str,
    cache_path: Path,
    model_name: str,
    ollama_url: str,
    language: str,
    prompt_variant: str,
    timeout: int,
    sleep_s: float,
) -> pd.DataFrame:
    out = df.copy()
    out["sentiment"] = None
    out["sentiment_norm"] = None
    out["sentiment_confidence"] = 0.0
    out["sentiment_evidence_short"] = None
    out["sentiment_status"] = None
    out["sentiment_error"] = None

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
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache[key] = value
        with cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")

    def is_done(row) -> bool:
        return row.get("sentiment_status") in ("ok", "empty")

    todo_idx = [i for i, row in out.iterrows() if not is_done(row)]
    pbar = tqdm(total=len(todo_idx), desc=f"Sentence sentiment ({model_name})", unit="row")

    try:
        for i in todo_idx:
            text = str(out.at[i, text_col] if text_col in out.columns else "") or ""
            if not text.strip():
                out.at[i, "sentiment"] = "neutral"
                out.at[i, "sentiment_norm"] = "neutral"
                out.at[i, "sentiment_confidence"] = 0.0
                out.at[i, "sentiment_evidence_short"] = "Empty sentence."
                out.at[i, "sentiment_status"] = "empty"
                out.at[i, "sentiment_error"] = None
                pbar.update(1)
                continue

            key = _fingerprint(text, model_name=model_name, prompt_variant=prompt_variant, language=language)
            cached = cache_get(key)
            try:
                result = cached if cached is not None else call_ollama_sentiment(
                    text=text,
                    model_name=model_name,
                    ollama_url=ollama_url,
                    language=language,
                    prompt_variant=prompt_variant,
                    timeout=timeout,
                )
                if cached is None:
                    cache_put(key, result)
                out.at[i, "sentiment"] = result["sentiment"]
                out.at[i, "sentiment_norm"] = result["sentiment"]
                out.at[i, "sentiment_confidence"] = float(result["confidence"])
                out.at[i, "sentiment_evidence_short"] = result["evidence_short"]
                out.at[i, "sentiment_status"] = "ok"
                out.at[i, "sentiment_error"] = None
            except Exception as exc:
                out.at[i, "sentiment_status"] = "error"
                out.at[i, "sentiment_error"] = repr(exc)

            pbar.update(1)
            if sleep_s:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        print(f"\nStopped by user. Progress saved to:\n- {cache_path}")
        raise
    finally:
        pbar.close()

    return out


def incomplete_count(out: pd.DataFrame) -> int:
    if "sentiment_status" not in out.columns:
        return len(out)
    return int((~out["sentiment_status"].isin(["ok", "empty"])).sum())


def restart_error_rows(out: pd.DataFrame) -> int:
    if "sentiment_status" not in out.columns:
        return 0
    error_mask = out["sentiment_status"].eq("error")
    restarted = int(error_mask.sum())
    if restarted:
        reset_values = {
            "sentiment": None,
            "sentiment_norm": None,
            "sentiment_confidence": 0.0,
            "sentiment_evidence_short": None,
            "sentiment_status": None,
            "sentiment_error": None,
        }
        for col, value in reset_values.items():
            out.loc[error_mask, col] = value
    return restarted


def sentiment_result_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        "sentiment",
        "sentiment_norm",
        "sentiment_confidence",
        "sentiment_evidence_short",
        "sentiment_status",
        "sentiment_error",
    ]
    return [c for c in cols if c in df.columns]


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    os.chdir(project_dir)

    _install_interrupt_handlers()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    cache_path = Path(args.cache)

    df = pd.read_csv(input_csv)
    if args.text_col not in df.columns:
        raise ValueError(f"Column {args.text_col!r} not found in {input_csv}")

    df = df.copy()
    if "sentence_uid" not in df.columns:
        df["sentence_uid"] = df.apply(make_uid, axis=1)
    df = set_unique_row_index(df)

    out = batch_sentiment_resumable(
        df=df,
        text_col=args.text_col,
        cache_path=cache_path,
        model_name=args.model,
        ollama_url=args.ollama_url,
        language=args.language,
        prompt_variant=args.prompt_variant,
        timeout=args.timeout,
        sleep_s=args.sleep_s,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    to_write = out.reset_index(drop=True)
    ok_n = int((to_write.get("sentiment_status") == "ok").sum()) if "sentiment_status" in to_write.columns else 0
    err_n = int((to_write.get("sentiment_status") == "error").sum()) if "sentiment_status" in to_write.columns else 0
    remaining = incomplete_count(out)
    if remaining:
        raise RuntimeError(
            f"Sentence sentiment classification is incomplete ({remaining} rows unfinished). "
            f"Completed model calls are saved in {cache_path}; "
            "not writing the final Snakemake output CSV."
        )
    to_write.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Wrote sentiment output: {output_csv}")
    print(f"Sentiment rows={len(to_write)}, ok={ok_n}, error={err_n}")
    print(f"[workflow_table] sentences_sent_to_sentiment: {len(to_write)}")
    print(f"[workflow_table] sentences_with_sentiment_ok: {ok_n}")
    print(f"[workflow_table] sentences_with_sentiment_error: {err_n}")


if __name__ == "__main__":
    main()
