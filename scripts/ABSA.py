import re
import argparse
import os
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
from pyabsa import AspectTermExtraction as ATEPC


# ----------------------------
# 1) Load model
# ----------------------------
aspect_extractor = ATEPC.AspectExtractor(
    "multilingual",
    auto_device=True,
    cal_perplexity=False,
)


# ----------------------------
# 2) Cleaning + Dutch-ish sentence/clause splitting
# ----------------------------
_SENT_SPLIT = re.compile(r'(?<=[\.\!\?])\s+(?=[A-ZÁÉÍÓÚÄËÏÖÜ"“])')
_CLAUSE_SPLIT = re.compile(r"\s*(?:;|:|—)\s*|\s+(?:maar|echter|hoewel|toch)\s+", re.IGNORECASE)

def clean_news_text(t: str) -> str:
    """Light cleanup for newspaper paragraphs (tune as needed)."""
    if t is None:
        return ""
    t = str(t).replace("\t", " ")
    t = re.sub(r"\s+", " ", t).strip()

    # Drop photo credit style tails (often noisy)
    t = re.sub(r"\bFOTO\b.*?$", "", t, flags=re.IGNORECASE).strip()

    # Drop leading ALL-CAPS headers / location tags (heuristic)
    # e.g., "GEOTHERMIE ...", "ZUIDPLASPOLDER ..."
    t = re.sub(r"^[A-ZÄËÏÖÜÁÉÍÓÚ0-9\s\-]{8,}\s+", "", t).strip()

    return t

def split_dutch_sentences_and_clauses(text: str, max_words: int = 35) -> list[str]:
    """Split into sentences; if a sentence is long, split into smaller clauses."""
    text = clean_news_text(text)
    if not text:
        return []

    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    out: list[str] = []

    for s in sents:
        if len(s.split()) > max_words:
            parts = [p.strip() for p in _CLAUSE_SPLIT.split(s) if p.strip()]
            out.extend(parts if parts else [s])
        else:
            out.append(s)

    return out


# ----------------------------
# 3) ABSA prediction (batched)
# ----------------------------
def batched_predict(texts, batch_size=64, desc="Running ABSA"):
    outputs = []
    n = len(texts)

    for i in tqdm(
        range(0, n, batch_size),
        total=(n + batch_size - 1) // batch_size,
        desc=desc,
        unit="batch",
    ):
        batch = texts[i : i + batch_size]
        pred = aspect_extractor.predict(
            batch,
            print_result=False,
            save_result=False,
            ignore_error=True,
            pred_sentiment=True,
        )
        outputs.extend(pred)

    return outputs


# ----------------------------
# 4) Sentiment normalization: low confidence -> Neutral/Uncertain
# ----------------------------
def normalize_sentiment(label, conf, tau=0.60):
    if label is None:
        return None
    try:
        if conf is not None and float(conf) < tau:
            return "Neutral/Uncertain"
    except Exception:
        pass
    return label

import uuid

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=str, default=str(DEFAULT_PROJECT_DIR))
    ap.add_argument("--input-csv", type=str, default="output/text/paragraph_geothermal_ollama.csv")
    ap.add_argument("--output-csv", type=str, default="output/text/sentences_with_absa_v2.csv")
    return ap.parse_args()


def main():
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    os.chdir(project_dir)

    # ============================================================
    # PIPELINE: paragraph -> sentences -> ABSA -> normalize 
    # ============================================================
    df = pd.read_csv(args.input_csv, dtype=str)

    # keep only geothermal-related rows
    df = df[df["llm_is_geothermal"] == "YES"].reset_index(drop=True)

    # 1) prepare paragraph text
    df["absa_text"] = df["paragraph_text"].fillna("").astype(str).map(clean_news_text)

    # 2) expand to sentence-level table
    sent_records = []
    for row_id, para in tqdm(zip(df.index.tolist(), df["absa_text"].tolist()),
                             total=len(df), desc="Splitting", unit="para"):
        sents = split_dutch_sentences_and_clauses(para, max_words=35)
        if not sents:
            sent_records.append({"row_id": row_id, "sentence_id": 0, "sentence_text": ""})
        else:
            for j, s in enumerate(sents):
                sent_records.append({"row_id": row_id, "sentence_id": j, "sentence_text": s})

    sent_df = pd.DataFrame(sent_records)

    # 3) run ABSA on sentences
    sent_texts = sent_df["sentence_text"].tolist()
    sent_preds = batched_predict(sent_texts, batch_size=64, desc="ABSA on sentences")

    # 4) build long-form sentence-level ABSA table
    long_rows = []
    has_geo_relevance = "geo_relevance" in df.columns

    for (row_id, sent_id, sent_text), p in zip(
        sent_df[["row_id", "sentence_id", "sentence_text"]].itertuples(index=False, name=None),
        sent_preds
    ):
        aspects = p.get("aspect", []) or []
        sentiments = p.get("sentiment", []) or []
        positions = p.get("position", []) or []
        conf = p.get("confidence", []) or []

        for j, a in enumerate(aspects):
            c = conf[j] if j < len(conf) else None
            s = sentiments[j] if j < len(sentiments) else None
            s_norm = normalize_sentiment(s, c, tau=0.60)

            rec = {
                "row_id": row_id,
                "sentence_id": sent_id,
                "sentence_text": sent_text,
                "aspect": a,
                "sentiment": s,
                "sentiment_norm": s_norm,
                "confidence": c,
                "position": positions[j] if j < len(positions) else None,
            }
            if has_geo_relevance:
                rec["geo_relevance"] = df.at[row_id, "geo_relevance"]
            long_rows.append(rec)

    absa_sentence_long = pd.DataFrame(long_rows)

    meta_cols = [
        "llm_location",
        "llm_granularity",
        "source_file",
        "source_path",
        "newspaper",
        "date",
        "body",
        "source",
        "region_name",
    ]

    missing = [c for c in meta_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in df: {missing}")

    absa_sentence_long["sentence_uid"] = [uuid.uuid4().hex for _ in range(len(absa_sentence_long))]
    for col in meta_cols:
        absa_sentence_long[col] = absa_sentence_long["row_id"].map(df[col])

    ordered_cols = (
        [
            "sentence_uid",
            "row_id",
            "sentence_id",
            "sentence_text",
            "aspect",
            "sentiment",
            "sentiment_norm",
            "confidence",
            "position",
            "llm_location",
            "llm_granularity",
        ]
        + meta_cols
    )
    ordered_cols = [c for c in ordered_cols if c in absa_sentence_long.columns]
    absa_sentence_long = absa_sentence_long[ordered_cols]

    out_sent = Path(args.output_csv)
    out_sent.parent.mkdir(parents=True, exist_ok=True)
    absa_sentence_long.to_csv(out_sent, index=False, encoding="utf-8")
    print("Wrote:", out_sent)
    print(len(absa_sentence_long))


if __name__ == "__main__":
    main()
