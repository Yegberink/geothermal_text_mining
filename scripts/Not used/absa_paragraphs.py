
import pandas as pd
from pyabsa import AspectTermExtraction as ATEPC
from tqdm.auto import tqdm

aspect_extractor = ATEPC.AspectExtractor(
    "multilingual",
    auto_device=True,
    cal_perplexity=False,
)

df = location_paragraphs.copy()

df["absa_text"] = df["paragraph_text"].fillna("")

def batched_predict(texts, batch_size=64, desc="Running ABSA"):
    outputs = []
    n = len(texts)

    for i in tqdm(
        range(0, n, batch_size),
        total=(n + batch_size - 1) // batch_size,
        desc=desc,
        unit="batch",
    ):
        batch = texts[i:i + batch_size]
        pred = aspect_extractor.predict(
            batch,
            print_result=False,
            save_result=False,
            ignore_error=True,
            pred_sentiment=True,
        )
        outputs.extend(pred)

    return outputs

texts = df["absa_text"].tolist()
preds = batched_predict(texts, batch_size=64)

def summarize_pred(p):
    aspects = p.get("aspect", []) or []
    sentiments = p.get("sentiment", []) or []
    conf = p.get("confidence", []) or []

    triples = []
    for a, s, c in zip(aspects, sentiments, conf):
        try:
            triples.append(f"{a}:{s}({float(c):.3f})")
        except Exception:
            triples.append(f"{a}:{s}")

    return {
        "absa_aspects": aspects,
        "absa_sentiments": sentiments,
        "absa_confidence": conf,
        "absa_summary": " | ".join(triples),
        "absa_n_aspects": len(aspects),
    }

absa_df = pd.DataFrame([summarize_pred(p) for p in preds], index=df.index)

# ✅ correct column assignment
df[absa_df.columns] = absa_df

# Long-form table: one row per (paragraph, aspect)
records = []
row_ids = df.index.tolist()

has_geo_relevance = "geo_relevance" in df.columns

for idx, p in zip(row_ids, preds):
    aspects = p.get("aspect", []) or []
    sentiments = p.get("sentiment", []) or []
    positions = p.get("position", []) or []
    conf = p.get("confidence", []) or []

    for j, a in enumerate(aspects):
        rec = {
            "row_id": idx,
            "absa_text": df.at[idx, "absa_text"],
            "aspect": a,
            "sentiment": sentiments[j] if j < len(sentiments) else None,
            "confidence": conf[j] if j < len(conf) else None,
            "position": positions[j] if j < len(positions) else None,
        }
        if has_geo_relevance:
            rec["geo_relevance"] = df.at[idx, "geo_relevance"]
        records.append(rec)

absa_long = pd.DataFrame(records)

out_main = "output/text/paragraphs_with_absa.csv"
df.to_csv(out_main, index=False, encoding="utf-8")
print("Wrote:", out_main)

out_long = "output/text/paragraphs_absa_long.csv"
absa_long.to_csv(out_long, index=False, encoding="utf-8")
print("Wrote:", out_long)
