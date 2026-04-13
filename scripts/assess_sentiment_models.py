#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("annotation/sentiment_model_assessment")
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_INPUT_CANDIDATES = [
    Path("output/dutch/text/sentence_sentiment_llm.csv"),
    Path("output/dutch/text/sentences_with_frames_long.csv"),
]
DEFAULT_MODELS = [
    {
        "model_id": "tabularisai/multilingual-sentiment-analysis",
        "model_slug": "tabularisai_multilingual",
        "display_name": "TabularisAI Multilingual",
        "language_scope": "multilingual",
    },
    {
        "model_id": "BramVanroy/robbert-v2-dutch-base-hebban-reviews",
        "model_slug": "bramvanroy_robbert_v2_dutch_base_hebban_reviews",
        "display_name": "BramVanroy RobBERT v2 Dutch Hebban",
        "language_scope": "dutch",
    },
    {
        "model_id": "BramVanroy/bert-base-multilingual-cased-hebban-reviews",
        "model_slug": "bramvanroy_bert_base_multilingual_cased_hebban_reviews",
        "display_name": "BramVanroy BERT multilingual Hebban",
        "language_scope": "multilingual",
    },
    {
        "model_id": "BramVanroy/bert-base-dutch-cased-hebban-reviews",
        "model_slug": "bramvanroy_bert_base_dutch_cased_hebban_reviews",
        "display_name": "BramVanroy BERT Dutch Hebban",
        "language_scope": "dutch",
    },
    {
        "model_id": "BramVanroy/xlm-roberta-base-hebban-reviews",
        "model_slug": "bramvanroy_xlm_roberta_base_hebban_reviews",
        "display_name": "BramVanroy XLM-RoBERTa base Hebban",
        "language_scope": "multilingual",
    },
    {
        "model_id": "BramVanroy/bert-base-dutch-cased-hebban-reviews5",
        "model_slug": "bramvanroy_bert_base_dutch_cased_hebban_reviews5",
        "display_name": "BramVanroy BERT Dutch Hebban v5",
        "language_scope": "dutch",
    },
    {
        "model_id": "BramVanroy/bert-base-multilingual-cased-hebban-reviews5",
        "model_slug": "bramvanroy_bert_base_multilingual_cased_hebban_reviews5",
        "display_name": "BramVanroy BERT multilingual Hebban v5",
        "language_scope": "multilingual",
    },
    {
        "model_id": "BramVanroy/robbert-v2-dutch-base-hebban-reviews5",
        "model_slug": "bramvanroy_robbert_v2_dutch_base_hebban_reviews5",
        "display_name": "BramVanroy RobBERT v2 Dutch Hebban v5",
        "language_scope": "dutch",
    },
    {
        "model_id": "oxygeneDev/sentiment-multilingual",
        "model_slug": "oxygene_dev_sentiment_multilingual",
        "display_name": "oxygeneDev Sentiment Multilingual",
        "language_scope": "multilingual",
    },
    {
        "model_id": "ZombitX64/MultiSent-E5-Pro",
        "model_slug": "zombitx64_multisent_e5_pro",
        "display_name": "ZombitX64 MultiSent E5 Pro",
        "language_scope": "multilingual",
    },
    {
        "model_id": "oralunal/sentiment",
        "model_slug": "oralunal_sentiment",
        "display_name": "oralunal Sentiment",
        "language_scope": "multilingual",
    },
    {
        "model_id": DEFAULT_OLLAMA_MODEL,
        "model_slug": "ollama_qwen2_5_7b",
        "display_name": "Ollama Qwen2.5 7B",
        "language_scope": "multilingual",
        "backend": "ollama",
    },
]
SENTIMENTS = ["negative", "neutral", "positive"]
SENTIMENT_EXAMPLES = {
    "negative": [
        "Volgens de berekeningen van verschillende adviseurs bestaat een kans op 'cosmetische schade' van gebouwen in het gebied als gevolg van mogelijke 'seismische activiteit'.",
        "Volgens de Rekenkamer zien de ministers 'de urgentie van het probleem onvoldoende in'.",
        "Een investering van negentien miljoen euro moet als verloren worden beschouwd, aldus Gedeputeerde Staten (GS).",
        "Desondanks besloot minister Stef Blok in juni om geen goedkeuring aan het project te geven omdat te veel risico's kleven aan het winnen van aardwarmte in Californië.",
        "Aardwarmte wordt gezien als een serieuze alternatieve energiebron - al klinkt er ook kritiek in verband met de risico's, met name van de boringen.",
        "Hoge investeringskosten spelen een rol, maar de nog beperkte vraag naar warmte is volgens de gemeente minstens een even grote factor. \"",
        "Wat bewoners moeten betalen, is sterk afhankelijk van subsidies, want rendabel is aardwarmte nog niet.''",
        "Drinkwaterbedrijven maken zich grote zorgen over de kwaliteit van ons drinkwater nu de ene na de andere vergunning wordt verleend om in drinkwatergebieden te speuren naar aardwarmte.",
        "Velsen De overstap naar duurzame energie in het Noordzeekanaalgebied plaatst de IJmond voor grote uitdagingen.",
        "Volgens de Rekenkamer beschermt de overheid de Nederlandse drinkwatervoorraden 'niet afdoende' tegen de risico's van het boren naar aardwarmte, zo staat in een nieuw rapport.",
        "Daarbij werken niet alleen praktische zaken, maar ook wet- en regelgeving belemmerend.",
        "Iedereen, en niet in de laatste plaats het ministerie, was zo enthousiast over geothermie dat de risico's voor mens en milieu werden vergeten.",
        "De regelgeving is behoorlijk lastig, er worden nieuwe technieken gevraagd en er is veel kennis nodig.",
    ],
    "neutral": [
        "Het ministerie van Economische Zaken, dat de vergunning verstrekt, acht de risico's zeer beperkt.",
        "Eerder gaf de gemeente Dijk en Waard aan uitvoering van dit plan haar goedkeuring.",
        "Technische en financiële haalbaarheid staan centraal in het onderzoek.",
        "Uiteindelijk moet het warmtenetwerk omgezet worden naar een duurzame bron.\"",
        "Ze kunnen wat doffe plofgeluiden horen en soms voelen ze wat trillingen.",
        "Het ministerie van Economische Zaken heeft met een definitief besluit groen licht gegeven voor het winnen van aardwarmte in de centrale aan de Leyweg.",
        "De afgelopen periode is een bijdrage geleverd aan onder meer Beleefcentrum Duurzame Energie Leven van de Wind in Wieringerwerf en Natuurcentrum De Marel bij Waalenburg op Texel.",
        "Doet GTD dat onvoldoende, dan krijgt het bedrijf problemen met het verkrijgen van een vergunning van het ministerie.",
        "De lijnen die we nu onderzoeken zijn gebieden waar we nog geen kennis van de diepe ondergrond hebben.\"",
        "De behoefte aan duurzame warmte is erg groot.",
        "Het doel is de energietransitie daar te versnellen, zodat de sectorale klimaatdoelen binnen bereik komen.",
        "Er wordt geen materiaal uit de ondergrond gehaald, onderstreept de gemeente die een eigen 'Beleidsvisie op de ondergrond Barendrecht' heeft opgesteld.",
        "Het boren en laten verrijzen van benodigde bouwwerken is niet toegestaan in dat bestemmingsplan.",
    ],
    "positive": [
        "Ook dorpen en industrieterreinen in de omgeving profiteren van deze schone energie.\"",
        "Het draagvlak voor het project is groot hier in Leeuwarden.",
        "Door de verwachting dat de Europese CO2-prijs hoger uitvalt, is er minder subsidie nodig dan verwacht voor de eerder goedgekeurde projecten.",
        "We zijn heel enthousiast over dit initiatief, dat ook grote kansen biedt voor betaalbare en duurzame warmte voor woningen en andere gebouwen.",
        "Posthouwer verwacht de goedkeuring hiervoor ieder moment.",
        "De korte afstand betekent een belangrijke besparing op de totale investeringskosten, omdat er geen kilometerslange buizen in de grond gelegd hoeven te worden.",
        "Leverancier Richard Donkers (Donkers Green Energy) is ervan overtuigd dat de investering zich binnen acht jaar terugverdient. ,,",
        "Noodzakelijk voor de financiële haalbaarheid van het project, bleek eerder al. Voordeel van geothermie is dat het niet afhankelijk is van het weer of de seizoenen, er is sprake van een constante warmtelevering.",
        "Er is nog budget voor extra onderzoek\", zegt Klut. ,,",
        "Wij geloven dat aardwarmte een van de grote kansen is voor duurzame warmte in Nederland\", zegt Japikse.",
        "Productieput Zij hebben aardwarmte op de kaart gezet en hun kennis en ervaring gedeeld met nieuwe initiatiefnemers, overheden, toeleveranciers en vele andere geïnteresseerden.",
        "Dankzij de 1,5 miljoen euro van Kansen voor West voor aardwarmteproject Polanen wordt de financiering van het uitgebreide warmtenet flink geholpen.",
        "Dankzij deze innovatieve samenwerking wordt de wijk voorzien van duurzame warmte uit de buurt. \"",
        "BOLSWARD Ze zochten naar draagvlak en ze lijken het te krijgen: de Stichting Ontwikkeling Geothermie Friesland (STOGEF) hield vorige week een informatiebijeenkomst om belangstellenden te vertellen over hun initiatief om Bolsward te verwarmen door middel van geothermie.",
        "De brancheorganisatie vindt SDE++ het belangrijkste instrument voor de stimulering van duurzame energie.",
    ],
}
OLLAMA_SYSTEM_PROMPT = (
    "You are a careful sentiment classification assistant for Dutch newspaper sentences about geothermal energy. "
    "You must return a single sentiment label and a short rationale in valid JSON only."
)


def resolve_model_configs(ollama_model: str) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for model_cfg in DEFAULT_MODELS:
        current = dict(model_cfg)
        if current.get("backend") == "ollama":
            current["model_id"] = ollama_model
            current["display_name"] = f"Ollama {ollama_model}"
        resolved.append(current)
    return resolved


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Sample Dutch sentences, assign them across annotators with overlap, and run sentiment models."
    )
    ap.add_argument("--project-dir", type=str, default=str(DEFAULT_PROJECT_DIR))
    ap.add_argument("--input-csv", type=str, default="")
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--annotations-per-annotator", type=int, default=150)
    ap.add_argument("--overlap-sentences", type=int, default=30)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--ollama-url", type=str, default=DEFAULT_OLLAMA_URL)
    ap.add_argument("--ollama-model", type=str, default=DEFAULT_OLLAMA_MODEL)
    ap.add_argument("--ollama-timeout", type=int, default=120)
    return ap.parse_args()


def resolve_input_csv(explicit_path: str) -> Path:
    if explicit_path:
        candidate = Path(explicit_path)
        if not candidate.exists():
            raise FileNotFoundError(f"Input CSV not found: {candidate}")
        return candidate

    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate

    tried = ", ".join(str(path) for path in DEFAULT_INPUT_CANDIDATES)
    raise FileNotFoundError(f"Could not find an input CSV. Tried: {tried}")


def normalize_sentiment_label(label: str, model_id: str) -> str:
    value = str(label or "").strip().lower()
    collapsed = re.sub(r"[\s_\-/]+", " ", value)

    direct_map = {
        "negative": "negative",
        "neg": "negative",
        "label 0": "negative",
        "1 star": "negative",
        "1 stars": "negative",
        "2 star": "negative",
        "2 stars": "negative",
        "neutral": "neutral",
        "neu": "neutral",
        "label 1": "neutral",
        "3 star": "neutral",
        "3 stars": "neutral",
        "positive": "positive",
        "pos": "positive",
        "label 2": "positive",
        "4 star": "positive",
        "4 stars": "positive",
        "5 star": "positive",
        "5 stars": "positive",
    }
    if collapsed in direct_map:
        return direct_map[collapsed]

    star_match = re.search(r"([1-5])", collapsed)
    if star_match:
        stars = int(star_match.group(1))
        if stars <= 2:
            return "negative"
        if stars == 3:
            return "neutral"
        return "positive"

    if "neg" in collapsed:
        return "negative"
    if "neu" in collapsed:
        return "neutral"
    if "pos" in collapsed:
        return "positive"

    if collapsed.startswith("label "):
        if collapsed.endswith("0"):
            return "negative"
        if collapsed.endswith("1"):
            return "neutral"
        if collapsed.endswith("2"):
            return "positive"

    raise ValueError(f"Unsupported label '{label}' for model '{model_id}'.")


def make_sample_id(row: pd.Series) -> str:
    sentence_uid = str(row.get("sentence_uid", "")).strip()
    if sentence_uid:
        return sentence_uid

    base = "||".join(
        [
            str(row.get("source", "")),
            str(row.get("document_title", "")),
            str(row.get("publish_date", "")),
            str(row.get("paragraph_uid", "")),
            str(row.get("sentence_id", "")),
            str(row.get("sentence_text", "")),
        ]
    )
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def prepare_source_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.loc[:, ~df.columns.duplicated()].copy()
    if "sentence_text" not in out.columns:
        raise ValueError("Expected 'sentence_text' column in the input CSV.")

    out = out.dropna(subset=["sentence_text"]).copy()
    out["sentence_text"] = out["sentence_text"].astype(str).str.strip()
    out = out[out["sentence_text"] != ""].copy()

    if "paragraph_text" not in out.columns:
        out["paragraph_text"] = out["sentence_text"]
    if "document_title" not in out.columns and "title" in out.columns:
        out["document_title"] = out["title"]
    if "source" not in out.columns and "newspaper" in out.columns:
        out["source"] = out["newspaper"]
    if "publish_date" not in out.columns and "date_parsed" in out.columns:
        out["publish_date"] = out["date_parsed"]

    out["sample_id"] = out.apply(make_sample_id, axis=1)
    out = out.drop_duplicates(subset=["sample_id"]).copy()

    baseline_col = None
    for candidate in ["sentiment_norm", "sentiment"]:
        if candidate in out.columns:
            baseline_col = candidate
            break
    if baseline_col is not None:
        out["baseline_sentiment"] = out[baseline_col].astype(str).str.strip().str.lower()
        out.loc[~out["baseline_sentiment"].isin(SENTIMENTS), "baseline_sentiment"] = "unknown"
    else:
        out["baseline_sentiment"] = "unknown"

    return out


def sample_rows(df: pd.DataFrame, sample_size: int, random_seed: int) -> pd.DataFrame:
    if len(df) < sample_size:
        raise ValueError(f"Requested {sample_size} unique sentences, but only found {len(df)} usable rows.")

    usable = df.copy()
    stratified = usable[usable["baseline_sentiment"].isin(SENTIMENTS)].copy()
    groups = [group for _, group in stratified.groupby("baseline_sentiment", sort=False) if not group.empty]

    sampled_parts: list[pd.DataFrame] = []
    if groups:
        per_group = max(1, sample_size // len(groups))
        for offset, group in enumerate(groups):
            take = min(len(group), per_group)
            sampled_parts.append(group.sample(n=take, random_state=random_seed + offset))

    sampled = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else usable.head(0).copy()
    remaining_n = sample_size - len(sampled)
    if remaining_n > 0:
        already = set(sampled["sample_id"].tolist())
        remainder = usable[~usable["sample_id"].isin(already)].copy()
        sampled = pd.concat(
            [sampled, remainder.sample(n=remaining_n, random_state=random_seed + 999)],
            ignore_index=True,
        )

    sampled = sampled.sample(frac=1.0, random_state=random_seed + 2024).reset_index(drop=True)
    return sampled


def assign_annotators(
    sampled: pd.DataFrame,
    annotations_per_annotator: int,
    overlap_sentences: int,
    random_seed: int,
) -> pd.DataFrame:
    if overlap_sentences > annotations_per_annotator:
        raise ValueError("overlap_sentences cannot exceed annotations_per_annotator.")

    exclusive_per_annotator = annotations_per_annotator - overlap_sentences
    needed_unique = overlap_sentences + (2 * exclusive_per_annotator)
    if len(sampled) != needed_unique:
        raise ValueError(f"Expected {needed_unique} sampled rows, found {len(sampled)}.")

    out = sampled.copy().reset_index(drop=True)
    out["annotation_id"] = range(1, len(out) + 1)
    out["assignment_group"] = None
    out.loc[: overlap_sentences - 1, "assignment_group"] = "overlap"
    out.loc[overlap_sentences : overlap_sentences + exclusive_per_annotator - 1, "assignment_group"] = "dekker_only"
    out.loc[overlap_sentences + exclusive_per_annotator :, "assignment_group"] = "egberink_only"

    out["assigned_to_dekker"] = out["assignment_group"].isin(["overlap", "dekker_only"])
    out["assigned_to_egberink"] = out["assignment_group"].isin(["overlap", "egberink_only"])

    dekker_ids = out.loc[out["assigned_to_dekker"], "sample_id"].sample(
        frac=1.0, random_state=random_seed + 17
    ).tolist()
    egberink_ids = out.loc[out["assigned_to_egberink"], "sample_id"].sample(
        frac=1.0, random_state=random_seed + 29
    ).tolist()

    dekker_order = {sample_id: order for order, sample_id in enumerate(dekker_ids, start=1)}
    egberink_order = {sample_id: order for order, sample_id in enumerate(egberink_ids, start=1)}
    out["annotation_order_dekker"] = out["sample_id"].map(dekker_order)
    out["annotation_order_egberink"] = out["sample_id"].map(egberink_order)
    return out


def load_classifier(model_id: str):
    try:
        return pipeline(task="sentiment-analysis", model=model_id, tokenizer=model_id)
    except Exception as exc:
        message = str(exc)
        if "endswith" in message or "convert_slow_tokenizer" in message:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
            return pipeline(task="sentiment-analysis", model=model_id, tokenizer=tokenizer)
        offline_markers = ["Failed to resolve", "MaxRetryError", "NameResolutionError", "ConnectionError"]
        if any(marker in message for marker in offline_markers):
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_id, local_files_only=True)
            return pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)
        raise


def format_few_shot_examples() -> str:
    lines: list[str] = []
    for sentiment in SENTIMENTS:
        lines.append(f"{sentiment.upper()} examples:")
        for index, sentence in enumerate(SENTIMENT_EXAMPLES[sentiment], start=1):
            lines.append(f"{index}. {sentence}")
        lines.append("")
    return "\n".join(lines).strip()


def build_ollama_prompt(text: str) -> str:
    examples_block = format_few_shot_examples()
    return f"""
Task: Classify the sentiment of this Dutch newspaper sentence about geothermal energy.

Use only these labels:
- negative
- neutral
- positive

Interpretation rules:
- negative: emphasizes risk, costs, obstacles, criticism, harm, uncertainty, or failure
- neutral: mainly factual, procedural, descriptive, or mixed without clear evaluative polarity
- positive: emphasizes benefits, support, progress, feasibility, opportunity, or success

Use the following labeled examples as guidance:
{examples_block}

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


def parse_ollama_sentiment_response(raw_response: str, model_id: str) -> dict[str, Any]:
    candidate = extract_json_object(raw_response)
    if candidate is not None:
        try:
            obj = json.loads(candidate)
            raw_label = str(obj.get("sentiment", "") or "").strip()
            return {
                "predicted_sentiment": normalize_sentiment_label(raw_label, model_id=model_id),
                "raw_label": raw_label,
                "confidence": max(0.0, min(1.0, float(obj.get("confidence", 0.0) or 0.0))),
                "raw_response": raw_response,
                "rationale_short": str(obj.get("rationale_short", "") or "").strip(),
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    lowered = raw_response.lower()
    label_match = re.search(r"\b(negative|neutral|positive)\b", lowered)
    if label_match:
        raw_label = label_match.group(1)
        score_match = re.search(r"(?:confidence|score)\s*[:=]?\s*([01](?:\.\d+)?)", lowered)
        score = float(score_match.group(1)) if score_match else 0.0
        return {
            "predicted_sentiment": normalize_sentiment_label(raw_label, model_id=model_id),
            "raw_label": raw_label,
            "confidence": max(0.0, min(1.0, score)),
            "raw_response": raw_response,
            "rationale_short": "",
        }

    raise ValueError(f"Ollama did not return a parseable sentiment label: {raw_response!r}")


def call_ollama_sentiment(
    text: str,
    model_id: str,
    ollama_url: str,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model_id,
        "system": OLLAMA_SYSTEM_PROMPT,
        "prompt": build_ollama_prompt(text),
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
                f"Ollama model '{model_id}' was not found at {ollama_url}. "
                f"Pull it first with: ollama pull {model_id}"
            ) from exc
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Ollama at {ollama_url}: {exc.reason}") from exc

    raw_response = str(body.get("response", "") or "").strip()
    return parse_ollama_sentiment_response(raw_response=raw_response, model_id=model_id)


def classify_texts(
    classifier: Any,
    texts: list[str],
    model_id: str,
    batch_size: int,
    max_length: int,
) -> list[dict[str, Any]]:
    raw_predictions = classifier(
        texts,
        batch_size=batch_size,
        truncation=True,
        max_length=max_length,
    )
    rows: list[dict[str, Any]] = []
    for prediction in raw_predictions:
        raw_label = str(prediction.get("label", "") or "").strip()
        score = float(prediction.get("score", 0.0) or 0.0)
        rows.append(
            {
                "predicted_sentiment": normalize_sentiment_label(raw_label, model_id=model_id),
                "raw_label": raw_label,
                "confidence": max(0.0, min(1.0, score)),
            }
        )
    return rows


def classify_texts_with_ollama(
    texts: list[str],
    model_id: str,
    ollama_url: str,
    timeout: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    progress = tqdm(
        texts,
        desc=f"Ollama sentiment ({model_id})",
        unit="sentence",
    )
    for text in progress:
        rows.append(
            call_ollama_sentiment(
                text=text,
                model_id=model_id,
                ollama_url=ollama_url,
                timeout=timeout,
            )
        )
    return rows


def build_predictions_long(
    sampled: pd.DataFrame,
    model_configs: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
    ollama_url: str,
    ollama_timeout: int,
) -> pd.DataFrame:
    base_cols = [
        "annotation_id",
        "sample_id",
        "sentence_uid",
        "sentence_text",
        "paragraph_text",
        "document_title",
        "source",
        "publish_date",
        "matched_categories_str",
        "matched_keywords_str",
        "baseline_sentiment",
        "assignment_group",
        "assigned_to_dekker",
        "assigned_to_egberink",
        "annotation_order_dekker",
        "annotation_order_egberink",
    ]
    for column in base_cols:
        if column not in sampled.columns:
            sampled[column] = None

    texts = sampled["sentence_text"].astype(str).tolist()
    prediction_frames: list[pd.DataFrame] = []

    for model_cfg in model_configs:
        model_id = model_cfg["model_id"]
        print(f"Running {model_id} ...")
        model_df = sampled[base_cols].copy()
        model_df["model_id"] = model_id
        model_df["model_slug"] = model_cfg["model_slug"]
        model_df["model_display_name"] = model_cfg["display_name"]
        model_df["language_scope"] = model_cfg["language_scope"]
        model_df["predicted_sentiment"] = pd.Series([None] * len(model_df), dtype="object")
        model_df["raw_label"] = pd.Series([None] * len(model_df), dtype="object")
        model_df["confidence"] = pd.Series([float("nan")] * len(model_df), dtype="float64")
        model_df["model_status"] = pd.Series([None] * len(model_df), dtype="object")
        model_df["model_error"] = pd.Series([None] * len(model_df), dtype="object")
        try:
            if model_cfg.get("backend") == "ollama":
                predictions = classify_texts_with_ollama(
                    texts=texts,
                    model_id=model_id,
                    ollama_url=ollama_url,
                    timeout=ollama_timeout,
                )
            else:
                classifier = load_classifier(model_id)
                predictions = classify_texts(
                    classifier=classifier,
                    texts=texts,
                    model_id=model_id,
                    batch_size=batch_size,
                    max_length=max_length,
                )
            model_df["predicted_sentiment"] = [row["predicted_sentiment"] for row in predictions]
            model_df["raw_label"] = [row["raw_label"] for row in predictions]
            model_df["confidence"] = [row["confidence"] for row in predictions]
            model_df["model_status"] = "ok"
            model_df["model_error"] = None
        except Exception as exc:
            print(f"Model failed: {model_id} -> {exc!r}")
            model_df["model_status"] = "error"
            model_df["model_error"] = repr(exc)
        prediction_frames.append(model_df)

    return pd.concat(prediction_frames, ignore_index=True)


def build_predictions_wide(
    sampled: pd.DataFrame,
    predictions_long: pd.DataFrame,
    model_configs: list[dict[str, Any]],
) -> pd.DataFrame:
    metadata_cols = [
        "annotation_id",
        "sample_id",
        "sentence_uid",
        "sentence_text",
        "paragraph_text",
        "document_title",
        "source",
        "publish_date",
        "matched_categories_str",
        "matched_keywords_str",
        "baseline_sentiment",
        "assignment_group",
        "assigned_to_dekker",
        "assigned_to_egberink",
        "annotation_order_dekker",
        "annotation_order_egberink",
    ]
    metadata_cols = [column for column in metadata_cols if column in sampled.columns]
    wide = sampled[metadata_cols].copy()

    for model_cfg in model_configs:
        current = predictions_long[predictions_long["model_slug"] == model_cfg["model_slug"]].copy()
        current = current[
            [
                "sample_id",
                "predicted_sentiment",
                "raw_label",
                "confidence",
                "model_status",
                "model_error",
            ]
        ].rename(
            columns={
                "predicted_sentiment": f"{model_cfg['model_slug']}__sentiment",
                "raw_label": f"{model_cfg['model_slug']}__raw_label",
                "confidence": f"{model_cfg['model_slug']}__confidence",
                "model_status": f"{model_cfg['model_slug']}__status",
                "model_error": f"{model_cfg['model_slug']}__error",
            }
        )
        wide = wide.merge(current, on="sample_id", how="left")

    return wide


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    os.chdir(project_dir)

    if args.overlap_sentences < 0:
        raise ValueError("--overlap-sentences must be >= 0.")
    if args.annotations_per_annotator <= 0:
        raise ValueError("--annotations-per-annotator must be > 0.")

    unique_sample_size = (2 * args.annotations_per_annotator) - args.overlap_sentences
    input_csv = resolve_input_csv(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_configs = resolve_model_configs(args.ollama_model)

    source_df = prepare_source_df(pd.read_csv(input_csv))
    sampled = sample_rows(source_df, sample_size=unique_sample_size, random_seed=args.random_seed)
    sampled["sentence_uid"] = sampled.get("sentence_uid", sampled["sample_id"])
    sampled = assign_annotators(
        sampled=sampled,
        annotations_per_annotator=args.annotations_per_annotator,
        overlap_sentences=args.overlap_sentences,
        random_seed=args.random_seed,
    )

    annotation_path = output_dir / "sentences_for_annotation.csv"
    sampled.to_csv(annotation_path, index=False, encoding="utf-8")
    print(f"Wrote annotation set: {annotation_path} (unique sentences={len(sampled)})")
    print(
        "Assignments: "
        f"Dekker={int(sampled['assigned_to_dekker'].sum())}, "
        f"Egberink={int(sampled['assigned_to_egberink'].sum())}, "
        f"overlap={int((sampled['assignment_group'] == 'overlap').sum())}"
    )

    predictions_long = build_predictions_long(
        sampled=sampled,
        model_configs=model_configs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        ollama_url=args.ollama_url,
        ollama_timeout=args.ollama_timeout,
    )
    predictions_wide = build_predictions_wide(
        sampled=sampled,
        predictions_long=predictions_long,
        model_configs=model_configs,
    )

    long_path = output_dir / "model_predictions_long.csv"
    wide_path = output_dir / "model_predictions_wide.csv"
    models_path = output_dir / "models.json"

    predictions_long.to_csv(long_path, index=False, encoding="utf-8")
    predictions_wide.to_csv(wide_path, index=False, encoding="utf-8")
    models_path.write_text(json.dumps(model_configs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote long predictions: {long_path}")
    print(f"Wrote wide predictions: {wide_path}")
    print(f"Wrote model metadata: {models_path}")


if __name__ == "__main__":
    main()
