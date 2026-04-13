# Geothermal Text Mining Workflow

This repository contains a Snakemake workflow for mining geothermal-related newspaper text, extracting locations, assigning frames and sentiment at sentence level, geocoding the matched locations, aggregating the results to Dutch administrative areas, and preparing annotation data for manual review.

The workflow is paragraph-based at the geothermal relevance and location-extraction stage, and sentence-based for frame detection, sentiment, geocoding outputs, visualisation inputs, and annotation.

## Current workflow

The current `Snakefile` runs the following pipeline:

1. `scripts/preprocess_rtf_to_paragraphs.py`
   Converts raw `.rtf` newspaper files into cleaned article and paragraph tables.
2. `scripts/is_geothermal.py`
   Uses Ollama to classify whether each paragraph is mainly about geothermal energy.
3. `scripts/locations_ollama.py`
   Uses Ollama to extract the primary location discussed in each geothermal paragraph.
4. `scripts/split_paragraphs_to_sentences.py`
   Splits geothermal paragraphs into sentence-level rows while keeping paragraph context and the paragraph-level location output.
5. `scripts/classification_sentences.py`
   Runs keyword-based frame matching on the sentence table and keeps only sentences with at least one matched frame.
6. `scripts/sentiment_classification.py`
   Runs a multilingual Hugging Face sentiment model on the frame-bearing sentences.
   Duplicate `sentence_text` values are deduplicated before inference and merged back afterward.
7. `scripts/geocoding_offline.py`
   Matches the extracted paragraph location against municipality and province layers locally.
8. `scripts/geocoding_online.py`
   Optionally enriches unresolved or fine-grained locations through online geocoding.
9. `scripts/classification_sentences.py`
   Rebuilds the geocoded sentence-level frame outputs and exports long and short sentence tables.
10. `scripts/geographic_aggregation.py`
    Adds municipality and province labels to the sentence-level geocoded outputs.
11. `scripts/visualize_absa_results.py`
    Produces the province-level and category-level sentiment figures plus an interactive HTML location map.
12. `scripts/make_annotation_df.py`
    Builds the sentence-level annotation CSV used by the Streamlit annotation app.

In short, the current workflow is:

`RTF files -> cleaned paragraphs -> geothermal paragraph classification -> paragraph location extraction -> sentence split -> frame matching -> sentence sentiment -> geocoding -> sentence frame outputs -> admin aggregation -> figures + interactive map + annotation export`

## Main files

- `Snakefile`
  Workflow definition.
- `config/config.yaml`
  Central configuration for paths, language, Ollama models, the Hugging Face sentiment model, geocoding settings, and annotation export filters.
- `pixi.toml`
  Environment definition for Python, Snakemake, geopandas, transformers, torch, plotly, and the rest of the pipeline dependencies.

## Inputs

The workflow expects:

- raw newspaper `.rtf` files in `input_data/{language}/`
- a newspaper-region mapping CSV in `data/{language}/newspaper_region_mapping.csv`
- municipality and province layers in `data/{language}/`
- a frame/topic keyword file in `vocab/{language}/keywords_topics.csv`

## Key outputs

With `language: dutch`, the main outputs are written under `output/dutch/`.

### Main intermediate outputs

- `output/dutch/text/newspapers_cleaned_paragraphs.csv`
- `output/dutch/text/paragraph_geothermal_ollama.csv`
- `output/dutch/text/paragraph_locations_ollama.csv`
- `output/dutch/text/sentence_locations_ollama.csv`
- `output/dutch/text/sentences_with_frames_long.csv`
- `output/dutch/text/sentence_sentiment_llm.csv`
- `output/dutch/text/sentences_with_geo.gpkg`
- `output/dutch/text/sentences_with_categories.gpkg`
- `output/dutch/text/sentences_with_categories_admin.csv`

### Default final targets

Running `snakemake` with no explicit target builds:

- `output/{language}/text/sentences_with_categories_admin.csv`
- `output/{language}/text/sentences_with_categories_admin.gpkg`
- `output/{language}/figures/province_sentiment_table.csv`
- `output/{language}/figures/provinces_sentiment_balance.png`
- `output/{language}/figures/provinces_sentiment_distribution.png`
- `output/{language}/figures/categories_sentiment_distribution.png`
- `output/{language}/figures/locations_map.html`

If `make_annotation_df: true`, it also builds:

- `annotation/{language}/sentences_for_annotation.csv`

## Models used

### Ollama

The current workflow uses Ollama for:

- paragraph-level geothermal relevance classification
- paragraph-level primary location extraction

These models are configured in `config/config.yaml` under `ollama`.

### Hugging Face sentiment model

Sentence-level sentiment is currently handled by:

- `nlptown/bert-base-multilingual-uncased-sentiment`

This is configured in `config/config.yaml` under `sentiment_hf`.

## Installation

Install the Pixi environment:

```bash
pixi install
```

## Running the workflow

Dry-run the full DAG:

```bash
pixi run snakemake -n
```

Run the default workflow:

```bash
pixi run snakemake --cores 4
```

Build a specific target:

```bash
pixi run snakemake --cores 4 output/dutch/text/sentences_with_categories_admin.csv
```

Build the annotation export only:

```bash
pixi run snakemake --cores 4 annotation/dutch/sentences_for_annotation.csv
```

## Annotation workflow

The annotation app lives in `annotation/app.py`.

It uses the generated sentence-level annotation CSV and stores reviewer answers in a local SQLite database.

Run it with:

```bash
streamlit run annotation/app.py
```

The current annotation flow supports review of:

- geothermal relevance
- sentiment correctness
- frame correctness
- location correctness

The app shows the sentence, paragraph context, predicted frame(s), predicted sentiment, and extracted location, then walks through the review questions step by step.

## Sentiment model assessment

A separate sentiment study workflow is available under `annotation/sentiment_model_assessment/`.

It samples Dutch sentence-level records from the existing workflow, runs a curated comparison set of Dutch and multilingual Hugging Face sentiment classifiers plus an optional local Ollama instruction model, assigns the sampled sentences across `Dekker` and `Egberink` with overlap, and evaluates model performance afterward in a notebook.

Generate the comparison set:

```bash
python scripts/assess_sentiment_models.py
```

Use a specific local Ollama model for the few-shot sentiment run:

```bash
ollama pull qwen2.5:14b
python scripts/assess_sentiment_models.py --ollama-model qwen2.5:14b
```

Run the sentiment-only review app:

```bash
streamlit run annotation/sentiment_model_assessment/app.py
```

Analyze model performance afterward in:

```bash
annotation/sentiment_model_assessment/analysis.ipynb
```

## Configuration notes

- `language` in `config/config.yaml` rewrites `input_data/`, `output/`, `cache/`, `data/`, `vocab/`, and `annotation/` paths into language-specific subfolders.
- Runtime cache and checkpoint files are written under `cache/{language}/`.
- The geothermal relevance and location extraction steps require a local Ollama server and the configured models.
- The first run of the Hugging Face sentiment model may need to download model weights if they are not already cached locally.
- If you want to change file names or locations, update `config/config.yaml` instead of editing the `Snakefile`.

## Notes on outputs

- The sentence-level frame filter is applied before sentiment, so only sentences with a matched frame are sent to the sentiment classifier.
- Sentiment is deduplicated on `sentence_text` before inference to reduce repeated model calls.
- The location output in `output/{language}/figures/locations_map.html` is interactive:
  hover shows the matched location and clicking a point reveals the associated text.
