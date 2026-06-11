# Geothermal Text Mining Workflow

This repository contains a Snakemake workflow for mining geothermal-related newspaper text, extracting and geocoding paragraph-level locations, assigning frames and sentiment at sentence level, aggregating the results to language-specific administrative areas, and preparing annotation data for manual review.

The workflow is paragraph-based for keyword prefiltering, geothermal relevance, location extraction, and geocoding. It is sentence-based for frame detection, sentiment, visualisation inputs, and annotation. Sentence rows inherit the paragraph-level location and geometry fields.

## Current workflow

The current `Snakefile` runs the following pipeline:

1. `scripts/preprocess_rtf_to_paragraphs.py`
   Converts raw `.rtf` newspaper files into cleaned article and paragraph tables.
2. `scripts/update_keywords_framework.py`
   Builds the effective keyword framework from the base vocabulary and accepted review additions.
3. `scripts/filter_paragraphs_by_keywords.py`
   Keeps only paragraphs that mention at least one keyword from the effective framework.
4. `scripts/is_geothermal.py`
   Uses Ollama to classify whether each paragraph is mainly about geothermal energy.
5. `scripts/locations_ollama.py`
   Uses Ollama to extract the primary location discussed in each geothermal paragraph.
6. `scripts/geocoding_offline.py`
   Matches the extracted paragraph location against municipality and province layers locally.
7. `scripts/geocoding_online.py`
   Optionally enriches unresolved or fine-grained paragraph locations through online geocoding.
8. `scripts/split_paragraphs_to_sentences.py`
   Splits geocoded paragraphs into sentence-level rows while keeping paragraph context and paragraph-level location/geometry output.
9. `scripts/classification_sentences.py`
   Runs keyword-based frame matching on the sentence table and keeps only sentences with at least one matched frame.
10. `scripts/sentiment_classification.py`
   Runs a local Ollama sentiment classifier on the frame-bearing sentences.
   Duplicate `sentence_text` values are cached locally to avoid repeated model calls.
11. `scripts/classification_sentences.py`
   Builds the geocoded sentence-level frame outputs from inherited paragraph geometry and exports long/short tables plus a GeoPackage.
12. `scripts/geographic_aggregation.py`
    Adds municipality and province labels to the sentence-level geocoded outputs.
13. `scripts/visualize_absa_results.py`
    Produces the province-level and category-level sentiment figures plus an interactive HTML location map.
14. `scripts/make_annotation_df.py`
    Builds the sentence-level annotation CSV used by the Streamlit annotation app.

In short, the current workflow is:

`RTF files -> cleaned paragraphs -> paragraph keyword filter -> geothermal paragraph classification -> paragraph location extraction -> paragraph geocoding -> sentence split with inherited geo fields -> sentence frame matching -> sentence sentiment -> sentence frame outputs -> admin aggregation -> figures + interactive map + annotation export`

## Main files

- `Snakefile`
  Workflow definition.
- `config/config.yaml`
  Central configuration for paths, active app language, workflow languages, language-specific geography labels, Ollama models, geocoding settings, and annotation export filters.
- `pixi.toml`
  Environment definition for Python, Snakemake, geopandas, transformers, torch, plotly, and the rest of the pipeline dependencies.

## Inputs

The workflow expects one folder per language:

- raw newspaper `.rtf` files in `input_data/{language}/`
- a newspaper-region mapping CSV in `data/{language}/newspaper_region_mapping.csv`
- municipality and province layers in `data/{language}/`
- a frame/topic keyword file in `vocab/{language}/keywords_topics.csv`

With `languages: auto`, Snakemake discovers every `vocab/{language}/keywords_topics.csv` that also has `input_data/{language}/`.

## Key outputs

The main outputs are written under `output/{language}/` for every discovered workflow language.
Cross-language overview figures are written under `output/figures/`.

### Main intermediate outputs

- `output/dutch/text/newspapers_cleaned_paragraphs.csv`
- `output/dutch/text/newspapers_keyword_filtered_paragraphs.csv`
- `output/dutch/text/paragraph_geothermal_ollama.csv`
- `output/dutch/text/paragraph_locations_ollama.csv`
- `output/dutch/text/paragraphs_with_geo.csv`
- `output/dutch/text/sentence_locations_ollama.csv`
- `output/dutch/text/sentences_with_frames_long.csv`
- `output/dutch/text/sentence_sentiment_llm.csv`
- `output/dutch/text/sentences_with_categories.gpkg`
- `output/dutch/text/sentences_with_categories_admin.csv`

### Default final targets

Running `snakemake` with no explicit target builds these outputs for every discovered language:

- `output/{language}/text/sentences_with_categories_admin.csv`
- `output/{language}/text/sentences_with_categories_admin.gpkg`
- `output/{language}/figures/province_sentiment_table.csv`
- `output/{language}/figures/provinces_sentiment_balance.png`
- `output/{language}/figures/provinces_sentiment_distribution.png`
- `output/{language}/figures/categories_sentiment_distribution.png`
- `output/{language}/figures/locations_map.html`
- `output/figures/all_languages_province_sentiment_table.csv`
- `output/figures/all_languages_province_sentiment_balance.png`
- `output/figures/all_languages_frames_sentiment_table.csv`
- `output/figures/all_languages_frames_sentiment_distribution.png`
- `output/figures/all_languages_frames_country_sentiment_balance_table.csv`
- `output/figures/all_languages_frames_country_sentiment_balance.png`
- `output/figures/all_languages_province_sentiment_map.png`

If `make_annotation_df: true`, it also builds:

- `annotation/{language}/sentences_for_annotation.csv`
- `annotation/sentences_for_annotation_all_languages.csv`

By default each language-specific evaluation file contains separate random samples for each evaluation: `500` sentence rows for frame identification, `500` sentence rows for sentiment classification, `100` paragraph rows for geothermal relevance, and `100` geothermal paragraph rows for location extraction.

## Models used

### Ollama

The current workflow uses Ollama for:

- paragraph-level geothermal relevance classification
- paragraph-level primary location extraction
- sentence-level sentiment classification

Paragraph-level models are configured in `config/config.yaml` under `ollama`.

Sentence-level sentiment is currently handled by:

- `llama3.1:8b`

using a zero-shot prompt. This is configured in `config/config.yaml` under `sentiment_ollama`.

## Geographic scope

The workflow separates text language from geographic scope. For German-language text, `config/config.yaml` uses `German-speaking countries` and online geocoding is biased to `de,at,ch,li`. The currently available German admin layers are still Germany-specific; Austrian, Swiss, or Liechtenstein locations may geocode online but will not receive full region polygons until matching region data is added under `data/german/`.

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

This runs all languages discovered from `vocab/` and creates the combined annotation CSV at the end.

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

It uses the generated mixed sentence/paragraph evaluation CSV and stores reviewer answers in a local SQLite database.

Run it with:

```bash
streamlit run annotation/app.py
```

The current annotation flow supports four selectable evaluations:

- paragraph geothermal relevance
- paragraph location extraction
- sentence frame correctness
- sentence sentiment correctness
- configurable workflow questions

The app lets you select the language and evaluation, shows each sampled sentence or paragraph with the relevant model output, and asks the review question for that evaluation. It reports accuracy for the selected queue from the saved annotations. Frame-identification corrections are also exported to the same keyword-review CSV used by `scripts/update_keywords_framework.py`, so the framework can be improved from the evaluation workflow.

The app uses `annotation/{language}/sentences_for_annotation.csv` when available, and can also read language rows from `annotation/sentences_for_annotation_all_languages.csv`.

Additional workflow questions can be added under `annotation.workflow_questions` in `config/config.yaml`. Supported question types are `yes_no`, `select`, `radio`, `multiselect`, `text`, and `textarea`; answers are saved in the annotation database as `workflow_answers_json`.

```yaml
annotation:
  workflow_questions:
    - id: article_level_match
      prompt: "Does the workflow identify the correct article-level geothermal context?"
      type: yes_no
      required: true
    - id: failure_reason
      prompt: "If the workflow output is wrong, what is the main failure type?"
      type: select
      required: false
      options:
        - Keyword filter
        - Geothermal classification
        - Frame matching
        - Sentiment classification
```

## Sentiment model assessment

A separate sentiment study workflow is available under `annotation/sentiment_model_assessment/`.

It samples Dutch sentence-level records from the existing workflow, runs a local Ollama sentiment comparison set, assigns the sampled sentences across `Dekker` and `Egberink` with overlap, and evaluates model performance afterward in a notebook.

Generate the comparison set:

```bash
python scripts/assess_sentiment_models.py
```

Use specific local Ollama models for the comparison run:

```bash
ollama pull qwen2.5:14b
ollama pull llama3.1:8b
ollama pull mistral
ollama pull phi3
python scripts/assess_sentiment_models.py --ollama-models qwen2.5:14b,llama3.1:8b,mistral,phi3
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
- The geothermal relevance, location extraction, and sentence sentiment steps require a local Ollama server and the configured models.
- If you want to change file names or locations, update `config/config.yaml` instead of editing the `Snakefile`.

## Notes on outputs

- The paragraph keyword filter is only an inclusion filter; frame labels are still assigned at sentence level.
- The sentence-level frame filter is applied before sentiment, so only sentences with a matched frame are sent to the sentiment classifier.
- Sentiment caching uses `sentence_text` together with the configured model and prompt variant to reduce repeated model calls.
- The location output in `output/{language}/figures/locations_map.html` is interactive:
  hover shows the matched location and clicking a point reveals the associated text.
