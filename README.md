# Geothermal Text Mining Workflow

This repository contains a Snakemake workflow for mining geothermal-related newspaper text, extracting and geocoding paragraph-level locations, assigning frames and sentiment at sentence level, aggregating the results to language-specific administrative areas, and preparing annotation data for manual review.

The workflow is paragraph-based for geothermal relevance, location extraction, and geocoding. It is sentence-based for frame detection, sentiment, visualisation inputs, and annotation. Sentence rows inherit the paragraph-level location and geometry fields. Sentiment covers every sentence from a paragraph labelled `YES` with point or polygon geometry; frame keywords do not determine sentiment eligibility.

## Current workflow

The current `Snakefile` runs the following pipeline:

1. `scripts/core_workflow/preprocess_rtf_to_paragraphs.py`
   Converts raw `.rtf` newspaper files into cleaned article and paragraph tables.
2. `scripts/core_workflow/filter_paragraphs_by_keywords.py`
   Remains available as a legacy standalone keyword-filtered paragraph export. Neither sentiment nor the compact tracking table depends on it.
3. `scripts/core_workflow/is_geothermal.py`
   Reads cleaned paragraphs directly and uses Ollama to classify whether each paragraph is mainly about geothermal energy, retaining the existing 20–499-word restriction.
4. `scripts/core_workflow/locations_ollama.py`
   Keeps only paragraphs labelled `YES` and uses Ollama to extract the primary location discussed in each geothermal paragraph.
5. `scripts/core_workflow/geocoding_offline.py`
   Matches the extracted paragraph location against `data/shapes.parquet` NUTS2/country shapes.
6. `scripts/core_workflow/geocoding_online.py`
   Finalises paragraph geocoding offline: language-specific manual overrides, GeoNames country gazetteers, and any previously filled geocoder cache are applied to remaining unmatched locations. The step also writes unmatched-location and suggestion reports for review.
7. `scripts/core_workflow/split_paragraphs_to_sentences.py`
   Requires a nonempty point or polygon geometry, then splits each eligible paragraph into all its sentences while keeping paragraph context and location/geometry output.
8. `scripts/core_workflow/classification_sentences.py`
   Runs keyword-based frame matching on a separate reporting branch and keeps only sentences with at least one matched frame. This does not filter the sentiment input.
9. `scripts/core_workflow/sentiment_classification.py`
   Reads the sentence splitter output directly and runs a local Ollama sentiment classifier on every sentence, including sentences without frame matches.
   Duplicate `sentence_text` values are cached locally to avoid repeated model calls.
10. `scripts/core_workflow/classification_sentences.py`
   Builds the geocoded sentence-level frame outputs from inherited paragraph geometry and exports long/short tables plus a GeoPackage.
11. `scripts/core_workflow/geographic_aggregation.py`
    Assigns sentence-level geocoded outputs to NUTS2 regions from `data/shapes.parquet`.
12. `scripts/results_tracking/visualize_absa_results.py`
    Produces the province-level and category-level sentiment figures plus an interactive HTML location map.
13. `scripts/core_workflow/make_annotation_df.py`
    Builds the sentence-level annotation CSV used by the Streamlit annotation app.

The main sentiment and results path is:

`RTF files -> cleaned paragraphs -> geothermal paragraph classification (20–499 words) -> YES paragraphs -> paragraph location extraction -> shapes-parquet paragraph geocoding -> offline final geocoding -> geometry filter -> sentence split with inherited geo fields -> sentence sentiment -> sentence frame outputs -> NUTS2 aggregation -> figures + interactive map + annotation export`

Language-specific frame outputs, aggregation, visualisations, and annotation retain their existing behavior. Generic reporting uses all eligible sentence sentiments for regional balances.

## Main files

- `Snakefile`
  Workflow definition.
- `config/config.yaml`
  Central configuration for paths, active app language, workflow languages, language-specific geography labels, Ollama models, geocoding settings, and annotation export filters.
- `pixi.toml`
  Environment definition for Python, Snakemake, geopandas, transformers, torch, plotly, and the rest of the pipeline dependencies.

## Inputs

The workflow expects one folder per language:

- raw newspaper `.rtf` and `.pdf` files in `data/text_data/{language}/`
- a frame/topic keyword file in `data/vocab/{language}/keywords_topics.csv`

The shared administrative geography source is `data/shapes.parquet`, containing European country shapes and NUTS2 regions for the countries of interest. It is used for local matching, aggregation, and map visualisation. GeoNames country extracts live in `data/geonames/` for offline point matching. Public online geocoders are not contacted by the default workflow.

With `languages: auto`, Snakemake discovers every `data/vocab/{language}/keywords_topics.csv` that also has `data/text_data/{language}/`.

## Key outputs

The main outputs are written under `output/{language}/` for every discovered workflow language.
Workflow intermediates live in `output/{language}/workflow/`, result tables in `output/{language}/text/`, and figures/maps in `output/{language}/figures/`.
Cross-language outputs live under `output/generic/text/` and `output/generic/figures/`.

### Main intermediate outputs

- `output/dutch/workflow/newspapers_cleaned_paragraphs.csv`
- `output/dutch/workflow/newspapers_keyword_filtered_paragraphs.csv`
- `output/dutch/workflow/paragraph_geothermal_ollama.csv`
- `output/dutch/workflow/paragraph_locations_ollama.csv`
- `output/dutch/workflow/paragraphs_with_geo.csv`
- `output/dutch/workflow/sentence_locations_ollama.csv`
- `output/dutch/workflow/sentences_with_frames_long.csv`
- `output/dutch/workflow/sentence_sentiment_llm.csv`
- `output/dutch/workflow/sentences_with_categories.gpkg`
- `output/dutch/workflow/sentences_with_categories_admin.csv`

### Default final targets

Running `snakemake` with no explicit target builds these outputs for every discovered language:

- `output/{language}/workflow/sentences_with_categories_admin.csv`
- `output/{language}/workflow/sentences_with_categories_admin.gpkg`
- `output/{language}/text/province_sentiment_table.csv`
- `output/{language}/figures/provinces_sentiment_balance.png`
- `output/{language}/figures/provinces_sentiment_distribution.png`
- `output/{language}/figures/categories_sentiment_distribution.png`
- `output/{language}/figures/locations_map.html`
- `output/generic/text/all_languages_province_sentiment_table.csv`
- `output/generic/figures/all_languages_province_sentiment_balance.png`
- `output/generic/text/all_languages_extreme_province_frame_shares_table.csv`
- `output/generic/text/frame_mentions_100pct_stacked_table.csv`
- `output/generic/figures/frame_mentions_100pct_stacked.png`
- `output/generic/figures/all_languages_province_sentiment_map.png`

Generic figure generation produces only the three PNGs listed above. All three use the same global selection: the three lowest and three highest regional sentiment balances (positive percentage minus negative percentage), across countries, excluding country averages and regions with fewer than 40 sentences. Ties prefer more sentences, then country, region, and language alphabetically. If fewer than six regions qualify, the two tails remain disjoint. The map and balance chart highlight these regions; the stacked figure shows their frame composition, grouped by low/high sentiment. Regions with no frame mentions remain selected and have an empty composition bar.

Regional balances use all successfully classified eligible sentences, including sentences without frames. Frame composition uses matched frames only. Existing obsolete generic images are not deleted, but are no longer generated or required by the workflow.

`output/generic/text/data_tracking.csv` has ten rows, each with counts and unique-document counts by language: documents extracted from files, unique cleaned documents, total paragraphs, paragraphs after the 20–499-word filter, geothermal `YES` paragraphs, paragraphs with geometry, total sentences, sentences with successful sentiment, sentences with frames, and sentences with both frames and successful sentiment. The last row matches sentence identifiers across the frame and sentiment tables. Counts describe saved intermediate files; rerun upstream stages to reflect changes to filtering.

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

The workflow separates text language from geographic scope. `language` controls language resources and prompts; `country_scope` controls the allowed countries for location extraction, geocoding, aggregation, and country-specific figures.

German-language text uses one language scope with three canonical countries:

```yaml
country_scope:
  countries_by_language:
    german:
      - Germany
      - Austria
      - Switzerland
```

Dutch and Italian use the same structure with one country (`Netherlands` and `Italy`). Country fallback is automatic only when the scope has exactly one canonical country. In a multi-country scope, unresolved fallback rows keep `llm_location = NONE`, store the possible countries in `llm_country_candidates`, and are flagged for review instead of being geocoded as a fake country group.

Country-specific averages and country-level visualisations include only rows with one canonical country (`llm_country_assignment_type == single_country`, or an equivalent geocoded country field). Rows with multiple possible countries or no single country are intentionally excluded from country-specific averages, but they remain in the general frame-level sentiment analysis.

Migration note: replace legacy German config such as:

```yaml
country: German-speaking countries
```

with:

```yaml
country_scope:
  countries:
    - Germany
    - Austria
    - Switzerland
```

## Geocoding Overrides and Online Cache

Default geocoding first resolves locations deterministically with overrides and GeoNames, then builds one combined geocoding candidate table for any remaining locations. If `online_geocoding.provider` is set to `nominatim`, `opencage`, or `photon`, the normal Snakemake DAG fills one shared cache before final paragraph geocoding is consumed by downstream sentence splitting.

Add manual geocoding decisions to:

```text
data/vocab/{language}/location_geocoding_overrides.csv
```

The override schema is:

```text
location,action,target_type,target_name,country_id,nuts2_id,lat,lon,notes
```

Supported actions are `nuts2`, `point`, and `ignore`. After each default run, review:

- `output/{language}/text/geocoding_unmatched.csv`
- `output/{language}/text/geocoding_suggestions.csv`

The shared online cache is written to `cache/geocode_unmatched_online.jsonl`. For public Nominatim, set `online_geocoding.nominatim_policy_ack: true` after reviewing the OSMF usage policy: https://operations.osmfoundation.org/policies/nominatim/. Photon is an OSM-based public demo that may throttle or change without notice, Pelias is a self-hostable open-data geocoder, and OpenCage requires an API key.

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

This runs all languages discovered from `data/vocab/` and creates the combined annotation CSV at the end.

Build a specific target:

```bash
pixi run snakemake --cores 4 output/dutch/workflow/sentences_with_categories_admin.csv
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

The app lets you select the language and evaluation, shows each sampled sentence or paragraph with the relevant model output, and asks the review question for that evaluation. It reports accuracy for the selected queue from the saved annotations. Annotation results do not feed back into the workflow vocabulary.

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

## Configuration notes

- `language` in `config/config.yaml` rewrites `data/text_data/`, `data/vocab/`, `output/`, `cache/`, and `annotation/` paths into language-specific subfolders where appropriate.
- Runtime LLM caches are written under `cache/{language}/`, one JSONL file per script step.
- The geothermal relevance, location extraction, and sentence sentiment steps require a local Ollama server and the configured models.
- If you want to change file names or locations, update `config/config.yaml` instead of editing the `Snakefile`.

## Notes on outputs

- The legacy paragraph keyword filter remains available as a standalone target; sentence frame matching supplies reporting counts. Neither gates sentiment.
- Sentiment includes all sentences from `YES` geothermal paragraphs of 20–499 words with nonempty point or polygon geometry, regardless of frame matches.
- Sentiment caching uses `sentence_text` together with the configured model and prompt variant to reduce repeated model calls.
- The location output in `output/{language}/figures/locations_map.html` is interactive:
  hover shows the matched location and clicking a point reveals the associated text.
