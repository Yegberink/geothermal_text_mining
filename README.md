# Geothermal Snakemake Workflow

This repository now has a Snakemake workflow that connects the main text-mining steps:

1. `scripts/preprocess_rtf_to_paragraphs.py`
2. `scripts/locations_ollama.py`
3. `scripts/is_geothermal.py`
4. `scripts/ABSA.py`
5. `scripts/geocoding_offline.py`
6. `scripts/classification_sentences.py`
7. `scripts/geographic_aggregation.py`

Snakemake will only rerun the steps needed to build the output you request.

## Main files

- `Snakefile`: workflow definition
- `config/config.yaml`: paths, Ollama settings, and layer names
- `pixi.toml`: environment definition, now including `snakemake-minimal`

## Install

Refresh the Pixi environment so `snakemake` is available:

```bash
pixi install
```

## Run

Dry-run the full DAG:

```bash
snakemake -n
```

Build the default final targets:

```bash
snakemake --cores 1
```

Build a specific later-stage file and let Snakemake infer all prerequisites:

```bash
snakemake --cores 1 output/text/sentences_with_absa_and_geo_v2.gpkg
```

Build the administrative aggregation output:

```bash
snakemake --cores 1 output/text/sentences_with_categories_admin.csv
```

## Notes

- The Ollama-based rules require a local Ollama server and the configured model(s) in `config/config.yaml`.
- The ABSA step uses `pyabsa` and the existing checkpoint/model setup already present in the project.
- Some steps write cache/checkpoint files such as `geo_checkpoint.parquet` and `geo_cache.jsonl`; those are treated as runtime artifacts rather than final targets, and when `language` is set they are written under `cache/{language}`.
- To run a specific language subfolder, set `language: dutch` in `config/config.yaml`. This makes the workflow read from `input_data/dutch` and write outputs to `output/dutch/...`.
- If you change output names, cache locations, or input locations, update `config/config.yaml` instead of editing the `Snakefile`.

## Current default target

Running `snakemake` with no explicit target builds:

- `output/text/sentences_with_categories_admin.csv`
- `output/text/sentences_with_categories_admin.gpkg`
