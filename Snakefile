from pathlib import Path

configfile: "config/config.yaml"

PROJECT_DIR = Path(workflow.basedir).resolve()
PYTHON = config.get("python", "python")
OLLAMA = config["ollama"]
PATHS = config["paths"]
LANGUAGE = config.get("language")
LAYERS = config["layers"]
COUNTRY = config.get("country")


def _local_path(path_str):
    path = Path(path_str)
    if LANGUAGE and len(path.parts) > 0:
        if path.parts[0] == "output":
            return str(Path("output") / LANGUAGE / Path(*path.parts[1:]))
        if path.parts[0] == "cache":
            return str(Path("cache") / LANGUAGE / Path(*path.parts[1:]))
        if path.parts[0] == "input_data":
            return str(Path("input_data") / LANGUAGE / Path(*path.parts[1:]))
    return str(path)


if LANGUAGE:
    PATHS = {key: _local_path(value) for key, value in PATHS.items()}


rule all:
    input:
        PATHS["sentences_with_categories_admin_csv"],
        PATHS["sentences_with_categories_admin_gpkg"],


rule preprocess_rtf_to_paragraphs:
    input:
        input_dir=PATHS["input_rtf_dir"],
        regions=PATHS["newspaper_regions_csv"],
    output:
        paragraphs=PATHS["paragraphs_csv"],
        articles=PATHS["articles_csv"],
    shell:
        """
        {PYTHON} scripts/preprocess_rtf_to_paragraphs.py \
          --project-dir {PROJECT_DIR} \
          --input-rtf-dir {input.input_dir} \
          --newspaper-region-csv {input.regions} \
          --output-paragraph-csv {output.paragraphs} \
          --output-articles-csv {output.articles} \
          --write-articles-csv
        """


rule classify_geothermal:
    input:
        PATHS["paragraphs_csv"],
    output:
        csv=PATHS["paragraph_geothermal_csv"],
    params:
        checkpoint=PATHS["geo_class_checkpoint"],
        cache=PATHS["geo_class_cache"],
        partial=PATHS["geo_class_partial_csv"],
        model=OLLAMA["geothermal_model"],
        url=OLLAMA["url"],
        sleep_s=OLLAMA["sleep_s"],
        save_every=OLLAMA["save_every"],
    shell:
        """
        {PYTHON} scripts/is_geothermal.py \
          --project-dir {PROJECT_DIR} \
          --input-csv {input} \
          --out-csv {output.csv} \
          --checkpoint {params.checkpoint} \
          --cache {params.cache} \
          --partial-csv {params.partial} \
          --ollama-url {params.url} \
          --model {params.model} \
          --country {COUNTRY} \
          --sleep-s {params.sleep_s} \
          --save-every {params.save_every}
        """


rule extract_locations:
    input:
        PATHS["paragraph_geothermal_csv"],
    output:
        csv=PATHS["paragraph_locations_csv"],
    params:
        checkpoint=PATHS["geo_checkpoint"],
        cache=PATHS["geo_cache"],
        partial=PATHS["geo_partial_csv"],
        model=OLLAMA["location_model"],
        url=OLLAMA["url"],
        sleep_s=OLLAMA["sleep_s"],
        save_every=OLLAMA["save_every"],
    shell:
        """
        {PYTHON} scripts/locations_ollama.py \
          --project-dir {PROJECT_DIR} \
          --input-csv {input} \
          --out-csv {output.csv} \
          --checkpoint {params.checkpoint} \
          --cache {params.cache} \
          --partial-csv {params.partial} \
          --ollama-url {params.url} \
          --model {params.model} \
          --country {COUNTRY} \
          --sleep-s {params.sleep_s} \
          --save-every {params.save_every}
        """


rule run_absa:
    input:
        PATHS["paragraph_locations_csv"],
    output:
        PATHS["sentences_with_absa_csv"],
    shell:
        """
        {PYTHON} scripts/ABSA.py \
          --project-dir {PROJECT_DIR} \
          --input-csv {input} \
          --output-csv {output}
        """


rule geocode_sentences_offline:
    input:
        csv=PATHS["sentences_with_absa_csv"],
        cbs=PATHS["cbs_gpkg"],
    output:
        gpkg=PATHS["sentences_with_absa_and_geo_gpkg"],
        csv=PATHS["sentence_offline_geocoding_csv"],
    params:
        layer_muni=LAYERS["municipality"],
        layer_prov=LAYERS["province"],
    shell:
        """
        {PYTHON} scripts/geocoding_offline.py \
          --project-dir {PROJECT_DIR} \
          --input-csv {input.csv} \
          --cbs-gpkg {input.cbs} \
          --layer-muni {params.layer_muni} \
          --layer-prov {params.layer_prov} \
          --output-gpkg {output.gpkg} \
          --output-csv {output.csv}
        """


rule classify_sentence_categories:
    input:
        gpkg=PATHS["sentences_with_absa_and_geo_gpkg"],
        keywords=PATHS["keywords_topics_csv"],
    output:
        gpkg=PATHS["sentences_with_categories_gpkg"],
        long_csv=PATHS["sentences_with_categories_long_csv"],
        short_csv=PATHS["sentences_with_categories_short_csv"],
    shell:
        """
        {PYTHON} scripts/classification_sentences.py \
          --project-dir {PROJECT_DIR} \
          --input-gpkg {input.gpkg} \
          --keywords-csv {input.keywords} \
          --output-gpkg {output.gpkg} \
          --output-long-csv {output.long_csv} \
          --output-short-csv {output.short_csv}
        """


rule aggregate_to_admin_areas:
    input:
        gpkg=PATHS["sentences_with_categories_gpkg"],
        cbs=PATHS["cbs_gpkg"],
    output:
        gpkg=PATHS["sentences_with_categories_admin_gpkg"],
        csv=PATHS["sentences_with_categories_admin_csv"],
    params:
        layer_muni=LAYERS["municipality"],
        layer_prov=LAYERS["province"],
    shell:
        """
        {PYTHON} scripts/geographic_aggregation.py \
          --project-dir {PROJECT_DIR} \
          --input-gpkg {input.gpkg} \
          --cbs-gpkg {input.cbs} \
          --layer-gemeente {params.layer_muni} \
          --layer-provincie {params.layer_prov} \
          --output-gpkg {output.gpkg} \
          --output-csv {output.csv}
        """
