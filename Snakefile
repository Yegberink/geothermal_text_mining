from pathlib import Path
import hashlib
import re
import shlex

configfile: "config/config.yaml"

PROJECT_DIR = Path(workflow.basedir).resolve()
PYTHON = config.get("python", "python")
OLLAMA = config["ollama"]
ONLINE_GEOCODING = config.get("online_geocoding", {})
MAKE_ANNOTATION_DF = config.get("make_annotation_df", False)
ANNOTATION = config.get("annotation", {})
PATHS = config["paths"]
LANGUAGE = config.get("language")
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
        if path.parts[0] == "data":
            return str(Path("data") / LANGUAGE / Path(*path.parts[1:]))
        if path.parts[0] == "annotation":
            return str(Path("annotation") / LANGUAGE / Path(*path.parts[1:]))
        if path.parts[0] == "vocab":
            return str(Path("vocab") / LANGUAGE / Path(*path.parts[1:]))
    return str(path)


if LANGUAGE:
    PATHS = {key: _local_path(value) for key, value in PATHS.items()}


def _default_country_codes(country):
    country_norm = str(country or "").strip().lower()
    mapping = {
        "netherlands": "nl,be,bq,aw,cw,sx",
        "italy": "it,sm,va",
        "germany": "de",
        "deutschland": "de",
    }
    return mapping.get(country_norm, "")


def _language_resource_path(filename):
    if LANGUAGE:
        return str(Path("vocab") / LANGUAGE / filename)
    return str(Path("vocab") / filename)


PREPROCESS_RTF_CHUNKS_DIR = PATHS.get(
    "preprocess_rtf_chunks_dir",
    str(Path(PATHS["paragraphs_csv"]).parent / "_preprocess_rtf_chunks"),
)


def _discover_rtf_input_files():
    input_dir = Path(PATHS["input_rtf_dir"])
    if not input_dir.exists():
        return []
    return sorted(
        str(path)
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".rtf"
        and "doclist" not in path.stem.lower()
    )


def _rtf_id_for_path(path_str):
    input_dir = Path(PATHS["input_rtf_dir"])
    path = Path(path_str)
    try:
        rel = path.relative_to(input_dir).as_posix()
    except ValueError:
        rel = path.name
    stem = Path(rel).with_suffix("").as_posix()
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-") or "rtf"
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:10]
    return f"{safe_stem[:90]}-{digest}"


RTF_INPUT_FILES = _discover_rtf_input_files()
RTF_ID_TO_INPUT = {_rtf_id_for_path(path): path for path in RTF_INPUT_FILES}
RTF_IDS = sorted(RTF_ID_TO_INPUT)
RTF_RAW_ARTICLE_CHUNKS = [
    str(Path(PREPROCESS_RTF_CHUNKS_DIR) / "raw_articles" / f"{rtf_id}.csv")
    for rtf_id in RTF_IDS
]
RTF_RAW_ARTICLE_ARGS = (
    "--input-raw-articles-csv " + " ".join(shlex.quote(path) for path in RTF_RAW_ARTICLE_CHUNKS)
    if RTF_RAW_ARTICLE_CHUNKS
    else ""
)


ALL_TARGETS = [
    PATHS["sentences_with_categories_admin_csv"],
    PATHS["sentences_with_categories_admin_gpkg"],
    PATHS["articles_per_year_csv"],
    PATHS["articles_per_year_png"],
    PATHS["top_newspapers_csv"],
    PATHS["top_newspapers_png"],
    PATHS["article_descriptives_summary_csv"],
    PATHS["frame_keywords_dir"],
    PATHS["region_frames_dir"],
    PATHS["province_sentiment_table_csv"],
    PATHS["provinces_sentiment_balance_png"],
    PATHS["provinces_sentiment_distribution_png"],
    PATHS["categories_sentiment_distribution_png"],
    PATHS["locations_map_html"],
]
if MAKE_ANNOTATION_DF:
    ALL_TARGETS.append(PATHS["annotation_sentences_csv"])


rule all:
    input:
        ALL_TARGETS,


rule preprocess_single_rtf_to_raw_articles:
    input:
        rtf=lambda wildcards: RTF_ID_TO_INPUT[wildcards.rtf_id],
        script=str(PROJECT_DIR / "scripts" / "preprocess_rtf_to_paragraphs.py"),
    output:
        raw=str(Path(PREPROCESS_RTF_CHUNKS_DIR) / "raw_articles" / "{rtf_id}.csv"),
    params:
        input_dir=PATHS["input_rtf_dir"],
    shell:
        """
        {PYTHON} scripts/preprocess_rtf_to_paragraphs.py \
          --project-dir {PROJECT_DIR} \
          --input-rtf-dir {params.input_dir:q} \
          --input-rtf-file {input.rtf:q} \
          --output-raw-articles-csv {output.raw:q} \
          --raw-only
        """


rule preprocess_rtf_to_paragraphs:
    input:
        chunks=RTF_RAW_ARTICLE_CHUNKS,
        regions=PATHS["newspaper_regions_csv"],
        script=str(PROJECT_DIR / "scripts" / "preprocess_rtf_to_paragraphs.py"),
    output:
        paragraphs=PATHS["paragraphs_csv"],
        articles=PATHS["articles_csv"],
    params:
        input_dir=PATHS["input_rtf_dir"],
        raw_article_args=RTF_RAW_ARTICLE_ARGS,
    shell:
        """
        {PYTHON} scripts/preprocess_rtf_to_paragraphs.py \
          --project-dir {PROJECT_DIR} \
          --input-rtf-dir {params.input_dir:q} \
          {params.raw_article_args} \
          --newspaper-region-csv {input.regions:q} \
          --output-paragraph-csv {output.paragraphs:q} \
          --output-articles-csv {output.articles:q} \
          --write-articles-csv
        """


rule update_keyword_framework:
    input:
        base_csv=PATHS["keywords_topics_base_csv"],
    output:
        keywords_csv=PATHS["keywords_topics_csv"],
        audit_csv=PATHS["keyword_framework_audit_csv"],
    params:
        review_csv=PATHS["keyword_review_export_csv"],
    shell:
        """
        {PYTHON} scripts/update_keywords_framework.py \
          --project-dir {PROJECT_DIR} \
          --base-csv {input.base_csv} \
          --review-csv {params.review_csv} \
          --output-csv {output.keywords_csv} \
          --audit-csv {output.audit_csv}
        """


rule filter_paragraphs_by_keywords:
    input:
        paragraphs=PATHS["paragraphs_csv"],
        keywords=PATHS["keywords_topics_csv"],
    output:
        PATHS["paragraph_keyword_filtered_csv"],
    shell:
        """
        {PYTHON} scripts/filter_paragraphs_by_keywords.py \
          --project-dir {PROJECT_DIR} \
          --input-csv {input.paragraphs} \
          --keywords-csv {input.keywords} \
          --output-csv {output}
        """


rule classify_geothermal:
    input:
        PATHS["paragraph_keyword_filtered_csv"],
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


rule geocode_paragraphs_offline:
    input:
        csv=PATHS["paragraph_locations_csv"],
        muni=PATHS["municipality_gpkg"],
        prov=PATHS["province_gpkg"],
    output:
        gpkg=PATHS["paragraphs_with_geo_offline_gpkg"],
        csv=PATHS["paragraph_offline_geocoding_csv"],
    shell:
        """
        {PYTHON} scripts/geocoding_offline.py \
          --project-dir {PROJECT_DIR} \
          --input-csv {input.csv} \
          --municipality-gpkg {input.muni} \
          --province-gpkg {input.prov} \
          --output-gpkg {output.gpkg} \
          --output-csv {output.csv} \
          --country "{COUNTRY}" \
          --points-layer paragraphs_points \
          --polygons-layer paragraphs_polygons
        """


rule geocode_paragraphs_online:
    input:
        csv=PATHS["paragraph_offline_geocoding_csv"],
        muni=PATHS["municipality_gpkg"],
        prov=PATHS["province_gpkg"],
    output:
        gpkg=PATHS["paragraphs_with_geo_gpkg"],
        csv=PATHS["paragraphs_with_geo_csv"],
    params:
        cache=PATHS["nominatim_cache_json"],
        country_codes=ONLINE_GEOCODING.get("country_codes") or _default_country_codes(COUNTRY),
        user_agent=ONLINE_GEOCODING.get("user_agent", "absa-geo-mapper"),
        save_every=ONLINE_GEOCODING.get("save_every", 50),
        print_every=ONLINE_GEOCODING.get("print_every", 25),
        min_delay_seconds=ONLINE_GEOCODING.get("min_delay_seconds", 1.1),
        timeout_seconds=ONLINE_GEOCODING.get("timeout_seconds", 10),
        max_retries=ONLINE_GEOCODING.get("max_retries", 6),
        backoff_base=ONLINE_GEOCODING.get("backoff_base", 1.6),
        jitter=ONLINE_GEOCODING.get("jitter", 0.25),
        max_queries=ONLINE_GEOCODING.get("max_queries", 0),
    shell:
        """
        {PYTHON} scripts/geocoding_online.py \
          --project-dir {PROJECT_DIR} \
          --input-csv {input.csv} \
          --municipality-gpkg {input.muni} \
          --province-gpkg {input.prov} \
          --output-gpkg {output.gpkg} \
          --output-csv {output.csv} \
          --cache-path {params.cache} \
          --country "{COUNTRY}" \
          --country-codes "{params.country_codes}" \
          --user-agent {params.user_agent} \
          --save-every {params.save_every} \
          --print-every {params.print_every} \
          --min-delay-seconds {params.min_delay_seconds} \
          --timeout-seconds {params.timeout_seconds} \
          --max-retries {params.max_retries} \
          --backoff-base {params.backoff_base} \
          --jitter {params.jitter} \
          --max-queries {params.max_queries} \
          --points-layer paragraphs_points \
          --polygons-layer paragraphs_polygons
        """


rule split_paragraphs_to_sentences:
    input:
        PATHS["paragraphs_with_geo_csv"],
    output:
        PATHS["sentence_locations_csv"],
    shell:
        """
        {PYTHON} scripts/split_paragraphs_to_sentences.py \
          --project-dir {PROJECT_DIR} \
          --input-csv {input} \
          --output-csv {output} \
          --language {LANGUAGE}
        """


rule export_frame_keyword_review_candidates:
    input:
        sentences_csv=PATHS["sentence_locations_csv"],
        keywords_csv=PATHS["keywords_topics_csv"],
    output:
        PATHS["keyword_review_candidates_csv"],
    shell:
        """
        {PYTHON} scripts/export_frame_keyword_review_candidates.py \
          --project-dir {PROJECT_DIR} \
          --sentences-csv {input.sentences_csv} \
          --keywords-csv {input.keywords_csv} \
          --output-csv {output}
        """


rule identify_sentence_frames:
    input:
        csv=PATHS["sentence_locations_csv"],
        keywords=PATHS["keywords_topics_csv"],
    output:
        long_csv=PATHS["sentences_with_frames_long_csv"],
        short_csv=PATHS["sentences_with_frames_short_csv"],
    shell:
        """
        {PYTHON} scripts/classification_sentences.py \
          --project-dir {PROJECT_DIR} \
          --input-csv {input.csv} \
          --keywords-csv {input.keywords} \
          --output-long-csv {output.long_csv} \
          --output-short-csv {output.short_csv} \
          --keep-only-matched True
        """


rule classify_sentence_sentiment:
    input:
        PATHS["sentences_with_frames_long_csv"],
    output:
        PATHS["sentence_sentiment_csv"],
    params:
        checkpoint=PATHS["sentence_sentiment_checkpoint"],
        cache=PATHS["sentence_sentiment_cache"],
        partial=PATHS["sentence_sentiment_partial_csv"],
        model=config.get("sentiment_ollama", {}).get("model", "llama3.1:8b"),
        ollama_url=OLLAMA["url"],
        language=LANGUAGE,
        prompt_variant=config.get("sentiment_ollama", {}).get("prompt_variant", "zero_shot"),
        timeout=config.get("sentiment_ollama", {}).get("timeout", 120),
        sleep_s=config.get("sentiment_ollama", {}).get("sleep_s", 0.0),
        save_every=config.get("sentiment_ollama", {}).get("save_every", 25),
    shell:
        """
        {PYTHON} scripts/sentiment_classification.py \
          --project-dir {PROJECT_DIR} \
          --input-csv {input} \
          --output-csv {output} \
          --text-col sentence_text \
          --checkpoint {params.checkpoint} \
          --cache {params.cache} \
          --partial-csv {params.partial} \
          --model {params.model} \
          --ollama-url {params.ollama_url} \
          --language {params.language} \
          --prompt-variant {params.prompt_variant} \
          --timeout {params.timeout} \
          --sleep-s {params.sleep_s} \
          --save-every {params.save_every}
        """


rule classify_sentence_categories:
    input:
        csv=PATHS["sentence_sentiment_csv"],
        keywords=PATHS["keywords_topics_csv"],
    output:
        gpkg=PATHS["sentences_with_categories_gpkg"],
        long_csv=PATHS["sentences_with_categories_long_csv"],
        short_csv=PATHS["sentences_with_categories_short_csv"],
    shell:
        """
        {PYTHON} scripts/classification_sentences.py \
          --project-dir {PROJECT_DIR} \
          --input-csv {input.csv} \
          --keywords-csv {input.keywords} \
          --output-gpkg {output.gpkg} \
          --output-layer sentences_with_categories \
          --output-long-csv {output.long_csv} \
          --output-short-csv {output.short_csv} \
          --keep-only-matched True
        """


rule aggregate_to_admin_areas:
    input:
        gpkg=PATHS["sentences_with_categories_gpkg"],
        muni=PATHS["municipality_gpkg"],
        prov=PATHS["province_gpkg"],
    output:
        gpkg=PATHS["sentences_with_categories_admin_gpkg"],
        csv=PATHS["sentences_with_categories_admin_csv"],
    params:
        location_overrides=_language_resource_path("location_province_overrides.csv"),
    shell:
        """
        {PYTHON} scripts/geographic_aggregation.py \
          --project-dir {PROJECT_DIR} \
          --input-gpkg {input.gpkg} \
          --input-layer sentences_with_categories \
          --municipality-gpkg {input.muni} \
          --province-gpkg {input.prov} \
          --output-gpkg {output.gpkg} \
          --output-csv {output.csv} \
          --location-province-overrides {params.location_overrides}
        """


rule visualize_article_descriptives:
    input:
        articles_csv=PATHS["articles_csv"],
        script=str(PROJECT_DIR / "scripts" / "visualize_article_descriptives.py"),
    output:
        articles_per_year_csv=PATHS["articles_per_year_csv"],
        articles_per_year_png=PATHS["articles_per_year_png"],
        top_newspapers_csv=PATHS["top_newspapers_csv"],
        top_newspapers_png=PATHS["top_newspapers_png"],
        summary_csv=PATHS["article_descriptives_summary_csv"],
    params:
        output_dir=PATHS["figures_dir"],
    shell:
        """
        {PYTHON} scripts/visualize_article_descriptives.py \
          --project-dir {PROJECT_DIR} \
          --articles-csv {input.articles_csv} \
          --output-dir {params.output_dir} \
          --top-n-newspapers 6
        """


rule visualize_absa_results:
    input:
        admin_csv=PATHS["sentences_with_categories_admin_csv"],
        categories_csv=PATHS["sentences_with_categories_short_csv"],
        keywords_csv=PATHS["keywords_topics_csv"],
        province_gpkg=PATHS["province_gpkg"],
        script=str(PROJECT_DIR / "scripts" / "visualize_absa_results.py"),
    output:
        frame_keywords=directory(PATHS["frame_keywords_dir"]),
        region_frames=directory(PATHS["region_frames_dir"]),
        table=PATHS["province_sentiment_table_csv"],
        balance=PATHS["provinces_sentiment_balance_png"],
        distribution=PATHS["provinces_sentiment_distribution_png"],
        categories=PATHS["categories_sentiment_distribution_png"],
        heatmap=PATHS["locations_map_html"],
    params:
        output_dir=PATHS["figures_dir"],
        location_overrides=_language_resource_path("location_province_overrides.csv"),
    shell:
        """
        {PYTHON} scripts/visualize_absa_results.py \
          --project-dir {PROJECT_DIR} \
          --admin-csv {input.admin_csv} \
          --categories-csv {input.categories_csv} \
          --keywords-csv {input.keywords_csv} \
          --province-gpkg {input.province_gpkg} \
          --output-dir {params.output_dir} \
          --country "{COUNTRY}" \
          --location-province-overrides {params.location_overrides}
        """


rule make_annotation_df:
    input:
        admin_csv=PATHS["sentences_with_categories_admin_csv"],
    output:
        PATHS["annotation_sentences_csv"],
    params:
        province_filter_regex=ANNOTATION.get("province_filter_regex", ""),
        exclude_neutral=ANNOTATION.get("exclude_neutral", True),
    shell:
        """
        {PYTHON} scripts/make_annotation_df.py \
          --project-dir {PROJECT_DIR} \
          --input-csv {input.admin_csv} \
          --output-csv {output} \
          --province-filter-regex "{params.province_filter_regex}" \
          --exclude-neutral {params.exclude_neutral}
        """
