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
DEFAULT_COUNTRY = config.get("country")
LANGUAGE_COUNTRIES = config.get("countries", {})


def _discover_languages():
    requested = config.get("languages", [LANGUAGE] if LANGUAGE else "auto")
    if requested in (None, "", "auto"):
        candidates = sorted(
            path.parent.name
            for path in Path("vocab").glob("*/keywords_topics.csv")
            if Path("input_data", path.parent.name).exists()
        )
    elif isinstance(requested, str):
        candidates = [requested]
    else:
        candidates = list(requested)
    return [language for language in candidates if language]


LANGUAGES = _discover_languages()
if not LANGUAGES:
    raise ValueError("No workflow languages found. Expected vocab/{language}/keywords_topics.csv and input_data/{language}/.")

LANGUAGE_PATTERN = "|".join(re.escape(language) for language in LANGUAGES)


def _localize_path(path_str, language):
    path = Path(path_str)
    if language and len(path.parts) > 0:
        if path.parts[0] == "output":
            return str(Path("output") / language / Path(*path.parts[1:]))
        if path.parts[0] == "cache":
            return str(Path("cache") / language / Path(*path.parts[1:]))
        if path.parts[0] == "input_data":
            return str(Path("input_data") / language / Path(*path.parts[1:]))
        if path.parts[0] == "data":
            return str(Path("data") / language / Path(*path.parts[1:]))
        if path.parts[0] == "annotation":
            return str(Path("annotation") / language / Path(*path.parts[1:]))
        if path.parts[0] == "vocab":
            return str(Path("vocab") / language / Path(*path.parts[1:]))
    return str(path)


def path_for(language, key):
    return _localize_path(PATHS[key], language)


def pattern_for(key):
    return _localize_path(PATHS[key], "{language}")


def country_for(language):
    return LANGUAGE_COUNTRIES.get(language) or DEFAULT_COUNTRY or language


def _default_country_codes(country):
    country_norm = str(country or "").strip().lower()
    mapping = {
        "netherlands": "nl,be,bq,aw,cw,sx",
        "italy": "it,sm,va",
        "germany": "de",
        "deutschland": "de",
        "german-speaking countries": "de,at,ch,li",
        "german speaking countries": "de,at,ch,li",
    }
    return mapping.get(country_norm, "")


def _country_codes_for(language):
    configured = ONLINE_GEOCODING.get("country_codes_by_language", {}).get(language)
    if configured:
        return configured
    return ONLINE_GEOCODING.get("country_codes") or _default_country_codes(country_for(language))


def _language_resource_path(language, filename):
    return str(Path("vocab") / language / filename)


wildcard_constraints:
    language=LANGUAGE_PATTERN,


def _preprocess_chunks_dir(language):
    return path_for(
        language,
        "preprocess_rtf_chunks_dir",
    )


def _discover_rtf_input_files(language):
    input_dir = Path(path_for(language, "input_rtf_dir"))
    if not input_dir.exists():
        return []
    return sorted(
        str(path)
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".rtf"
        and "doclist" not in path.stem.lower()
    )


def _rtf_id_for_path(language, path_str):
    input_dir = Path(path_for(language, "input_rtf_dir"))
    path = Path(path_str)
    try:
        rel = path.relative_to(input_dir).as_posix()
    except ValueError:
        rel = path.name
    stem = Path(rel).with_suffix("").as_posix()
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-") or "rtf"
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:10]
    return f"{safe_stem[:90]}-{digest}"


RTF_ID_TO_INPUT_BY_LANGUAGE = {
    language: {
        _rtf_id_for_path(language, path): path
        for path in _discover_rtf_input_files(language)
    }
    for language in LANGUAGES
}


def _rtf_chunks(language):
    return [
        str(Path(_preprocess_chunks_dir(language)) / "raw_articles" / f"{rtf_id}.csv")
        for rtf_id in sorted(RTF_ID_TO_INPUT_BY_LANGUAGE.get(language, {}))
    ]


def _rtf_raw_article_args(language):
    chunks = _rtf_chunks(language)
    return (
        "--input-raw-articles-csv " + " ".join(shlex.quote(path) for path in chunks)
        if chunks
        else ""
    )


PER_LANGUAGE_TARGET_KEYS = [
    "sentences_with_categories_admin_csv",
    "sentences_with_categories_admin_gpkg",
    "articles_per_year_csv",
    "articles_per_year_png",
    "top_newspapers_csv",
    "top_newspapers_png",
    "article_descriptives_summary_csv",
    "frame_keywords_dir",
    "region_frames_dir",
    "province_sentiment_table_csv",
    "provinces_sentiment_balance_png",
    "provinces_sentiment_distribution_png",
    "categories_sentiment_distribution_png",
    "locations_map_html",
]

ALL_TARGETS = [
    path_for(language, key)
    for language in LANGUAGES
    for key in PER_LANGUAGE_TARGET_KEYS
]
if MAKE_ANNOTATION_DF:
    ALL_TARGETS.extend(path_for(language, "annotation_sentences_csv") for language in LANGUAGES)
    ALL_TARGETS.append("annotation/sentences_for_annotation_all_languages.csv")


rule all:
    input:
        ALL_TARGETS,


rule preprocess_single_rtf_to_raw_articles:
    input:
        rtf=lambda wildcards: RTF_ID_TO_INPUT_BY_LANGUAGE[wildcards.language][wildcards.rtf_id],
        script=str(PROJECT_DIR / "scripts" / "preprocess_rtf_to_paragraphs.py"),
    output:
        raw=pattern_for("preprocess_rtf_chunks_dir") + "/raw_articles/{rtf_id}.csv",
    params:
        input_dir=lambda wildcards: path_for(wildcards.language, "input_rtf_dir"),
    shell:
        """
        {PYTHON} scripts/preprocess_rtf_to_paragraphs.py \
          --project-dir {PROJECT_DIR} \
          --language {wildcards.language} \
          --input-rtf-dir {params.input_dir:q} \
          --input-rtf-file {input.rtf:q} \
          --output-raw-articles-csv {output.raw:q} \
          --raw-only
        """


rule preprocess_rtf_to_paragraphs:
    input:
        chunks=lambda wildcards: _rtf_chunks(wildcards.language),
        regions=pattern_for("newspaper_regions_csv"),
        script=str(PROJECT_DIR / "scripts" / "preprocess_rtf_to_paragraphs.py"),
    output:
        paragraphs=pattern_for("paragraphs_csv"),
        articles=pattern_for("articles_csv"),
    params:
        input_dir=lambda wildcards: path_for(wildcards.language, "input_rtf_dir"),
        raw_article_args=lambda wildcards: _rtf_raw_article_args(wildcards.language),
    shell:
        """
        {PYTHON} scripts/preprocess_rtf_to_paragraphs.py \
          --project-dir {PROJECT_DIR} \
          --language {wildcards.language} \
          --input-rtf-dir {params.input_dir:q} \
          {params.raw_article_args} \
          --newspaper-region-csv {input.regions:q} \
          --output-paragraph-csv {output.paragraphs:q} \
          --output-articles-csv {output.articles:q} \
          --write-articles-csv
        """


rule update_keyword_framework:
    input:
        base_csv=pattern_for("keywords_topics_base_csv"),
    output:
        keywords_csv=pattern_for("keywords_topics_csv"),
        audit_csv=pattern_for("keyword_framework_audit_csv"),
    params:
        review_csv=lambda wildcards: path_for(wildcards.language, "keyword_review_export_csv"),
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
        paragraphs=pattern_for("paragraphs_csv"),
        keywords=pattern_for("keywords_topics_csv"),
    output:
        pattern_for("paragraph_keyword_filtered_csv"),
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
        pattern_for("paragraph_keyword_filtered_csv"),
    output:
        csv=pattern_for("paragraph_geothermal_csv"),
    params:
        checkpoint=lambda wildcards: path_for(wildcards.language, "geo_class_checkpoint"),
        cache=lambda wildcards: path_for(wildcards.language, "geo_class_cache"),
        partial=lambda wildcards: path_for(wildcards.language, "geo_class_partial_csv"),
        model=OLLAMA["geothermal_model"],
        url=OLLAMA["url"],
        sleep_s=OLLAMA["sleep_s"],
        save_every=OLLAMA["save_every"],
        country=lambda wildcards: country_for(wildcards.language),
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
          --country "{params.country}" \
          --sleep-s {params.sleep_s} \
          --save-every {params.save_every}
        """


rule extract_locations:
    input:
        pattern_for("paragraph_geothermal_csv"),
    output:
        csv=pattern_for("paragraph_locations_csv"),
    params:
        checkpoint=lambda wildcards: path_for(wildcards.language, "geo_checkpoint"),
        cache=lambda wildcards: path_for(wildcards.language, "geo_cache"),
        partial=lambda wildcards: path_for(wildcards.language, "geo_partial_csv"),
        model=OLLAMA["location_model"],
        url=OLLAMA["url"],
        sleep_s=OLLAMA["sleep_s"],
        save_every=OLLAMA["save_every"],
        country=lambda wildcards: country_for(wildcards.language),
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
          --country "{params.country}" \
          --sleep-s {params.sleep_s} \
          --save-every {params.save_every}
        """


rule geocode_paragraphs_offline:
    input:
        csv=pattern_for("paragraph_locations_csv"),
        muni=pattern_for("municipality_gpkg"),
        prov=pattern_for("province_gpkg"),
    output:
        gpkg=pattern_for("paragraphs_with_geo_offline_gpkg"),
        csv=pattern_for("paragraph_offline_geocoding_csv"),
    params:
        country=lambda wildcards: country_for(wildcards.language),
    shell:
        """
        {PYTHON} scripts/geocoding_offline.py \
          --project-dir {PROJECT_DIR} \
          --input-csv {input.csv} \
          --municipality-gpkg {input.muni} \
          --province-gpkg {input.prov} \
          --output-gpkg {output.gpkg} \
          --output-csv {output.csv} \
          --country "{params.country}" \
          --points-layer paragraphs_points \
          --polygons-layer paragraphs_polygons
        """


rule geocode_paragraphs_online:
    input:
        csv=pattern_for("paragraph_offline_geocoding_csv"),
        muni=pattern_for("municipality_gpkg"),
        prov=pattern_for("province_gpkg"),
    output:
        gpkg=pattern_for("paragraphs_with_geo_gpkg"),
        csv=pattern_for("paragraphs_with_geo_csv"),
    params:
        cache=lambda wildcards: path_for(wildcards.language, "nominatim_cache_json"),
        country=lambda wildcards: country_for(wildcards.language),
        country_codes=lambda wildcards: _country_codes_for(wildcards.language),
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
          --country "{params.country}" \
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
        pattern_for("paragraphs_with_geo_csv"),
    output:
        pattern_for("sentence_locations_csv"),
    shell:
        """
        {PYTHON} scripts/split_paragraphs_to_sentences.py \
          --project-dir {PROJECT_DIR} \
          --input-csv {input} \
          --output-csv {output} \
          --language {wildcards.language}
        """


rule export_frame_keyword_review_candidates:
    input:
        sentences_csv=pattern_for("sentence_locations_csv"),
        keywords_csv=pattern_for("keywords_topics_csv"),
    output:
        pattern_for("keyword_review_candidates_csv"),
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
        csv=pattern_for("sentence_locations_csv"),
        keywords=pattern_for("keywords_topics_csv"),
    output:
        long_csv=pattern_for("sentences_with_frames_long_csv"),
        short_csv=pattern_for("sentences_with_frames_short_csv"),
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
        pattern_for("sentences_with_frames_long_csv"),
    output:
        pattern_for("sentence_sentiment_csv"),
    params:
        checkpoint=lambda wildcards: path_for(wildcards.language, "sentence_sentiment_checkpoint"),
        cache=lambda wildcards: path_for(wildcards.language, "sentence_sentiment_cache"),
        partial=lambda wildcards: path_for(wildcards.language, "sentence_sentiment_partial_csv"),
        model=config.get("sentiment_ollama", {}).get("model", "llama3.1:8b"),
        ollama_url=OLLAMA["url"],
        language=lambda wildcards: wildcards.language,
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
        csv=pattern_for("sentence_sentiment_csv"),
        keywords=pattern_for("keywords_topics_csv"),
    output:
        gpkg=pattern_for("sentences_with_categories_gpkg"),
        long_csv=pattern_for("sentences_with_categories_long_csv"),
        short_csv=pattern_for("sentences_with_categories_short_csv"),
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
        gpkg=pattern_for("sentences_with_categories_gpkg"),
        muni=pattern_for("municipality_gpkg"),
        prov=pattern_for("province_gpkg"),
    output:
        gpkg=pattern_for("sentences_with_categories_admin_gpkg"),
        csv=pattern_for("sentences_with_categories_admin_csv"),
    params:
        location_overrides=lambda wildcards: _language_resource_path(wildcards.language, "location_province_overrides.csv"),
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
        articles_csv=pattern_for("articles_csv"),
        script=str(PROJECT_DIR / "scripts" / "visualize_article_descriptives.py"),
    output:
        articles_per_year_csv=pattern_for("articles_per_year_csv"),
        articles_per_year_png=pattern_for("articles_per_year_png"),
        top_newspapers_csv=pattern_for("top_newspapers_csv"),
        top_newspapers_png=pattern_for("top_newspapers_png"),
        summary_csv=pattern_for("article_descriptives_summary_csv"),
    params:
        output_dir=lambda wildcards: path_for(wildcards.language, "figures_dir"),
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
        admin_csv=pattern_for("sentences_with_categories_admin_csv"),
        categories_csv=pattern_for("sentences_with_categories_short_csv"),
        keywords_csv=pattern_for("keywords_topics_csv"),
        province_gpkg=pattern_for("province_gpkg"),
        script=str(PROJECT_DIR / "scripts" / "visualize_absa_results.py"),
    output:
        frame_keywords=directory(pattern_for("frame_keywords_dir")),
        region_frames=directory(pattern_for("region_frames_dir")),
        table=pattern_for("province_sentiment_table_csv"),
        balance=pattern_for("provinces_sentiment_balance_png"),
        distribution=pattern_for("provinces_sentiment_distribution_png"),
        categories=pattern_for("categories_sentiment_distribution_png"),
        heatmap=pattern_for("locations_map_html"),
    params:
        output_dir=lambda wildcards: path_for(wildcards.language, "figures_dir"),
        country=lambda wildcards: country_for(wildcards.language),
        location_overrides=lambda wildcards: _language_resource_path(wildcards.language, "location_province_overrides.csv"),
    shell:
        """
        {PYTHON} scripts/visualize_absa_results.py \
          --project-dir {PROJECT_DIR} \
          --admin-csv {input.admin_csv} \
          --categories-csv {input.categories_csv} \
          --keywords-csv {input.keywords_csv} \
          --province-gpkg {input.province_gpkg} \
          --output-dir {params.output_dir} \
          --country "{params.country}" \
          --location-province-overrides {params.location_overrides}
        """


rule make_annotation_df:
    input:
        sentences_csv=pattern_for("sentence_locations_csv"),
        sentiment_csv=pattern_for("sentence_sentiment_csv"),
        keywords_csv=pattern_for("keywords_topics_csv"),
        paragraph_geothermal_csv=pattern_for("paragraph_geothermal_csv"),
    output:
        pattern_for("annotation_sentences_csv"),
    params:
        province_filter_regex=ANNOTATION.get("province_filter_regex", ""),
        exclude_neutral=ANNOTATION.get("exclude_neutral", True),
        sentence_sample_size=ANNOTATION.get("sentence_sample_size", 500),
        paragraph_sample_size=ANNOTATION.get("paragraph_sample_size", 100),
        random_state=ANNOTATION.get("sample_random_state", 42),
    shell:
        """
        {PYTHON} scripts/make_annotation_df.py \
          --project-dir {PROJECT_DIR} \
          --input-csv {input.sentences_csv} \
          --paragraph-geothermal-csv {input.paragraph_geothermal_csv} \
          --sentiment-csv {input.sentiment_csv} \
          --keywords-csv {input.keywords_csv} \
          --output-csv {output} \
          --province-filter-regex "{params.province_filter_regex}" \
          --exclude-neutral {params.exclude_neutral} \
          --sentence-sample-size {params.sentence_sample_size} \
          --paragraph-sample-size {params.paragraph_sample_size} \
          --random-state {params.random_state}
        """


rule combine_annotation_dataframes:
    input:
        lambda wildcards: [path_for(language, "annotation_sentences_csv") for language in LANGUAGES]
    output:
        "annotation/sentences_for_annotation_all_languages.csv"
    shell:
        """
        {PYTHON} scripts/combine_annotation_dataframes.py \
          --output-csv {output} \
          --inputs {input}
        """
