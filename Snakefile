from pathlib import Path
import hashlib
import re
import shlex
import sys

configfile: "config/config.yaml"

PROJECT_DIR = Path(workflow.basedir).resolve()
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from helpers.country_scope import country_scope_from_config, countries_from_value

PYTHON = f"PYTHONPATH={shlex.quote(str(PROJECT_DIR / 'scripts'))} {config.get('python', 'python')}"
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
            for path in Path("data/vocab").glob("*/keywords_topics.csv")
            if Path("data/text_data", path.parent.name).exists()
        )
    elif isinstance(requested, str):
        candidates = [requested]
    else:
        candidates = list(requested)
    return [language for language in candidates if language]


LANGUAGES = _discover_languages()
if not LANGUAGES:
    raise ValueError("No workflow languages found. Expected data/vocab/{language}/keywords_topics.csv and data/text_data/{language}/.")

LANGUAGE_PATTERN = "|".join(re.escape(language) for language in LANGUAGES)


def _localize_path(path_str, language):
    path = Path(path_str)
    if language and len(path.parts) > 0:
        if path.parts[0] == "output":
            return str(Path("output") / language / Path(*path.parts[1:]))
        if path.parts[0] == "cache":
            return str(Path("cache") / language / Path(*path.parts[1:]))
        if len(path.parts) >= 2 and path.parts[:2] == ("data", "text_data"):
            return str(Path("data/text_data") / language / Path(*path.parts[2:]))
        if len(path.parts) >= 2 and path.parts[:2] == ("data", "vocab"):
            return str(Path("data/vocab") / language / Path(*path.parts[2:]))
        if path.parts[0] == "annotation":
            return str(Path("annotation") / language / Path(*path.parts[1:]))
    return str(path)


def path_for(language, key):
    return _localize_path(PATHS[key], language)


def pattern_for(key):
    return _localize_path(PATHS[key], "{language}")


def country_for(language):
    scope = country_scope_from_config(config, language)
    if scope.countries:
        return scope.label
    return LANGUAGE_COUNTRIES.get(language) or DEFAULT_COUNTRY or language


def countries_for(language):
    return list(country_scope_from_config(config, language).countries)


def _default_country_codes(country):
    codes_by_country = {
        "Netherlands": ["nl", "be", "bq", "aw", "cw", "sx"],
        "Italy": ["it", "sm", "va"],
        "Germany": ["de"],
        "Austria": ["at"],
        "Switzerland": ["ch"],
    }
    codes = []
    for canonical_country in countries_from_value(country):
        codes.extend(codes_by_country.get(canonical_country, []))
    return ",".join(dict.fromkeys(codes))


def _country_codes_for(language):
    configured = ONLINE_GEOCODING.get("country_codes_by_language", {}).get(language)
    if configured:
        return configured
    return ONLINE_GEOCODING.get("country_codes") or _default_country_codes(country_for(language))


def _geonames_country_codes_for(language):
    configured = ONLINE_GEOCODING.get("geonames_country_codes_by_language", {}).get(language)
    if configured:
        return configured
    return _country_codes_for(language)


def _language_resource_path(language, filename):
    return str(Path("data/vocab") / language / filename)


wildcard_constraints:
    language=LANGUAGE_PATTERN,


def _preprocess_chunks_dir(language):
    return path_for(
        language,
        "preprocess_rtf_chunks_dir",
    )


def _discover_source_input_files(language):
    input_dir = Path(path_for(language, "input_rtf_dir"))
    if not input_dir.exists():
        return []
    return sorted(
        str(path)
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".rtf", ".pdf"}
        and "doclist" not in path.stem.lower()
    )


def _source_id_for_path(language, path_str):
    input_dir = Path(path_for(language, "input_rtf_dir"))
    path = Path(path_str)
    try:
        rel = path.relative_to(input_dir).as_posix()
    except ValueError:
        rel = path.name
    stem = Path(rel).with_suffix("").as_posix()
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-") or "source"
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:10]
    return f"{safe_stem[:90]}-{digest}"


SOURCE_ID_TO_INPUT_BY_LANGUAGE = {
    language: {
        _source_id_for_path(language, path): path
        for path in _discover_source_input_files(language)
    }
    for language in LANGUAGES
}


def _source_chunks(language):
    return [
        str(Path(_preprocess_chunks_dir(language)) / "raw_articles" / f"{rtf_id}.csv")
        for rtf_id in sorted(SOURCE_ID_TO_INPUT_BY_LANGUAGE.get(language, {}))
    ]


def _source_raw_article_args(language):
    chunks = _source_chunks(language)
    return (
        "--input-raw-articles-csv " + " ".join(shlex.quote(path) for path in chunks)
        if chunks
        else ""
    )


def _shell_join(values):
    return " ".join(shlex.quote(str(value)) for value in values)


def _countries_arg(language):
    countries = countries_for(language)
    return _shell_join(countries if countries else [country_for(language)])


def _geocoding_candidate_specs():
    return _shell_join(
        (
            f"{language}={country_for(language)}={_country_codes_for(language)}="
            f"{path_for(language, 'geocoding_cache_candidates_csv')}"
        )
        for language in LANGUAGES
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

DATA_TRACKING_CSV = PATHS.get("data_tracking_csv", "output/generic/text/data_tracking.csv")

OVERARCHING_TARGETS = [
    "output/generic/figures/all_languages_extreme_province_sentiment_balance.png",
    "output/generic/figures/keywords_frames_regions.png",
    "output/generic/text/keywords_frames_regions_table.csv",
    "output/generic/figures/all_languages_province_sentiment_balance.png",
    "output/generic/text/all_languages_province_sentiment_table.csv",
    "output/generic/text/all_languages_extreme_province_frame_shares_table.csv",
    "output/generic/text/frame_mentions_100pct_stacked_table.csv",
    "output/generic/figures/frame_mentions_100pct_stacked.png",
    "output/generic/figures/all_languages_province_sentiment_map.png",
    DATA_TRACKING_CSV,
]

ALL_TARGETS = [
    path_for(language, key)
    for language in LANGUAGES
    for key in PER_LANGUAGE_TARGET_KEYS
]
ALL_TARGETS.extend(OVERARCHING_TARGETS)
if MAKE_ANNOTATION_DF:
    ALL_TARGETS.extend(path_for(language, "annotation_sentences_csv") for language in LANGUAGES)
    ALL_TARGETS.append("annotation/sentences_for_annotation_all_languages.csv")


TRACKING_STAGE_KEYS = [
    "articles_csv", "paragraphs_csv", "paragraph_geothermal_csv",
    "paragraphs_with_geo_csv", "sentence_locations_csv",
    "sentences_with_frames_long_csv", "sentence_sentiment_csv",
]


def _data_tracking_inputs(wildcards=None):
    files = [str(PROJECT_DIR / "scripts" / "results_tracking" / "build_data_tracking_table.py")]
    for language in LANGUAGES:
        files.extend(_source_chunks(language))
        files.extend(path_for(language, key) for key in TRACKING_STAGE_KEYS)
    return files


rule all:
    input:
        ALL_TARGETS,


rule preprocess_single_source_to_raw_articles:
    input:
        source=lambda wildcards: SOURCE_ID_TO_INPUT_BY_LANGUAGE[wildcards.language][wildcards.source_id],
        script=str(PROJECT_DIR / "scripts" / "core_workflow" / "preprocess_rtf_to_paragraphs.py"),
    output:
        raw=pattern_for("preprocess_rtf_chunks_dir") + "/raw_articles/{source_id}.csv",
    params:
        input_dir=lambda wildcards: path_for(wildcards.language, "input_rtf_dir"),
    shell:
        """
        {PYTHON} scripts/core_workflow/preprocess_rtf_to_paragraphs.py \
          --project-dir {PROJECT_DIR} \
          --language {wildcards.language} \
          --input-dir {params.input_dir:q} \
          --input-file {input.source:q} \
          --output-raw-articles-csv {output.raw:q} \
          --raw-only
        """


rule preprocess_rtf_to_paragraphs:
    input:
        chunks=lambda wildcards: _source_chunks(wildcards.language),
        script=str(PROJECT_DIR / "scripts" / "core_workflow" / "preprocess_rtf_to_paragraphs.py"),
    output:
        paragraphs=pattern_for("paragraphs_csv"),
        articles=pattern_for("articles_csv"),
    params:
        input_dir=lambda wildcards: path_for(wildcards.language, "input_rtf_dir"),
        raw_article_args=lambda wildcards: _source_raw_article_args(wildcards.language),
    shell:
        """
        {PYTHON} scripts/core_workflow/preprocess_rtf_to_paragraphs.py \
          --project-dir {PROJECT_DIR} \
          --language {wildcards.language} \
          --input-dir {params.input_dir:q} \
          {params.raw_article_args} \
          --output-paragraph-csv {output.paragraphs:q} \
          --output-articles-csv {output.articles:q} \
          --write-articles-csv
        """


rule filter_paragraphs_by_keywords:
    input:
        paragraphs=pattern_for("paragraphs_csv"),
        keywords=pattern_for("keywords_topics_csv"),
    output:
        pattern_for("paragraph_keyword_filtered_csv"),
    shell:
        """
        {PYTHON} scripts/core_workflow/filter_paragraphs_by_keywords.py \
          --project-dir {PROJECT_DIR} \
          --input-csv {input.paragraphs} \
          --keywords-csv {input.keywords} \
          --output-csv {output}
        """


rule classify_geothermal:
    input:
        paragraphs=pattern_for("paragraphs_csv"),
        lexicon=lambda wildcards: _language_resource_path(wildcards.language, "geo_keywords.yaml"),
    output:
        csv=pattern_for("paragraph_geothermal_csv"),
    params:
        cache=lambda wildcards: path_for(wildcards.language, "geo_class_cache"),
        model=OLLAMA["geothermal_model"],
        thinking="--think" if OLLAMA.get("think", False) else "--no-think",
        url=OLLAMA["url"],
        sleep_s=OLLAMA["sleep_s"],
        save_every=OLLAMA["save_every"],
        country=lambda wildcards: country_for(wildcards.language),
        countries=lambda wildcards: _countries_arg(wildcards.language),
        language=lambda wildcards: wildcards.language,
    shell:
        """
        {PYTHON} scripts/core_workflow/is_geothermal.py \
          --project-dir {PROJECT_DIR} \
          --input-csv {input.paragraphs} \
          --out-csv {output.csv} \
          --cache {params.cache} \
          --ollama-url {params.url} \
          --model {params.model} \
          {params.thinking} \
          --country "{params.country}" \
          --countries {params.countries} \
          --language {params.language} \
          --sleep-s {params.sleep_s} \
          --save-every {params.save_every}
        """


rule extract_locations:
    input:
        pattern_for("paragraph_geothermal_csv"),
    output:
        csv=pattern_for("paragraph_locations_csv"),
    params:
        cache=lambda wildcards: path_for(wildcards.language, "geo_cache"),
        model=OLLAMA["location_model"],
        thinking="--think" if OLLAMA.get("think", False) else "--no-think",
        url=OLLAMA["url"],
        sleep_s=OLLAMA["sleep_s"],
        save_every=OLLAMA["save_every"],
        country=lambda wildcards: country_for(wildcards.language),
        countries=lambda wildcards: _countries_arg(wildcards.language),
    shell:
        """
        {PYTHON} scripts/core_workflow/locations_ollama.py \
          --project-dir {PROJECT_DIR} \
          --input-csv {input} \
          --out-csv {output.csv} \
          --cache {params.cache} \
          --ollama-url {params.url} \
          --model {params.model} \
          {params.thinking} \
          --country "{params.country}" \
          --countries {params.countries} \
          --sleep-s {params.sleep_s} \
          --save-every {params.save_every}
        """


rule geocode_paragraphs_shapes:
    input:
        csv=pattern_for("paragraph_locations_csv"),
        shapes=PATHS["shapes_parquet"],
    output:
        gpkg=pattern_for("paragraphs_with_geo_shapes_gpkg"),
        csv=pattern_for("paragraph_shapes_geocoding_csv"),
    params:
        country=lambda wildcards: country_for(wildcards.language),
        countries=lambda wildcards: _countries_arg(wildcards.language),
    shell:
        """
        {PYTHON} scripts/core_workflow/geocoding_offline.py \
          --project-dir {PROJECT_DIR} \
          --input-csv {input.csv} \
          --shapes-parquet {input.shapes} \
          --output-gpkg {output.gpkg} \
          --output-csv {output.csv} \
          --country "{params.country}" \
          --countries {params.countries} \
          --points-layer paragraphs_points \
          --polygons-layer paragraphs_polygons
        """


rule geocode_paragraphs_cache_candidates:
    input:
        csv=pattern_for("paragraph_shapes_geocoding_csv"),
        shapes=PATHS["shapes_parquet"],
        overrides=lambda wildcards: _language_resource_path(wildcards.language, "location_geocoding_overrides.csv"),
    output:
        gpkg=temp("output/{language}/workflow/paragraphs_with_geo_online_candidates.gpkg"),
        csv=temp("output/{language}/workflow/paragraphs_with_geo_online_candidates.csv"),
        candidates=pattern_for("geocoding_cache_candidates_csv"),
        suggestions=temp("output/{language}/workflow/geocoding_online_candidate_suggestions.csv"),
    params:
        country=lambda wildcards: country_for(wildcards.language),
        countries=lambda wildcards: _countries_arg(wildcards.language),
        country_codes=lambda wildcards: _country_codes_for(wildcards.language),
        geonames_dir=ONLINE_GEOCODING.get("geonames_dir", "data/geonames"),
        geonames_country_codes=lambda wildcards: _geonames_country_codes_for(wildcards.language),
    shell:
        """
        {PYTHON} scripts/core_workflow/geocoding_online.py \
          --project-dir {PROJECT_DIR} \
          --input-csv {input.csv} \
          --shapes-parquet {input.shapes} \
          --output-gpkg {output.gpkg} \
          --output-csv {output.csv} \
          --geocoder-cache-path /dev/null \
          --skip-geocoder-cache \
          --overrides-csv {input.overrides} \
          --unmatched-csv {output.candidates} \
          --suggestions-csv {output.suggestions} \
          --geonames-dir {params.geonames_dir} \
          --geonames-country-codes "{params.geonames_country_codes}" \
          --country "{params.country}" \
          --countries {params.countries} \
          --country-codes "{params.country_codes}" \
          --points-layer paragraphs_points \
          --polygons-layer paragraphs_polygons
        """


rule combine_geocoding_cache_candidates:
    input:
        candidates=expand(pattern_for("geocoding_cache_candidates_csv"), language=LANGUAGES),
    output:
        PATHS["geocoding_cache_candidates_all_csv"],
    params:
        specs=_geocoding_candidate_specs(),
    shell:
        """
        {PYTHON} scripts/core_workflow/combine_geocoding_candidates.py \
          --output-csv {output} \
          --inputs {params.specs}
        """


rule fill_geocoder_cache_all:
    input:
        unmatched=PATHS["geocoding_cache_candidates_all_csv"],
    output:
        done=touch(PATHS["geocode_unmatched_online_done"]),
    params:
        cache=PATHS["geocode_unmatched_online_cache_jsonl"],
        provider=ONLINE_GEOCODING.get("provider", "none"),
        policy_ack="--nominatim-policy-ack" if ONLINE_GEOCODING.get("nominatim_policy_ack", False) else "",
        api_key_arg="--api-key " + shlex.quote(str(ONLINE_GEOCODING.get("api_key"))) if ONLINE_GEOCODING.get("api_key") else "",
        retry_errors="--retry-errors" if ONLINE_GEOCODING.get("retry_errors", False) else "",
        user_agent=ONLINE_GEOCODING.get("user_agent", "absa-geo-mapper"),
        lock_path=ONLINE_GEOCODING.get("lock_path", "cache/geocode_unmatched_online.lock"),
        print_every=ONLINE_GEOCODING.get("print_every", 25),
        min_delay_seconds=ONLINE_GEOCODING.get("min_delay_seconds", 1.1),
        timeout_seconds=ONLINE_GEOCODING.get("timeout_seconds", 10),
        max_retries=ONLINE_GEOCODING.get("max_retries", 3),
        retry_wait_seconds=ONLINE_GEOCODING.get("service_error_cooldown_seconds", 20),
        max_queries=ONLINE_GEOCODING.get("max_queries", 0),
    resources:
        nominatim=1,
    shell:
        """
        {PYTHON} scripts/core_workflow/geocode_unmatched_online.py \
          --project-dir {PROJECT_DIR} \
          --input-unmatched-csv {input.unmatched} \
          --output-cache {params.cache} \
          --provider {params.provider} \
          --user-agent {params.user_agent} \
          --lock-path {params.lock_path} \
          --print-every {params.print_every} \
          --min-delay-seconds {params.min_delay_seconds} \
          --timeout-seconds {params.timeout_seconds} \
          --max-retries {params.max_retries} \
          --retry-wait-seconds {params.retry_wait_seconds} \
          --max-queries {params.max_queries} \
          {params.api_key_arg} \
          {params.retry_errors} \
          {params.policy_ack}
        """


rule geocode_paragraphs_final:
    input:
        csv=pattern_for("paragraph_shapes_geocoding_csv"),
        shapes=PATHS["shapes_parquet"],
        overrides=lambda wildcards: _language_resource_path(wildcards.language, "location_geocoding_overrides.csv"),
        cache_done=PATHS["geocode_unmatched_online_done"],
    output:
        gpkg=pattern_for("paragraphs_with_geo_gpkg"),
        csv=pattern_for("paragraphs_with_geo_csv"),
        unmatched=pattern_for("geocoding_unmatched_csv"),
        suggestions=pattern_for("geocoding_suggestions_csv"),
    params:
        cache=PATHS["geocode_unmatched_online_cache_jsonl"],
        country=lambda wildcards: country_for(wildcards.language),
        countries=lambda wildcards: _countries_arg(wildcards.language),
        country_codes=lambda wildcards: _country_codes_for(wildcards.language),
        geonames_dir=ONLINE_GEOCODING.get("geonames_dir", "data/geonames"),
        geonames_country_codes=lambda wildcards: _geonames_country_codes_for(wildcards.language),
    shell:
        """
        {PYTHON} scripts/core_workflow/geocoding_online.py \
          --project-dir {PROJECT_DIR} \
          --input-csv {input.csv} \
          --shapes-parquet {input.shapes} \
          --output-gpkg {output.gpkg} \
          --output-csv {output.csv} \
          --geocoder-cache-path {params.cache} \
          --overrides-csv {input.overrides} \
          --unmatched-csv {output.unmatched} \
          --suggestions-csv {output.suggestions} \
          --geonames-dir {params.geonames_dir} \
          --geonames-country-codes "{params.geonames_country_codes}" \
          --country "{params.country}" \
          --countries {params.countries} \
          --country-codes "{params.country_codes}" \
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
        {PYTHON} scripts/core_workflow/split_paragraphs_to_sentences.py \
          --project-dir {PROJECT_DIR} \
          --input-csv {input} \
          --output-csv {output} \
          --language {wildcards.language}
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
        {PYTHON} scripts/core_workflow/classification_sentences.py \
          --project-dir {PROJECT_DIR} \
          --input-csv {input.csv} \
          --keywords-csv {input.keywords} \
          --output-long-csv {output.long_csv} \
          --output-short-csv {output.short_csv} \
          --keep-only-matched True
        """


rule classify_sentence_sentiment:
    input:
        pattern_for("sentence_locations_csv"),
    output:
        pattern_for("sentence_sentiment_csv"),
    params:
        cache=lambda wildcards: path_for(wildcards.language, "sentence_sentiment_cache"),
        model=config.get("sentiment_ollama", {}).get("model", "ministral-3:14b"),
        thinking="--think" if config.get("sentiment_ollama", {}).get("think", False) else "--no-think",
        ollama_url=OLLAMA["url"],
        language=lambda wildcards: wildcards.language,
        prompt_variant=config.get("sentiment_ollama", {}).get("prompt_variant", "zero_shot"),
        timeout=config.get("sentiment_ollama", {}).get("timeout", 120),
        sleep_s=config.get("sentiment_ollama", {}).get("sleep_s", 0.0),
        save_every=config.get("sentiment_ollama", {}).get("save_every", 25),
    shell:
        """
        {PYTHON} scripts/core_workflow/sentiment_classification.py \
          --project-dir {PROJECT_DIR} \
          --input-csv {input} \
          --output-csv {output} \
          --text-col sentence_text \
          --cache {params.cache} \
          --model {params.model} \
          {params.thinking} \
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
        {PYTHON} scripts/core_workflow/classification_sentences.py \
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
        shapes=PATHS["shapes_parquet"],
    output:
        gpkg=pattern_for("sentences_with_categories_admin_gpkg"),
        csv=pattern_for("sentences_with_categories_admin_csv"),
    params:
        country=lambda wildcards: country_for(wildcards.language),
        countries=lambda wildcards: _countries_arg(wildcards.language),
        location_overrides=lambda wildcards: _language_resource_path(wildcards.language, "location_province_overrides.csv"),
    shell:
        """
        {PYTHON} scripts/core_workflow/geographic_aggregation.py \
          --project-dir {PROJECT_DIR} \
          --input-gpkg {input.gpkg} \
          --input-layer sentences_with_categories \
          --shapes-parquet {input.shapes} \
          --country "{params.country}" \
          --countries {params.countries} \
          --output-gpkg {output.gpkg} \
          --output-csv {output.csv} \
          --location-province-overrides {params.location_overrides}
        """


rule visualize_article_descriptives:
    input:
        articles_csv=pattern_for("articles_csv"),
        script=str(PROJECT_DIR / "scripts" / "results_tracking" / "visualize_article_descriptives.py"),
    output:
        articles_per_year_csv=pattern_for("articles_per_year_csv"),
        articles_per_year_png=pattern_for("articles_per_year_png"),
        top_newspapers_csv=pattern_for("top_newspapers_csv"),
        top_newspapers_png=pattern_for("top_newspapers_png"),
        summary_csv=pattern_for("article_descriptives_summary_csv"),
    params:
        output_dir=lambda wildcards: path_for(wildcards.language, "figures_dir"),
        table_dir=lambda wildcards: path_for(wildcards.language, "text_results_dir"),
    shell:
        """
        {PYTHON} scripts/results_tracking/visualize_article_descriptives.py \
          --project-dir {PROJECT_DIR} \
          --articles-csv {input.articles_csv} \
          --output-dir {params.output_dir} \
          --table-dir {params.table_dir} \
          --top-n-newspapers 6
        """


rule visualize_absa_results:
    input:
        admin_csv=pattern_for("sentences_with_categories_admin_csv"),
        categories_csv=pattern_for("sentences_with_categories_short_csv"),
        keywords_csv=pattern_for("keywords_topics_csv"),
        shapes=PATHS["shapes_parquet"],
        script=str(PROJECT_DIR / "scripts" / "results_tracking" / "visualize_absa_results.py"),
    output:
        frame_keywords=directory(pattern_for("frame_keywords_dir")),
        region_frames=directory(pattern_for("region_frames_dir")),
        table=pattern_for("province_sentiment_table_csv"),
        frame_keyword_index=pattern_for("frame_keyword_figure_index_csv"),
        province_frame_counts=pattern_for("province_frame_counts_csv"),
        province_sentiment_summary=pattern_for("province_sentiment_summary_csv"),
        balance=pattern_for("provinces_sentiment_balance_png"),
        distribution=pattern_for("provinces_sentiment_distribution_png"),
        categories=pattern_for("categories_sentiment_distribution_png"),
        heatmap=pattern_for("locations_map_html"),
    params:
        output_dir=lambda wildcards: path_for(wildcards.language, "figures_dir"),
        table_dir=lambda wildcards: path_for(wildcards.language, "text_results_dir"),
        country=lambda wildcards: country_for(wildcards.language),
        countries=lambda wildcards: _countries_arg(wildcards.language),
        location_overrides=lambda wildcards: _language_resource_path(wildcards.language, "location_province_overrides.csv"),
    shell:
        """
        {PYTHON} scripts/results_tracking/visualize_absa_results.py \
          --project-dir {PROJECT_DIR} \
          --admin-csv {input.admin_csv} \
          --categories-csv {input.categories_csv} \
          --keywords-csv {input.keywords_csv} \
          --shapes-parquet {input.shapes} \
          --output-dir {params.output_dir} \
          --table-dir {params.table_dir} \
          --country "{params.country}" \
          --countries {params.countries} \
          --location-province-overrides {params.location_overrides}
        """


rule visualize_overarching_results:
    input:
        keywords_csvs=expand(pattern_for("keywords_topics_csv"), language=LANGUAGES),
        admin_csvs=expand(pattern_for("sentences_with_categories_admin_csv"), language=LANGUAGES),
        sentiment_csvs=expand(pattern_for("sentence_sentiment_csv"), language=LANGUAGES),
        shapes=PATHS["shapes_parquet"],
        script=str(PROJECT_DIR / "scripts" / "results_tracking" / "visualize_overarching_results.py"),
    output:
        extreme_balance="output/generic/figures/all_languages_extreme_province_sentiment_balance.png",
        region_keywords="output/generic/figures/keywords_frames_regions.png",
        region_keywords_table="output/generic/text/keywords_frames_regions_table.csv",
        balance="output/generic/figures/all_languages_province_sentiment_balance.png",
        province_table="output/generic/text/all_languages_province_sentiment_table.csv",
        extreme_province_frame_shares_table="output/generic/text/all_languages_extreme_province_frame_shares_table.csv",
        frame_mentions_stacked_table="output/generic/text/frame_mentions_100pct_stacked_table.csv",
        frame_mentions_stacked_png="output/generic/figures/frame_mentions_100pct_stacked.png",
        sentiment_map="output/generic/figures/all_languages_province_sentiment_map.png",
    params:
        output_dir="output/generic/figures",
        table_dir="output/generic/text",
        languages=_shell_join(LANGUAGES),
        countries=_shell_join(country_for(language) for language in LANGUAGES),
    shell:
        """
        {PYTHON} scripts/results_tracking/visualize_overarching_results.py \
          --project-dir {PROJECT_DIR} \
          --languages {params.languages} \
          --countries {params.countries} \
          --admin-csvs {input.admin_csvs} \
          --sentiment-csvs {input.sentiment_csvs} \
          --shapes-parquet {input.shapes} \
          --output-dir {params.output_dir} \
          --table-dir {params.table_dir}
        """


rule build_data_tracking_table:
    input:
        _data_tracking_inputs,
    output:
        DATA_TRACKING_CSV,
    params:
        languages=_shell_join(LANGUAGES),
    shell:
        """
        {PYTHON} scripts/results_tracking/build_data_tracking_table.py \
          --project-dir {PROJECT_DIR} \
          --languages {params.languages} \
          --output-csv {output}
        """


rule make_annotation_df:
    input:
        sentences_csv=pattern_for("sentence_locations_csv"),
        sentiment_csv=pattern_for("sentence_sentiment_csv"),
        keywords_csv=pattern_for("keywords_topics_csv"),
        paragraph_geothermal_csv=pattern_for("paragraph_geothermal_csv"),
        paragraph_location_csv=pattern_for("paragraphs_with_geo_csv"),
    output:
        pattern_for("annotation_sentences_csv"),
    params:
        province_filter_regex=ANNOTATION.get("province_filter_regex", ""),
        exclude_neutral=ANNOTATION.get("exclude_neutral", True),
        sentence_sample_size=ANNOTATION.get("sentence_sample_size", 500),
        paragraph_sample_size=ANNOTATION.get("paragraph_sample_size", 100),
        paragraph_location_sample_size=ANNOTATION.get(
            "paragraph_location_sample_size",
            ANNOTATION.get("paragraph_sample_size", 100),
        ),
        random_state=ANNOTATION.get("sample_random_state", 42),
    shell:
        """
        {PYTHON} scripts/core_workflow/make_annotation_df.py \
          --project-dir {PROJECT_DIR} \
          --input-csv {input.sentences_csv} \
          --paragraph-geothermal-csv {input.paragraph_geothermal_csv} \
          --paragraph-location-csv {input.paragraph_location_csv} \
          --sentiment-csv {input.sentiment_csv} \
          --keywords-csv {input.keywords_csv} \
          --output-csv {output} \
          --province-filter-regex "{params.province_filter_regex}" \
          --exclude-neutral {params.exclude_neutral} \
          --sentence-sample-size {params.sentence_sample_size} \
          --paragraph-sample-size {params.paragraph_sample_size} \
          --paragraph-location-sample-size {params.paragraph_location_sample_size} \
          --random-state {params.random_state}
        """


rule combine_annotation_exports:
    input:
        lambda wildcards: [path_for(language, "annotation_sentences_csv") for language in LANGUAGES]
    output:
        "annotation/sentences_for_annotation_all_languages.csv"
    shell:
        """
        {PYTHON} scripts/core_workflow/make_annotation_df.py \
          --output-csv {output} \
          --combine-inputs {input}
        """
