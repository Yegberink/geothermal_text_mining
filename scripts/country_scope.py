from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


CANONICAL_COUNTRIES = ("Germany", "Austria", "Switzerland", "Netherlands", "Italy")
GERMAN_SCOPE_COUNTRIES = ("Germany", "Austria", "Switzerland")
COUNTRY_ORDER = ("Germany", "Austria", "Switzerland", "Netherlands", "Italy")

COUNTRY_IDS_BY_NAME = {
    "Austria": "AUT",
    "Germany": "DEU",
    "Italy": "ITA",
    "Netherlands": "NLD",
    "Switzerland": "CHE",
}
COUNTRY_NAMES_BY_ID = {country_id: country for country, country_id in COUNTRY_IDS_BY_NAME.items()}
COUNTRY_NAMES_BY_ISO2 = {
    "AT": "Austria",
    "CH": "Switzerland",
    "DE": "Germany",
    "IT": "Italy",
    "NL": "Netherlands",
}

COUNTRY_ALIASES = {
    "Germany": {
        "Germany",
        "Deutschland",
        "Duitsland",
        "Germania",
        "Allemagne",
        "Bundesrepublik Deutschland",
        "BRD",
    },
    "Austria": {
        "Austria",
        "Österreich",
        "Oesterreich",
        "Oostenrijk",
        "Autriche",
    },
    "Switzerland": {
        "Switzerland",
        "Schweiz",
        "Suisse",
        "Svizzera",
        "Zwitserland",
        "Switserland",
    },
    "Netherlands": {
        "Netherlands",
        "Nederland",
        "The Netherlands",
        "Holland",
        "Paesi Bassi",
        "Niederlande",
    },
    "Italy": {
        "Italy",
        "Italia",
        "Italien",
        "Italië",
        "Italie",
    },
}

MULTI_COUNTRY_LABELS = {
    "German-speaking countries": GERMAN_SCOPE_COUNTRIES,
    "German speaking countries": GERMAN_SCOPE_COUNTRIES,
    "DACH": GERMAN_SCOPE_COUNTRIES,
    "D-A-CH": GERMAN_SCOPE_COUNTRIES,
    "Germany/Austria/Switzerland": GERMAN_SCOPE_COUNTRIES,
    "Germany, Austria and Switzerland": GERMAN_SCOPE_COUNTRIES,
    "Deutschland Österreich Schweiz": GERMAN_SCOPE_COUNTRIES,
    "Deutschland, Österreich und Schweiz": GERMAN_SCOPE_COUNTRIES,
}


def normalize_country_key(value: object) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("’", "'")
    text = re.sub(r"[-_]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.'/&,]", "", text)
    return text.strip()


COUNTRY_BY_ALIAS = {
    normalize_country_key(alias): country
    for country, aliases in COUNTRY_ALIASES.items()
    for alias in aliases | {country}
}
MULTI_COUNTRY_BY_LABEL = {
    normalize_country_key(label): tuple(countries)
    for label, countries in MULTI_COUNTRY_LABELS.items()
}


def unique_countries(countries: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    order_rank = {country: idx for idx, country in enumerate(COUNTRY_ORDER)}
    for country in countries:
        value = str(country or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return sorted(ordered, key=lambda country: (order_rank.get(country, len(order_rank)), country))


def standardize_country_name(value: object) -> str | None:
    """Return the canonical English country name for a single-country label."""
    key = normalize_country_key(value)
    if not key:
        return None
    country = COUNTRY_BY_ALIAS.get(key)
    if country:
        return country
    upper = str(value or "").strip().upper()
    return COUNTRY_NAMES_BY_ID.get(upper)


def country_name_for_id(country_id: object) -> str | None:
    value = str(country_id or "").strip().upper()
    return COUNTRY_NAMES_BY_ID.get(value) or COUNTRY_NAMES_BY_ISO2.get(value)


def country_id_for_name(country: object) -> str | None:
    canonical = standardize_country_name(country) or str(country or "").strip()
    return COUNTRY_IDS_BY_NAME.get(canonical)


def country_ids_for_countries(countries: Sequence[str]) -> list[str]:
    ids = [country_id_for_name(country) for country in countries]
    return [country_id for country_id in ids if country_id]


def _filter_allowed(countries: Iterable[str], allowed_countries: Sequence[str] | None) -> list[str]:
    values = unique_countries(countries)
    if allowed_countries is None:
        return values
    allowed = {standardize_country_name(country) or str(country).strip() for country in allowed_countries}
    return [country for country in values if country in allowed]


def country_candidates_for_label(
    value: object,
    allowed_countries: Sequence[str] | None = None,
) -> list[str]:
    """Parse explicit country labels into canonical countries.

    This is intentionally only for country-level text or explicit country candidate
    parsing. City, province, municipality, and site names should be passed to the
    geocoder without pre-normalizing them as countries.
    """
    text = str(value or "").strip()
    key = normalize_country_key(text)
    if not key:
        return []

    if key in MULTI_COUNTRY_BY_LABEL:
        return _filter_allowed(MULTI_COUNTRY_BY_LABEL[key], allowed_countries)

    single = standardize_country_name(text)
    if single:
        return _filter_allowed([single], allowed_countries)

    split_pattern = r"\s*(?:,|;|\||/|&|\band\b|\bund\b|\ben\b|\bet\b)\s*"
    parts = [part.strip() for part in re.split(split_pattern, text, flags=re.IGNORECASE) if part.strip()]
    if len(parts) <= 1:
        return []

    parsed: list[str] = []
    for part in parts:
        country = standardize_country_name(part)
        if country:
            parsed.append(country)
    return _filter_allowed(parsed, allowed_countries) if parsed else []


def is_multi_country_label(value: object, allowed_countries: Sequence[str] | None = None) -> bool:
    return len(country_candidates_for_label(value, allowed_countries=allowed_countries)) > 1


def country_candidates_json(countries: Sequence[str] | None) -> str:
    return json.dumps(list(countries or []), ensure_ascii=False)


def parse_country_candidates_json(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return country_candidates_for_label(text)
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item).strip()]
    return []


def _countries_from_scalar(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    candidates = country_candidates_for_label(text)
    if candidates:
        return candidates
    country = standardize_country_name(text)
    return [country] if country else [text]


def countries_from_value(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return _countries_from_scalar(value)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        countries: list[str] = []
        for item in value:
            countries.extend(_countries_from_scalar(item))
        return unique_countries(countries)
    return _countries_from_scalar(value)


@dataclass(frozen=True)
class CountryScope:
    countries: tuple[str, ...]

    @property
    def is_single_country(self) -> bool:
        return len(self.countries) == 1

    @property
    def is_multi_country(self) -> bool:
        return len(self.countries) > 1

    @property
    def primary_country(self) -> str | None:
        return self.countries[0] if self.is_single_country else None

    @property
    def label(self) -> str:
        return ", ".join(self.countries)

    @property
    def cache_key(self) -> str:
        return "|".join(self.countries)


def country_scope_from_countries(countries: Sequence[str] | str | None, fallback_country: str | None = None) -> CountryScope:
    parsed = countries_from_value(countries)
    if not parsed and fallback_country:
        parsed = countries_from_value(fallback_country)
    return CountryScope(tuple(unique_countries(parsed)))


def country_scope_from_config(config: dict[str, Any], language: str | None = None) -> CountryScope:
    lang = str(language or config.get("language") or "").strip()
    scope_cfg = config.get("country_scope", {}) or {}
    if isinstance(scope_cfg, dict):
        by_language = scope_cfg.get("countries_by_language") or scope_cfg.get("by_language") or {}
        if lang and isinstance(by_language, dict) and lang in by_language:
            return country_scope_from_countries(by_language[lang])

        countries_cfg = scope_cfg.get("countries")
        if isinstance(countries_cfg, dict) and lang and lang in countries_cfg:
            return country_scope_from_countries(countries_cfg[lang])
        if countries_cfg:
            return country_scope_from_countries(countries_cfg)

    legacy_countries = config.get("countries", {}) or {}
    if isinstance(legacy_countries, dict) and lang and lang in legacy_countries:
        return country_scope_from_countries(legacy_countries[lang])

    return country_scope_from_countries(config.get("country"), fallback_country=lang)


def country_scope_from_args(
    country: str | None = None,
    countries: Sequence[str] | str | None = None,
    country_scope: str | None = None,
) -> CountryScope:
    if countries:
        return country_scope_from_countries(countries)
    if country_scope:
        return country_scope_from_countries(country_scope)
    return country_scope_from_countries(country)


def country_assignment_for_location(
    location: object,
    granularity: object,
    scope: CountryScope,
) -> dict[str, Any]:
    gran = str(granularity or "").strip().lower()
    candidates = country_candidates_for_label(location, allowed_countries=scope.countries or None)
    country_context = gran == "country" or bool(candidates)

    if not country_context:
        return {
            "llm_country": None,
            "llm_country_candidates": country_candidates_json([]),
            "llm_country_assignment_type": "not_country_level",
            "llm_has_single_country": False,
            "llm_multi_country_scope": scope.is_multi_country,
        }

    if len(candidates) == 1:
        return {
            "llm_country": candidates[0],
            "llm_country_candidates": country_candidates_json(candidates),
            "llm_country_assignment_type": "single_country",
            "llm_has_single_country": True,
            "llm_multi_country_scope": scope.is_multi_country,
        }
    if len(candidates) > 1:
        return {
            "llm_country": None,
            "llm_country_candidates": country_candidates_json(candidates),
            "llm_country_assignment_type": "multiple_countries",
            "llm_has_single_country": False,
            "llm_multi_country_scope": scope.is_multi_country,
        }

    return {
        "llm_country": None,
        "llm_country_candidates": country_candidates_json([]),
        "llm_country_assignment_type": "no_country",
        "llm_has_single_country": False,
        "llm_multi_country_scope": scope.is_multi_country,
    }


def single_country_from_row(row: Any) -> str | None:
    assignment = str(row.get("llm_country_assignment_type", "") or "").strip()
    country = standardize_country_name(row.get("llm_country"))
    if assignment == "single_country" and country:
        return country
    country_id = country_name_for_id(row.get("country_id"))
    if country_id and assignment != "multiple_countries":
        return country_id
    return None
