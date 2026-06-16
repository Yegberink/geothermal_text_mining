from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from country_scope import COUNTRY_ALIASES, standardize_country_name


KEYWORD_CSV_SEPARATOR = ";"

LANGUAGE_DIR_ALIASES = {
    "dutch": "dutch",
    "nl": "dutch",
    "nederlands": "dutch",
    "italian": "italian",
    "it": "italian",
    "italiano": "italian",
    "german": "german",
    "de": "german",
    "deutsch": "german",
}

DEFAULT_DATE_LOCALES: dict[str, dict[str, Any]] = {
    "dutch": {
        "month_translations": {
            "januari": "January",
            "februari": "February",
            "maart": "March",
            "april": "April",
            "mei": "May",
            "juni": "June",
            "juli": "July",
            "augustus": "August",
            "september": "September",
            "oktober": "October",
            "november": "November",
            "december": "December",
        },
        "weekday_names": [
            "maandag",
            "dinsdag",
            "woensdag",
            "donderdag",
            "vrijdag",
            "zaterdag",
            "zondag",
        ],
    },
    "italian": {
        "month_translations": {
            "gennaio": "January",
            "febbraio": "February",
            "marzo": "March",
            "aprile": "April",
            "maggio": "May",
            "giugno": "June",
            "luglio": "July",
            "agosto": "August",
            "settembre": "September",
            "ottobre": "October",
            "novembre": "November",
            "dicembre": "December",
        },
        "weekday_names": [
            "lunedì",
            "lunedi",
            "martedì",
            "martedi",
            "mercoledì",
            "mercoledi",
            "giovedì",
            "giovedi",
            "venerdì",
            "venerdi",
            "sabato",
            "domenica",
        ],
    },
    "german": {
        "month_translations": {
            "januar": "January",
            "februar": "February",
            "märz": "March",
            "maerz": "March",
            "april": "April",
            "mai": "May",
            "juni": "June",
            "juli": "July",
            "august": "August",
            "september": "September",
            "oktober": "October",
            "november": "November",
            "dezember": "December",
        },
        "weekday_names": [
            "montag",
            "dienstag",
            "mittwoch",
            "donnerstag",
            "freitag",
            "samstag",
            "sonntag",
        ],
    },
}

ENGLISH_WEEKDAY_NAMES = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]

DEFAULT_GEOTHERMAL_PATTERNS = {
    "dutch": [r"aardwarmte\w*", r"geotherm\w*"],
    "italian": [r"geoterm\w*", r"geotermia\w*"],
    "german": [r"geotherm\w*", r"erdwärme\w*", r"erdwaerme\w*", r"tiefengeotherm\w*"],
}

def normalize_language(language: str | None) -> str:
    value = str(language or "").strip().lower()
    return LANGUAGE_DIR_ALIASES.get(value, value or "dutch")


def vocab_dir(project_dir: Path, language: str | None) -> Path:
    return project_dir / "vocab" / normalize_language(language)


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def merge_date_locale(
    base: dict[str, Any],
    extra: dict[str, Any],
) -> dict[str, Any]:
    merged = {
        "month_translations": dict(base.get("month_translations", {}) or {}),
        "weekday_names": list(base.get("weekday_names", []) or []),
    }
    extra_months = extra.get("month_translations", {}) or {}
    if isinstance(extra_months, dict):
        merged["month_translations"].update({str(k): str(v) for k, v in extra_months.items()})
    extra_weekdays = extra.get("weekday_names", []) or []
    if isinstance(extra_weekdays, list):
        merged["weekday_names"].extend(str(day) for day in extra_weekdays)
    return merged


def load_date_locale(
    project_dir: Path,
    language: str | None,
    config_date_locale: dict[str, Any] | None = None,
) -> tuple[dict[str, str], list[str]]:
    lang = normalize_language(language)
    locale = merge_date_locale(DEFAULT_DATE_LOCALES.get(lang, {}), read_yaml(vocab_dir(project_dir, lang) / "date_locale.yaml"))
    locale = merge_date_locale(locale, config_date_locale or {})
    weekdays = list(dict.fromkeys([*locale["weekday_names"], *ENGLISH_WEEKDAY_NAMES]))
    return locale["month_translations"], weekdays


def load_geothermal_patterns(project_dir: Path, language: str | None) -> list[str]:
    lang = normalize_language(language)
    patterns = list(DEFAULT_GEOTHERMAL_PATTERNS.get(lang, DEFAULT_GEOTHERMAL_PATTERNS["dutch"]))
    path = vocab_dir(project_dir, lang) / "geothermal_patterns.txt"
    if path.exists():
        patterns.extend(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    return list(dict.fromkeys(patterns))


def read_csv_with_encoding_fallback(path: Path, **kwargs: object) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", **kwargs)


def load_keyword_csv(path: Path) -> pd.DataFrame:
    df = read_csv_with_encoding_fallback(path, sep=KEYWORD_CSV_SEPARATOR)
    return df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]


def load_location_province_overrides(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f)
        return {
            normalize_location(row.get("location", "")): str(row.get("province_name", "")).strip()
            for row in rows
            if str(row.get("location", "")).strip() and str(row.get("province_name", "")).strip()
        }


def normalize_location(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[-_]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def country_aliases(country: str | None) -> set[str]:
    canonical = standardize_country_name(country)
    if canonical:
        return {alias.lower() for alias in COUNTRY_ALIASES.get(canonical, set()) | {canonical}}
    value = str(country or "").strip().lower()
    return {value} if value else set()
