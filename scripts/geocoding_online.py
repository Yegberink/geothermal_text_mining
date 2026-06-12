#!/usr/bin/env python3

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from shape_resources import centroid_point, load_shapes_parquet, normalize_key, nuts2_shapes

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[1]

EXTERNAL_LOCATION_POINTS = {
    "kenia": ("Kenya", -0.0236, 37.9062),
    "kenya": ("Kenya", -0.0236, 37.9062),
    "calefornia": ("California, United States", 36.7783, -119.4179),
    "california": ("California, United States", 36.7783, -119.4179),
    "californie": ("California, United States", 36.7783, -119.4179),
    "curacao": ("Curaçao", 12.1696, -68.9900),
    "curaçao": ("Curaçao", 12.1696, -68.9900),
}

OVERRIDE_COLUMNS = [
    "location",
    "action",
    "target_type",
    "target_name",
    "country_id",
    "nuts2_id",
    "lat",
    "lon",
    "notes",
]

GEONAMES_BASE_URL = "https://download.geonames.org/export/dump"
GEONAMES_COLUMNS = [
    "geonameid",
    "name",
    "asciiname",
    "alternatenames",
    "latitude",
    "longitude",
    "feature_class",
    "feature_code",
    "country_code",
    "cc2",
    "admin1_code",
    "admin2_code",
    "admin3_code",
    "admin4_code",
    "population",
    "elevation",
    "dem",
    "timezone",
    "modification_date",
]
GEONAMES_FEATURE_PRIORITY = {
    "PPLC": 0,
    "PPLA": 5,
    "PPLA2": 6,
    "PPLA3": 7,
    "PPLA4": 8,
    "PPL": 10,
    "PPLX": 15,
    "ADM1": 20,
    "ADM2": 21,
    "ADM3": 22,
    "ADM4": 23,
}
GEONAMES_CLASS_PRIORITY = {
    "P": 10,
    "A": 20,
    "L": 35,
    "T": 40,
    "V": 45,
    "S": 60,
}


def configure_ssl_cert_bundle() -> None:
    if os.environ.get("SSL_CERT_FILE") and os.environ.get("REQUESTS_CA_BUNDLE"):
        return

    candidates = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "ssl" / "cert.pem")
    candidates.append(Path(sys.prefix) / "ssl" / "cert.pem")

    try:
        import certifi

        candidates.append(Path(certifi.where()))
    except Exception:
        pass

    for candidate in candidates:
        if candidate.exists():
            os.environ.setdefault("SSL_CERT_FILE", str(candidate))
            os.environ.setdefault("REQUESTS_CA_BUNDLE", str(candidate))
            return


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=str, default=str(DEFAULT_PROJECT_DIR))
    ap.add_argument("--input-csv", type=str, default="output/text/paragraph_shapes_geocoding.csv")
    ap.add_argument("--shapes-parquet", type=str, default="data/shapes.parquet")
    ap.add_argument("--output-gpkg", type=str, default="output/text/paragraphs_with_geo.gpkg")
    ap.add_argument("--output-csv", type=str, default="output/text/paragraphs_with_geo.csv")
    ap.add_argument(
        "--geocoder-cache-path",
        "--cache-path",
        dest="geocoder_cache_path",
        type=str,
        default="cache/geocoder_cache.jsonl",
    )
    ap.add_argument("--extra-geocoder-cache-path", action="append", default=[])
    ap.add_argument("--overrides-csv", type=str, default="")
    ap.add_argument("--unmatched-csv", type=str, default="cache/geocoding_unmatched.csv")
    ap.add_argument("--suggestions-csv", type=str, default="cache/geocoding_suggestions.csv")
    ap.add_argument("--skip-geocoder-cache", action="store_true")
    ap.add_argument("--geonames-dir", type=str, default="cache/geonames")
    ap.add_argument("--geonames-country-codes", type=str, default="")
    ap.add_argument("--disable-geonames", action="store_true")
    ap.add_argument("--download-geonames", action="store_true")
    ap.add_argument("--disable-nominatim", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--country", type=str, default="Netherlands")
    ap.add_argument("--country-codes", type=str, default="")
    ap.add_argument("--attach-nuts2-polygons-for-online-points", action="store_true")
    ap.add_argument("--points-layer", type=str, default="paragraphs_points")
    ap.add_argument("--polygons-layer", type=str, default="paragraphs_polygons")
    return ap.parse_args()


def parse_country_codes(value: str) -> list[str]:
    return [part.strip().upper() for part in str(value or "").split(",") if part.strip()]


def geonames_zip_path(geonames_dir: Path, country_code: str) -> Path:
    return geonames_dir / f"{country_code.upper()}.zip"


def ensure_geonames_zip(geonames_dir: Path, country_code: str, allow_download: bool = False) -> Path:
    country_code = country_code.upper()
    path = geonames_zip_path(geonames_dir, country_code)
    if path.exists():
        return path
    if not allow_download:
        raise FileNotFoundError(
            f"Missing GeoNames gazetteer {path}. Download it deliberately first, or rerun with "
            "--download-geonames if network access is intended."
        )

    geonames_dir.mkdir(parents=True, exist_ok=True)
    url = f"{GEONAMES_BASE_URL}/{country_code}.zip"
    print(f"Downloading GeoNames gazetteer: {url}")
    urllib.request.urlretrieve(url, path)
    return path


def read_geonames_country_zip(path: Path, country_code: str) -> pd.DataFrame:
    country_code = country_code.upper()
    with zipfile.ZipFile(path) as zf:
        txt_members = [name for name in zf.namelist() if name.lower().endswith(".txt")]
        member = f"{country_code}.txt" if f"{country_code}.txt" in txt_members else txt_members[0]
        with zf.open(member) as handle:
            df = pd.read_csv(
                handle,
                sep="\t",
                names=GEONAMES_COLUMNS,
                dtype=str,
                keep_default_na=False,
                na_values=[],
            )
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["population"] = pd.to_numeric(df["population"], errors="coerce").fillna(0).astype("int64")
    return df.dropna(subset=["latitude", "longitude"]).copy()


def geonames_priority(row: pd.Series) -> int:
    feature_code = str(row.get("feature_code") or "")
    feature_class = str(row.get("feature_class") or "")
    return GEONAMES_FEATURE_PRIORITY.get(feature_code, GEONAMES_CLASS_PRIORITY.get(feature_class, 99))


def geonames_name_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    base_cols = [
        "geonameid",
        "name",
        "asciiname",
        "latitude",
        "longitude",
        "feature_class",
        "feature_code",
        "country_code",
        "population",
        "_priority",
    ]
    for col in ["name", "asciiname"]:
        part = df[base_cols].copy()
        part["_match_name"] = df[col].astype(str)
        rows.append(part)

    alt = df[["alternatenames", *base_cols]].copy()
    alt["alternatenames"] = alt["alternatenames"].astype(str).str.split(",")
    alt = alt.explode("alternatenames")
    alt["_match_name"] = alt["alternatenames"].astype(str)
    rows.append(alt.drop(columns=["alternatenames"]))

    names = pd.concat(rows, ignore_index=True)
    names["_match_norm"] = names["_match_name"].map(normalize_key)
    names = names[names["_match_norm"].ne("")].copy()
    names = names.sort_values(["_match_norm", "_priority", "population"], ascending=[True, True, False])
    return names.drop_duplicates("_match_norm", keep="first")


def load_geonames_gazetteer(
    geonames_dir: Path,
    country_codes: list[str],
    allow_download: bool = False,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for country_code in country_codes:
        path = ensure_geonames_zip(geonames_dir, country_code, allow_download=allow_download)
        frame = read_geonames_country_zip(path, country_code)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()

    gazetteer = pd.concat(frames, ignore_index=True)
    gazetteer = gazetteer[gazetteer["feature_class"].isin(["A", "P", "L", "T", "V", "S"])].copy()
    gazetteer["_priority"] = gazetteer.apply(geonames_priority, axis=1)
    return geonames_name_rows(gazetteer)


def useful_granularity_mask(df: pd.DataFrame) -> pd.Series:
    gran = df["_gran"].astype(str).str.strip().str.lower()
    return ~gran.isin(["", "none", "country"])


def ignored_location_mask(df: pd.DataFrame) -> pd.Series:
    return df["geo_match_type"].astype(str).str.lower().eq("ignored_location")


def has_geometry_mask(df: pd.DataFrame) -> pd.Series:
    return df["geom_point_wkt"].notna() | df["geom_poly_wkt"].notna()


def load_location_geocoding_overrides(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        out = pd.DataFrame(columns=OVERRIDE_COLUMNS)
    else:
        out = pd.read_csv(path, dtype=str, keep_default_na=False)
        for col in OVERRIDE_COLUMNS:
            if col not in out.columns:
                out[col] = ""
        out = out[OVERRIDE_COLUMNS].copy()

    out = out.fillna("")
    out["action"] = out["action"].astype(str).str.strip().str.lower()
    out["target_type"] = out["target_type"].astype(str).str.strip().str.lower()
    out["_location_norm"] = out["location"].map(normalize_key)
    return out[out["_location_norm"].ne("")].copy()


def build_nuts2_lookup(nuts2: gpd.GeoDataFrame) -> dict[str, pd.Series]:
    lookup: dict[str, pd.Series] = {}
    for _, row in nuts2.iterrows():
        for value in [row.get("parent_id"), row.get("parent_name")]:
            key = normalize_key(value)
            if key:
                lookup.setdefault(key, row)
    return lookup


def resolve_nuts2_override(override: pd.Series, nuts2_lookup: dict[str, pd.Series]) -> pd.Series | None:
    for value in [override.get("nuts2_id"), override.get("target_name")]:
        key = normalize_key(value)
        if key and key in nuts2_lookup:
            return nuts2_lookup[key]
    return None


def apply_nuts2_geometry(df: pd.DataFrame, idx: Any, row: pd.Series, source: str, match_type: str) -> None:
    geom = row.geometry
    point = centroid_point(geom)
    df.at[idx, "geo_level"] = "nuts2"
    df.at[idx, "geo_source"] = source
    df.at[idx, "geo_match_type"] = match_type
    df.at[idx, "geo_name_matched"] = row.get("parent_name")
    df.at[idx, "geo_lat"] = float(point.y)
    df.at[idx, "geo_lon"] = float(point.x)
    df.at[idx, "geom_point_wkt"] = point.wkt
    df.at[idx, "geom_poly_wkt"] = geom.wkt
    df.at[idx, "country_id"] = row.get("country_id")
    df.at[idx, "nuts2_id"] = row.get("parent_id")
    df.at[idx, "nuts2_name"] = row.get("parent_name")
    df.at[idx, "province_code"] = row.get("parent_id")
    df.at[idx, "province_name"] = row.get("parent_name")


def parse_float(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def apply_location_geocoding_overrides(
    df: pd.DataFrame,
    overrides: pd.DataFrame,
    nuts2: gpd.GeoDataFrame,
) -> pd.DataFrame:
    if overrides.empty:
        print("[workflow_table] paragraphs_geocoded_overrides: 0")
        print("[workflow_table] paragraphs_ignored_overrides: 0")
        return df

    override_lookup = overrides.drop_duplicates("_location_norm", keep="last").set_index("_location_norm")
    nuts2_lookup = build_nuts2_lookup(nuts2)
    eligible = ~has_geometry_mask(df) & df["_loc_first"].astype(str).str.strip().ne("")
    matched = 0
    ignored = 0

    for idx, loc_norm in df.loc[eligible, "_loc_first"].map(normalize_key).items():
        if loc_norm not in override_lookup.index:
            continue
        override = override_lookup.loc[loc_norm]
        action = str(override.get("action") or "").strip().lower()
        display_name = str(override.get("target_name") or override.get("location") or df.at[idx, "_loc_first"]).strip()

        if action == "ignore":
            df.at[idx, "geo_level"] = "ignored"
            df.at[idx, "geo_source"] = "manual_override"
            df.at[idx, "geo_name_matched"] = display_name
            df.at[idx, "geo_match_type"] = "ignored_location"
            ignored += 1
            continue

        if action == "point":
            lat = parse_float(override.get("lat"))
            lon = parse_float(override.get("lon"))
            if lat is None or lon is None:
                print(f"[warning] Skipping point override without valid lat/lon for {override.get('location')!r}")
                continue
            df.at[idx, "geo_level"] = str(df.at[idx, "_gran"] or "point")
            df.at[idx, "geo_source"] = "manual_override"
            df.at[idx, "geo_name_matched"] = display_name
            df.at[idx, "geo_match_type"] = "override_point"
            df.at[idx, "geo_lat"] = lat
            df.at[idx, "geo_lon"] = lon
            df.at[idx, "geom_point_wkt"] = Point(lon, lat).wkt
            if str(override.get("country_id") or "").strip():
                df.at[idx, "country_id"] = str(override.get("country_id")).strip()
            matched += 1
            continue

        if action == "nuts2":
            nuts2_row = resolve_nuts2_override(override, nuts2_lookup)
            if nuts2_row is None:
                print(f"[warning] Skipping NUTS2 override without matching region for {override.get('location')!r}")
                continue
            apply_nuts2_geometry(df, idx, nuts2_row, "manual_override", "override_nuts2")
            matched += 1
            continue

        print(f"[warning] Unknown geocoding override action {action!r} for {override.get('location')!r}")

    print(f"[workflow_table] paragraphs_geocoded_overrides: {matched}")
    print(f"[workflow_table] paragraphs_ignored_overrides: {ignored}")
    return df


def apply_geonames_matches(df: pd.DataFrame, gazetteer: pd.DataFrame) -> pd.DataFrame:
    if gazetteer.empty:
        return df

    has_geo = has_geometry_mask(df)
    needs_point = (
        ~has_geo
        & ~ignored_location_mask(df)
        & useful_granularity_mask(df)
        & df["_loc_first"].astype(str).str.strip().ne("")
    )
    if not needs_point.any():
        return df

    lookup = gazetteer.set_index("_match_norm")
    matched = 0
    for idx, loc_norm in df.loc[needs_point, "_loc_first"].map(normalize_key).items():
        if loc_norm not in lookup.index:
            continue
        row = lookup.loc[loc_norm]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        display_name = str(row.get("name") or row.get("_match_name") or df.at[idx, "_loc_first"])
        df.at[idx, "geo_lat"] = lat
        df.at[idx, "geo_lon"] = lon
        df.at[idx, "geom_point_wkt"] = Point(lon, lat).wkt
        df.at[idx, "geo_level"] = str(df.at[idx, "_gran"] or "gazetteer_point")
        df.at[idx, "geo_source"] = "geonames"
        df.at[idx, "geo_name_matched"] = display_name
        df.at[idx, "geo_match_type"] = "geonames_exact"
        df.at[idx, "country_id"] = row.get("country_code")
        matched += 1
    print(f"[workflow_table] paragraphs_geocoded_geonames: {matched}")
    return df


def bias_query(query: str, country: str) -> str:
    query = str(query or "").strip()
    country = str(country or "").strip()
    if not query or not country:
        return query
    if "speaking countries" in country.lower():
        return query
    if re.search(rf"\b{re.escape(country)}\b", query, flags=re.IGNORECASE):
        return query
    return f"{query}, {country}"


def geocoder_cache_key(query: str, country_codes: str) -> str:
    return f"{str(query or '').strip()}||{country_codes or ''}"


def load_geocoder_cache(cache_path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if not cache_path.exists():
        return cache

    if cache_path.suffix.lower() == ".json":
        with cache_path.open("r", encoding="utf-8") as f:
            legacy = json.load(f)
        for key, value in legacy.items():
            if isinstance(value, dict):
                value.setdefault("provider", "nominatim")
                cache[key] = value
        return cache

    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("status") != "ok" or not isinstance(entry.get("result"), dict):
                continue
            key = str(entry.get("key") or geocoder_cache_key(entry.get("query", ""), entry.get("country_codes", "")))
            result = dict(entry["result"])
            result.setdefault("provider", entry.get("provider") or "geocoder")
            cache[key] = result
    return cache


def cached_result_for_query(
    query: str,
    cache: dict[str, dict],
    country: str,
    country_codes: str,
) -> dict | None:
    query = str(query or "").strip()
    if not query:
        return None
    candidates = [query, bias_query(query, country)]
    for candidate in dict.fromkeys(candidates):
        result = cache.get(geocoder_cache_key(candidate, country_codes))
        if result:
            return result
    return None


def apply_cached_geocoder_matches(
    df: pd.DataFrame,
    cache: dict[str, dict],
    country: str,
    country_codes: str,
) -> pd.DataFrame:
    if not cache:
        print("[workflow_table] paragraphs_geocoded_geocoder_cache: 0")
        return df

    eligible = (
        ~has_geometry_mask(df)
        & ~ignored_location_mask(df)
        & useful_granularity_mask(df)
        & df["_loc_first"].astype(str).str.strip().ne("")
    )
    matched = 0
    for idx, query in df.loc[eligible, "_loc_first"].astype(str).str.strip().items():
        result = cached_result_for_query(query, cache, country, country_codes)
        if not result:
            continue
        lat = parse_float(result.get("lat"))
        lon = parse_float(result.get("lon"))
        if lat is None or lon is None:
            continue
        provider = normalize_key(result.get("provider") or result.get("source") or "geocoder").replace(" ", "_")
        display_name = str(result.get("disp") or result.get("display_name") or result.get("name") or query)
        df.at[idx, "geo_lat"] = lat
        df.at[idx, "geo_lon"] = lon
        df.at[idx, "geom_point_wkt"] = Point(lon, lat).wkt
        df.at[idx, "geo_level"] = str(df.at[idx, "_gran"] or "point")
        df.at[idx, "geo_source"] = f"{provider}_cache"
        df.at[idx, "geo_name_matched"] = display_name
        df.at[idx, "geo_match_type"] = f"{provider}_cache"
        matched += 1

    print(f"[workflow_table] paragraphs_geocoded_geocoder_cache: {matched}")
    return df


def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["geo_level", "geo_name_matched", "geo_source", "geo_match_type"]:
        if col not in df.columns:
            df[col] = None
    for col in ["geo_lat", "geo_lon"]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")
    for col in ["geom_point_wkt", "geom_poly_wkt", "country_id", "nuts2_id", "nuts2_name", "province_code", "province_name"]:
        if col not in df.columns:
            df[col] = None
    if "_loc_first" not in df.columns:
        df["_loc_first"] = df["llm_location"].astype(str) if "llm_location" in df.columns else ""
    if "_gran" not in df.columns:
        df["_gran"] = (
            df["llm_granularity"].astype(str).str.strip().str.lower()
            if "llm_granularity" in df.columns
            else ""
        )
    return df


def apply_nuts2_assignment(df: pd.DataFrame, nuts2: gpd.GeoDataFrame, attach_polygons: bool) -> pd.DataFrame:
    source = df["geo_source"].astype(str)
    point_rows = (
        df["geom_point_wkt"].notna()
        & ~ignored_location_mask(df)
        & (
            source.isin(["geonames", "manual_override"])
            | source.str.endswith("_cache", na=False)
        )
    )
    if not point_rows.any() or nuts2.empty:
        return df

    points = df.loc[point_rows, ["geom_point_wkt"]].copy()
    points_gdf = gpd.GeoDataFrame(
        points,
        geometry=gpd.GeoSeries.from_wkt(points["geom_point_wkt"]),
        crs="EPSG:4326",
    )
    regions = nuts2[["country_id", "parent_id", "parent_name", "geometry"]].copy().to_crs(points_gdf.crs)
    joined = gpd.sjoin(points_gdf, regions, predicate="within", how="left").drop(columns=["index_right"], errors="ignore")
    matched = joined["parent_id"].notna()
    if not matched.any():
        return df

    idxs = joined.index[matched]
    df.loc[idxs, "country_id"] = joined.loc[matched, "country_id"].values
    df.loc[idxs, "nuts2_id"] = joined.loc[matched, "parent_id"].values
    df.loc[idxs, "nuts2_name"] = joined.loc[matched, "parent_name"].values
    df.loc[idxs, "province_code"] = joined.loc[matched, "parent_id"].values
    df.loc[idxs, "province_name"] = joined.loc[matched, "parent_name"].values
    empty_match_type = df.loc[idxs, "geo_match_type"].isna() | df.loc[idxs, "geo_match_type"].astype(str).str.strip().eq("")
    if empty_match_type.any():
        df.loc[empty_match_type[empty_match_type].index, "geo_match_type"] = "point_then_nuts2"

    if attach_polygons:
        region_lookup = regions.set_index("parent_id")["geometry"].to_dict()
        for idx in idxs:
            region_id = df.at[idx, "nuts2_id"]
            geom = region_lookup.get(region_id)
            if geom is not None:
                df.at[idx, "geom_poly_wkt"] = geom.wkt
    return df


def unmatched_location_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df["_loc_first"].astype(str).str.strip().ne("")
        & useful_granularity_mask(df)
        & ~has_geometry_mask(df)
        & ~ignored_location_mask(df)
    )


def text_examples(group: pd.DataFrame, limit: int = 3, width: int = 180) -> str:
    if "paragraph_text" not in group.columns:
        return ""
    examples = []
    for value in group["paragraph_text"].dropna().astype(str):
        text = re.sub(r"\s+", " ", value).strip()
        if text:
            examples.append(text[:width])
        if len(examples) >= limit:
            break
    return " || ".join(examples)


def mode_values(series: pd.Series, limit: int = 3) -> str:
    values = series.dropna().astype(str).str.strip()
    values = values[values.ne("")]
    if values.empty:
        return ""
    return "; ".join(values.value_counts().head(limit).index.tolist())


def build_unmatched_report(df: pd.DataFrame) -> pd.DataFrame:
    mask = unmatched_location_mask(df)
    columns = ["location", "normalized_location", "granularity", "row_count", "location_source", "examples"]
    if not mask.any():
        return pd.DataFrame(columns=columns)

    work = df.loc[mask].copy()
    work["_report_norm"] = work["_loc_first"].map(normalize_key)
    rows: list[dict[str, Any]] = []
    for loc_norm, group in work.groupby("_report_norm", sort=True):
        rows.append(
            {
                "location": mode_values(group["_loc_first"], limit=1),
                "normalized_location": loc_norm,
                "granularity": mode_values(group["_gran"]),
                "row_count": len(group),
                "location_source": mode_values(
                    group["llm_location_source"]
                    if "llm_location_source" in group.columns
                    else pd.Series("", index=group.index)
                ),
                "examples": text_examples(group),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["row_count", "location"], ascending=[False, True])


def write_unmatched_report(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    report = build_unmatched_report(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False, encoding="utf-8")
    print(f"[workflow_table] geocoding_unmatched_unique_locations: {len(report)}")
    print(f"Wrote geocoding unmatched report: {path}")
    return report


def suggestion_candidates(
    nuts2: gpd.GeoDataFrame,
    gazetteer: pd.DataFrame,
    overrides: pd.DataFrame,
) -> dict[str, tuple[str, str]]:
    candidates: dict[str, tuple[str, str]] = {}
    for _, row in nuts2.iterrows():
        for value in [row.get("parent_name"), row.get("parent_id")]:
            key = normalize_key(value)
            if key:
                candidates.setdefault(key, (str(value), "nuts2"))
    if not gazetteer.empty:
        for _, row in gazetteer.iterrows():
            value = str(row.get("_match_name") or row.get("name") or "").strip()
            key = normalize_key(value)
            if key:
                candidates.setdefault(key, (value, "geonames"))
    if not overrides.empty:
        for _, row in overrides.iterrows():
            value = str(row.get("location") or "").strip()
            key = normalize_key(value)
            if key:
                candidates.setdefault(key, (value, "override"))
    return candidates


def write_suggestions_report(
    unmatched: pd.DataFrame,
    path: Path,
    candidates: dict[str, tuple[str, str]],
) -> None:
    columns = [
        "location",
        "normalized_location",
        "row_count",
        "suggestion_1",
        "suggestion_1_source",
        "suggestion_2",
        "suggestion_2_source",
        "suggestion_3",
        "suggestion_3_source",
    ]
    rows: list[dict[str, Any]] = []
    candidate_keys = sorted(candidates)
    for _, row in unmatched.iterrows():
        loc_norm = str(row.get("normalized_location") or "")
        matches = [m for m in difflib.get_close_matches(loc_norm, candidate_keys, n=3, cutoff=0.78) if m != loc_norm]
        out = {
            "location": row.get("location", ""),
            "normalized_location": loc_norm,
            "row_count": row.get("row_count", 0),
            "suggestion_1": "",
            "suggestion_1_source": "",
            "suggestion_2": "",
            "suggestion_2_source": "",
            "suggestion_3": "",
            "suggestion_3_source": "",
        }
        for i, match in enumerate(matches, start=1):
            value, source = candidates[match]
            out[f"suggestion_{i}"] = value
            out[f"suggestion_{i}_source"] = source
        rows.append(out)

    suggestions = pd.DataFrame(rows, columns=columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    suggestions.to_csv(path, index=False, encoding="utf-8")
    print(f"Wrote geocoding suggestions report: {path}")


def write_outputs(df: pd.DataFrame, output_gpkg: Path, output_csv: Path, points_layer: str, polygons_layer: str) -> None:
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_gpkg.exists():
        output_gpkg.unlink()

    df_points = df[df["geom_point_wkt"].notna()].copy()
    if not df_points.empty:
        gdf_points = gpd.GeoDataFrame(
            df_points,
            geometry=gpd.GeoSeries.from_wkt(df_points["geom_point_wkt"]),
            crs="EPSG:4326",
        )
        gdf_points.to_file(output_gpkg, layer=points_layer, driver="GPKG")

    df_polys = df[df["geom_poly_wkt"].notna()].copy()
    if not df_polys.empty:
        gdf_polys = gpd.GeoDataFrame(
            df_polys,
            geometry=gpd.GeoSeries.from_wkt(df_polys["geom_poly_wkt"]),
            crs="EPSG:4326",
        )
        gdf_polys.to_file(output_gpkg, layer=polygons_layer, driver="GPKG")

    df.to_csv(output_csv, index=False, encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    os.chdir(project_dir)

    df = pd.read_csv(args.input_csv, dtype=str)
    df = ensure_output_columns(df)
    shapes = load_shapes_parquet(args.shapes_parquet)
    nuts2 = nuts2_shapes(shapes, args.country)
    gazetteer = pd.DataFrame()

    for idx, loc_norm in df["_loc_first"].map(normalize_key).items():
        if loc_norm not in EXTERNAL_LOCATION_POINTS:
            continue
        display_name, lat, lon = EXTERNAL_LOCATION_POINTS[loc_norm]
        df.at[idx, "geo_level"] = "external"
        df.at[idx, "geo_source"] = "manual_override"
        df.at[idx, "geo_name_matched"] = display_name
        df.at[idx, "geo_match_type"] = "override_point"
        df.at[idx, "geo_lat"] = float(lat)
        df.at[idx, "geo_lon"] = float(lon)
        df.at[idx, "geom_point_wkt"] = Point(float(lon), float(lat)).wkt
        df.at[idx, "geom_poly_wkt"] = None

    overrides = load_location_geocoding_overrides(Path(args.overrides_csv) if args.overrides_csv else None)
    df = apply_location_geocoding_overrides(df, overrides, nuts2)

    if not args.disable_geonames:
        geonames_country_codes = parse_country_codes(args.geonames_country_codes or args.country_codes)
        if geonames_country_codes:
            gazetteer = load_geonames_gazetteer(
                Path(args.geonames_dir),
                geonames_country_codes,
                allow_download=args.download_geonames,
            )
            df = apply_geonames_matches(df, gazetteer)
            df = apply_nuts2_assignment(df, nuts2, args.attach_nuts2_polygons_for_online_points)
        else:
            print("GeoNames lookup skipped: no country codes configured.")

    if args.skip_geocoder_cache:
        cache = {}
    else:
        cache = load_geocoder_cache(Path(args.geocoder_cache_path))
        for extra_cache_path in args.extra_geocoder_cache_path:
            cache.update(load_geocoder_cache(Path(extra_cache_path)))
    df = apply_cached_geocoder_matches(df, cache, args.country, args.country_codes)

    df = apply_nuts2_assignment(df, nuts2, args.attach_nuts2_polygons_for_online_points)
    unmatched = write_unmatched_report(df, Path(args.unmatched_csv))
    write_suggestions_report(
        unmatched,
        Path(args.suggestions_csv),
        suggestion_candidates(nuts2, gazetteer, overrides),
    )

    write_outputs(
        df,
        Path(args.output_gpkg),
        Path(args.output_csv),
        args.points_layer,
        args.polygons_layer,
    )

    has_geo = has_geometry_mask(df)
    print(f"[workflow_table] paragraphs_after_final_geocoding: {len(df)}")
    print(f"[workflow_table] paragraphs_geocoded_final: {int(has_geo.sum())}")
    print(f"[workflow_table] paragraphs_not_geocoded_final: {int((~has_geo).sum())}")


if __name__ == "__main__":
    main()
