from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import wkb
from shapely.geometry import Point

from helpers.country_scope import (
    COUNTRY_ALIASES,
    COUNTRY_NAMES_BY_ID,
    countries_from_value,
    country_id_for_name,
    country_ids_for_countries,
    country_name_for_id,
    standardize_country_name,
)


def normalize_key(value: object) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("’", "'")
    text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.'()]", "", text)
    return text.strip()


def shape_resource_paths(path: str | Path) -> list[Path]:
    """Include a sibling Italian supplement without changing the base file."""
    path = Path(path)
    supplement = path.with_name(f"{path.stem}_italian_nuts3.parquet")
    return [path, supplement] if supplement.exists() else [path]


def load_shapes_parquet(path: str | Path) -> gpd.GeoDataFrame:
    frames = [pd.read_parquet(resource) for resource in shape_resource_paths(path)]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if "geometry" not in df.columns:
        raise ValueError(f"Shape parquet is missing a geometry column: {path}")

    out = df.copy()
    out["geometry"] = out["geometry"].map(lambda value: wkb.loads(value) if isinstance(value, (bytes, bytearray)) else value)
    gdf = gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")
    return gdf


def country_ids_for(country: str | None) -> list[str]:
    countries = countries_from_value(country)
    ids = country_ids_for_countries(countries)
    if ids:
        return ids
    key = normalize_key(country)
    return [key.upper()] if len(key) == 3 else []


def country_aliases_for(country: str | None) -> set[str]:
    aliases: set[str] = set()
    for parsed_country in countries_from_value(country):
        canonical = standardize_country_name(parsed_country)
        if canonical:
            aliases.update(normalize_key(alias) for alias in COUNTRY_ALIASES.get(canonical, set()) | {canonical})
    for country_id in country_ids_for(country):
        aliases.add(normalize_key(country_id))
        aliases.add(normalize_key(COUNTRY_NAMES_BY_ID.get(country_id, country_id)))
    return {alias for alias in aliases if alias}


def country_alias_lookup(country: str | None) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for parsed_country in countries_from_value(country):
        canonical = standardize_country_name(parsed_country)
        country_id = country_id_for_name(canonical)
        if not canonical or not country_id:
            continue
        values = {country_id, canonical, *COUNTRY_ALIASES.get(canonical, set())}
        for value in values:
            key = normalize_key(value)
            if key:
                lookup[key] = country_id
    return lookup


def nuts2_shapes(shapes: gpd.GeoDataFrame, country: str | None = None) -> gpd.GeoDataFrame:
    return nuts_shapes(shapes, country, level=2)


def nuts_shapes(shapes: gpd.GeoDataFrame, country: str | None = None, level: int | None = None) -> gpd.GeoDataFrame:
    """Select NUTS3 for Italy and NUTS2 elsewhere, or an explicit level."""
    levels = pd.to_numeric(shapes["parent_subtype"], errors="coerce")
    wanted = shapes["country_id"].map(lambda value: 3 if value == "ITA" else 2) if level is None else level
    gdf = shapes[
        shapes["shape_class"].astype(str).str.lower().eq("land")
        & shapes["parent"].astype(str).str.lower().eq("nuts")
        & levels.eq(wanted)
    ].copy()
    country_ids = country_ids_for(country)
    if country_ids:
        gdf = gdf[gdf["country_id"].isin(country_ids)].copy()
    gdf["_name_norm"] = gdf["parent_name"].map(normalize_key)
    gdf["_id_norm"] = gdf["parent_id"].map(normalize_key)
    return gdf.to_crs("EPSG:4326")


def regional_shapes(shapes: gpd.GeoDataFrame, country: str | None = None) -> gpd.GeoDataFrame:
    """Workflow aggregation boundaries; never silently fall back to Italian NUTS2."""
    regions = nuts_shapes(shapes, country)
    scope = country_ids_for(country)
    if (not scope or "ITA" in scope) and shapes["country_id"].eq("ITA").any():
        if not regions["country_id"].eq("ITA").any():
            raise ValueError("Italian NUTS3 boundaries are missing from the shapes parquet.")
    return regions


def region_level(row: pd.Series) -> str:
    return f"nuts{int(row['parent_subtype'])}"


def set_region_fields(df: pd.DataFrame, idx, row: pd.Series) -> None:
    """Write level-specific IDs without labelling NUTS3 codes as NUTS2."""
    level = region_level(row)
    for candidate in ("nuts2", "nuts3"):
        df.at[idx, f"{candidate}_id"] = row.get("parent_id") if candidate == level else None
        df.at[idx, f"{candidate}_name"] = row.get("parent_name") if candidate == level else None
    df.at[idx, "province_code"] = row.get("parent_id")
    df.at[idx, "province_name"] = row.get("parent_name")


def country_shapes(shapes: gpd.GeoDataFrame, country: str | None = None) -> gpd.GeoDataFrame:
    nuts = regional_shapes(shapes, country)
    if nuts.empty:
        return gpd.GeoDataFrame(
            columns=["country_id", "parent_id", "parent_name", "geometry", "_name_norm", "_id_norm"],
            geometry="geometry",
            crs="EPSG:4326",
        )
    dissolved = nuts.dissolve(by="country_id", as_index=False)
    dissolved["parent_id"] = dissolved["country_id"]
    dissolved["parent_name"] = dissolved["country_id"].map(lambda value: country_name_for_id(value) or str(value))
    dissolved["_name_norm"] = dissolved["parent_name"].map(normalize_key)
    dissolved["_id_norm"] = dissolved["parent_id"].map(normalize_key)
    return dissolved[["country_id", "parent_id", "parent_name", "geometry", "_name_norm", "_id_norm"]].copy()


def centroid_point(geometry) -> Point:
    series = gpd.GeoSeries([geometry], crs="EPSG:4326")
    try:
        proj_crs = series.estimate_utm_crs()
    except Exception:
        proj_crs = "EPSG:3857"
    centroid = series.to_crs(proj_crs).centroid.to_crs("EPSG:4326").iloc[0]
    return Point(float(centroid.x), float(centroid.y))
