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


def load_shapes_parquet(path: str | Path) -> gpd.GeoDataFrame:
    df = pd.read_parquet(path)
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
    gdf = shapes[
        shapes["shape_class"].astype(str).str.lower().eq("land")
        & shapes["parent"].astype(str).str.lower().eq("nuts")
        & shapes["parent_subtype"].astype(str).eq("2")
    ].copy()
    country_ids = country_ids_for(country)
    if country_ids:
        gdf = gdf[gdf["country_id"].isin(country_ids)].copy()
    gdf["_name_norm"] = gdf["parent_name"].map(normalize_key)
    gdf["_id_norm"] = gdf["parent_id"].map(normalize_key)
    return gdf.to_crs("EPSG:4326")


def country_shapes(shapes: gpd.GeoDataFrame, country: str | None = None) -> gpd.GeoDataFrame:
    nuts = nuts2_shapes(shapes, country)
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
