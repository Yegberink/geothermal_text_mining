from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import wkb
from shapely.geometry import Point

from language_resources import country_aliases

COUNTRY_IDS_BY_COUNTRY = {
    "netherlands": ["NLD"],
    "nederland": ["NLD"],
    "italy": ["ITA"],
    "italia": ["ITA"],
    "germany": ["DEU"],
    "deutschland": ["DEU"],
    "german-speaking countries": ["DEU", "AUT", "CHE"],
    "german speaking countries": ["DEU", "AUT", "CHE"],
}

COUNTRY_NAMES_BY_ID = {
    "AUT": "Austria",
    "CHE": "Switzerland",
    "DEU": "Germany",
    "ITA": "Italy",
    "NLD": "Netherlands",
}

COUNTRY_ALIASES_BY_ID = {
    "AUT": {"austria", "osterreich", "österreich"},
    "CHE": {"switzerland", "schweiz", "suisse", "svizzera", "switserland", "zwitserland"},
    "DEU": {"germany", "deutschland", "bundesrepublik deutschland", "brd"},
    "ITA": {"italy", "italia", "italien"},
    "NLD": {"netherlands", "nederland", "the netherlands", "holland"},
}


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
    key = normalize_key(country)
    ids = COUNTRY_IDS_BY_COUNTRY.get(key)
    if ids:
        return ids
    return [key.upper()] if len(key) == 3 else []


def country_aliases_for(country: str | None) -> set[str]:
    aliases = {normalize_key(alias) for alias in country_aliases(country)}
    for country_id in country_ids_for(country):
        aliases.add(normalize_key(country_id))
        aliases.add(normalize_key(COUNTRY_NAMES_BY_ID.get(country_id, country_id)))
        aliases.update(normalize_key(alias) for alias in COUNTRY_ALIASES_BY_ID.get(country_id, set()))
    return {alias for alias in aliases if alias}


def country_alias_lookup(country: str | None) -> dict[str, str]:
    lookup: dict[str, str] = {}
    ids = country_ids_for(country)
    for country_id in country_ids_for(country):
        values = {
            country_id,
            COUNTRY_NAMES_BY_ID.get(country_id, country_id),
            *COUNTRY_ALIASES_BY_ID.get(country_id, set()),
        }
        for value in values:
            key = normalize_key(value)
            if key:
                lookup[key] = country_id
    for alias in country_aliases(country):
        key = normalize_key(alias)
        if key and len(ids) == 1:
            lookup[key] = ids[0]
    country_key = normalize_key(country)
    if country_key and len(ids) > 1:
        lookup[country_key] = "+".join(ids)
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
    dissolved["parent_name"] = dissolved["country_id"].map(lambda value: COUNTRY_NAMES_BY_ID.get(str(value), str(value)))
    dissolved["_name_norm"] = dissolved["parent_name"].map(normalize_key)
    dissolved["_id_norm"] = dissolved["parent_id"].map(normalize_key)
    out = dissolved[["country_id", "parent_id", "parent_name", "geometry", "_name_norm", "_id_norm"]].copy()

    country_ids = country_ids_for(country)
    if len(country_ids) > 1:
        synthetic_id = "+".join(country_ids)
        synthetic_name = str(country or synthetic_id)
        geometry_union = nuts.geometry.union_all() if hasattr(nuts.geometry, "union_all") else nuts.geometry.unary_union
        synthetic = gpd.GeoDataFrame(
            [
                {
                    "country_id": synthetic_id,
                    "parent_id": synthetic_name,
                    "parent_name": synthetic_name,
                    "geometry": geometry_union,
                    "_name_norm": normalize_key(synthetic_name),
                    "_id_norm": normalize_key(synthetic_name),
                }
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )
        out = pd.concat([out, synthetic], ignore_index=True)
    return out


def centroid_point(geometry) -> Point:
    series = gpd.GeoSeries([geometry], crs="EPSG:4326")
    try:
        proj_crs = series.estimate_utm_crs()
    except Exception:
        proj_crs = "EPSG:3857"
    centroid = series.to_crs(proj_crs).centroid.to_crs("EPSG:4326").iloc[0]
    return Point(float(centroid.x), float(centroid.y))
