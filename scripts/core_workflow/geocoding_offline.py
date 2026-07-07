#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd

from helpers.country_scope import (
    CountryScope,
    country_assignment_for_location,
    country_scope_from_args,
    country_candidates_for_label,
    country_candidates_json,
    country_name_for_id,
)
from helpers.shape_resources import (
    centroid_point,
    country_alias_lookup,
    country_shapes,
    load_shapes_parquet,
    normalize_key,
    nuts2_shapes,
)

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=str, default=str(DEFAULT_PROJECT_DIR))
    ap.add_argument("--input-csv", type=str, default="output/workflow/paragraph_locations_ollama.csv")
    ap.add_argument("--shapes-parquet", type=str, default="data/shapes.parquet")
    ap.add_argument("--output-gpkg", type=str, default="output/workflow/paragraphs_with_geo.gpkg")
    ap.add_argument("--output-csv", type=str, default="output/workflow/paragraphs_with_geo.csv")
    ap.add_argument("--country", type=str, default="Netherlands", help="Backward-compatible single-country shorthand.")
    ap.add_argument("--countries", nargs="+", default=None)
    ap.add_argument("--country-scope", type=str, default="")
    ap.add_argument("--points-layer", type=str, default="paragraphs_points")
    ap.add_argument("--polygons-layer", type=str, default="paragraphs_polygons")
    return ap.parse_args()


def clean_loc(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(
        r"^\s*(gemeente|provincie|stad|regio|comune|provincia|citta metropolitana|citt[aà]|"
        r"gemeinde|landkreis|kreis|stadt|bundesland|region)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*\(.*?\)\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def first_candidate(value: object) -> str:
    text = clean_loc(value)
    if not text:
        return ""
    parts = re.split(r"\s*(?:,|/|;| en | & )\s*", text, flags=re.IGNORECASE)
    parts = [part.strip() for part in parts if part.strip()]
    return parts[0] if parts else ""


def build_lookup(gdf: gpd.GeoDataFrame) -> dict[str, pd.Series]:
    lookup: dict[str, pd.Series] = {}
    for _, row in gdf.iterrows():
        for value in [row.get("parent_name"), row.get("parent_id")]:
            key = normalize_key(value)
            if key:
                lookup.setdefault(key, row)
    return lookup


def apply_geometry(df: pd.DataFrame, idx, row: pd.Series, level: str, source: str, match_type: str) -> None:
    geom = row.geometry
    point = centroid_point(geom)
    df.at[idx, "geo_level"] = level
    df.at[idx, "geo_name_matched"] = row.get("parent_name")
    df.at[idx, "geo_source"] = source
    df.at[idx, "geo_match_type"] = match_type
    df.at[idx, "geo_lat"] = float(point.y)
    df.at[idx, "geo_lon"] = float(point.x)
    df.at[idx, "geom_point_wkt"] = point.wkt
    df.at[idx, "geom_poly_wkt"] = geom.wkt
    df.at[idx, "country_id"] = row.get("country_id")
    canonical_country = country_name_for_id(row.get("country_id"))
    if canonical_country:
        df.at[idx, "llm_country"] = canonical_country
        df.at[idx, "llm_country_candidates"] = country_candidates_json([canonical_country])
        df.at[idx, "llm_country_assignment_type"] = "single_country"
        df.at[idx, "llm_has_single_country"] = True
    if level == "nuts2":
        df.at[idx, "nuts2_id"] = row.get("parent_id")
        df.at[idx, "nuts2_name"] = row.get("parent_name")
        df.at[idx, "province_code"] = row.get("parent_id")
        df.at[idx, "province_name"] = row.get("parent_name")


def should_skip_geocoding(row: pd.Series, country_scope: CountryScope) -> bool:
    location = row.get("llm_location")
    granularity = row.get("llm_granularity")
    loc_norm = normalize_key(first_candidate(location))
    if not loc_norm or loc_norm in {"none", "nan", "null"}:
        return True
    if str(row.get("llm_country_assignment_type", "") or "").strip() == "multiple_countries":
        return True
    candidates = country_candidates_for_label(location, allowed_countries=country_scope.countries or None)
    return len(candidates) > 1


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
    country_scope = country_scope_from_args(
        country=args.country,
        countries=args.countries,
        country_scope=args.country_scope,
    )

    shapes = load_shapes_parquet(args.shapes_parquet)
    nuts2 = nuts2_shapes(shapes, country_scope.label)
    countries = country_shapes(shapes, country_scope.label)
    nuts_lookup = build_lookup(nuts2)
    country_lookup = build_lookup(countries)
    country_aliases = country_alias_lookup(country_scope.label)

    df = pd.read_csv(args.input_csv)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    missing = {"llm_location", "llm_granularity"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["_gran"] = df["llm_granularity"].astype(str).str.strip().str.lower()
    df["_loc_first"] = df["llm_location"].map(first_candidate)
    df["_loc_norm"] = df["_loc_first"].map(normalize_key)

    for col in ["geo_level", "geo_name_matched", "geo_source", "geo_match_type"]:
        if col not in df.columns:
            df[col] = None
    for col in ["geo_lat", "geo_lon"]:
        df[col] = pd.NA
    for col in ["geom_point_wkt", "geom_poly_wkt", "country_id", "nuts2_id", "nuts2_name", "province_code", "province_name"]:
        if col not in df.columns:
            df[col] = None
        else:
            df[col] = None

    for idx, loc_norm in df["_loc_norm"].items():
        if should_skip_geocoding(df.loc[idx], country_scope):
            continue
        if not loc_norm:
            continue
        country_id = country_aliases.get(loc_norm)
        if country_id:
            assignment = country_assignment_for_location(df.at[idx, "llm_location"], df.at[idx, "llm_granularity"], country_scope)
            if assignment["llm_country_assignment_type"] != "single_country":
                continue
            country_row = country_lookup.get(normalize_key(country_id))
            if country_row is not None:
                apply_geometry(df, idx, country_row, "country", "shapes_parquet", "country_alias")
            continue
        nuts_row = nuts_lookup.get(loc_norm)
        if nuts_row is not None:
            apply_geometry(df, idx, nuts_row, "nuts2", "shapes_parquet", "nuts2_name_or_id")

    write_outputs(
        df,
        Path(args.output_gpkg),
        Path(args.output_csv),
        args.points_layer,
        args.polygons_layer,
    )

    has_geo = df["geom_point_wkt"].notna() | df["geom_poly_wkt"].notna()
    print(f"[workflow_table] paragraphs_after_shapes_geocoding: {len(df)}")
    print(f"[workflow_table] paragraphs_geocoded_shapes: {int(has_geo.sum())}")
    print(f"[workflow_table] paragraphs_not_geocoded_shapes: {int((~has_geo).sum())}")
    if (~has_geo).any():
        print("\nTop unmatched llm_location values:")
        print(df.loc[~has_geo, "_loc_first"].value_counts().head(20))


if __name__ == "__main__":
    main()
