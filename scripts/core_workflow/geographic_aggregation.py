#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

import geopandas as gpd

from helpers.country_scope import country_name_for_id, country_scope_from_args
from helpers.shape_resources import load_shapes_parquet, regional_shapes, region_level, set_region_fields

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=str, default=str(DEFAULT_PROJECT_DIR))
    ap.add_argument("--input-gpkg", type=str, default="output/workflow/sentences_with_categories.gpkg")
    ap.add_argument("--input-layer", type=str, default="sentences_with_categories")
    ap.add_argument("--shapes-parquet", type=str, default="data/shapes.parquet")
    ap.add_argument("--country", type=str, default="", help="Backward-compatible single-country shorthand.")
    ap.add_argument("--countries", nargs="+", default=None)
    ap.add_argument("--country-scope", type=str, default="")
    ap.add_argument("--output-gpkg", type=str, default="output/workflow/sentences_with_categories_admin.gpkg")
    ap.add_argument("--output-layer", type=str, default="sentences_with_categories_admin")
    ap.add_argument("--output-csv", type=str, default="output/workflow/sentences_with_categories_admin.csv")
    ap.add_argument("--location-province-overrides", type=str, default="")
    return ap.parse_args()


def assign_regions(text_gdf: gpd.GeoDataFrame, regions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    text_gdf = text_gdf.loc[:, ~text_gdf.columns.duplicated()].copy()
    if text_gdf.crs is None:
        raise ValueError("Input geometries have no CRS.")
    if regions.crs is None:
        raise ValueError("Region geometries have no CRS.")
    if regions.crs != text_gdf.crs:
        regions = regions.to_crs(text_gdf.crs)

    country_rows = text_gdf["geo_level"].astype(str).str.lower().eq("country")
    if country_rows.any():
        country_names = text_gdf.loc[country_rows, "llm_country"]
        fallback_names = text_gdf.loc[country_rows, "country_id"].map(country_name_for_id)
        country_names = country_names.where(country_names.notna() & country_names.astype(str).str.strip().ne(""), fallback_names)
        country_names = country_names.where(country_names.notna() & country_names.astype(str).str.strip().ne(""), text_gdf.loc[country_rows, "geo_name_matched"])
        text_gdf.loc[country_rows, "country_name"] = country_names.values
        for col in ["province_name", "province_code", "nuts2_id", "nuts2_name", "nuts3_id", "nuts3_name"]:
            if col in text_gdf.columns:
                text_gdf.loc[country_rows, col] = None
        text_gdf.loc[country_rows, "admin_level"] = "country"

    # A NUTS2 polygon/centroid cannot identify an Italian NUTS3 province.
    coarse_italy = text_gdf["geo_level"].astype(str).str.lower().eq("nuts2") & text_gdf["country_id"].eq("ITA")
    for col in ["province_name", "province_code", "nuts3_id", "nuts3_name"]:
        if col in text_gdf.columns:
            text_gdf.loc[coarse_italy, col] = None
    text_gdf.loc[coarse_italy, "admin_level"] = "nuts2"

    non_country = text_gdf["geo_level"].astype(str).str.lower().ne("country")
    eligible = non_country & ~coarse_italy & text_gdf.geometry.notna()
    if not eligible.any() or regions.empty:
        return text_gdf

    reps = text_gdf.loc[eligible, ["geometry"]].copy()
    reps["geometry"] = reps.geometry.representative_point()
    reps = gpd.GeoDataFrame(reps, geometry="geometry", crs=text_gdf.crs)

    keep = ["country_id", "parent_id", "parent_name", "parent_subtype", "geometry"]
    joined = gpd.sjoin(reps, regions[keep], how="left", predicate="within")
    joined = joined.drop(columns=["index_right"], errors="ignore")
    matched = joined["parent_id"].notna()
    if matched.any():
        idx = joined.loc[matched].index
        text_gdf.loc[idx, "country_id"] = joined.loc[matched, "country_id"].values
        text_gdf.loc[idx, "country_name"] = joined.loc[matched, "country_id"].map(country_name_for_id).values
        for row_idx, row in joined.loc[matched].iterrows():
            set_region_fields(text_gdf, row_idx, row)
            text_gdf.at[row_idx, "admin_level"] = region_level(row)
    return text_gdf


# Backward-compatible entry point for the Dutch evaluation workflow.
assign_nuts2 = assign_regions


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    os.chdir(project_dir)

    text_gdf = gpd.read_file(args.input_gpkg, layer=args.input_layer)
    if text_gdf.crs is None:
        raise ValueError("Input layer has no CRS.")
    text_gdf = text_gdf.to_crs("EPSG:4326")

    shapes = load_shapes_parquet(args.shapes_parquet)
    country_scope = country_scope_from_args(
        country=args.country,
        countries=args.countries,
        country_scope=args.country_scope,
    )
    regions = regional_shapes(shapes, country_scope.label).to_crs(text_gdf.crs)
    text_gdf = assign_regions(text_gdf, regions)

    Path(args.output_gpkg).parent.mkdir(parents=True, exist_ok=True)
    text_gdf.to_file(args.output_gpkg, layer=args.output_layer, driver="GPKG")

    text_csv = text_gdf.to_crs("EPSG:4326").copy()
    geom_type = text_csv.geometry.geom_type.fillna("")
    if (geom_type == "Point").all():
        text_csv["lon"] = text_csv.geometry.x
        text_csv["lat"] = text_csv.geometry.y
    else:
        centroids = text_csv.geometry.representative_point()
        text_csv["lon"] = centroids.x
        text_csv["lat"] = centroids.y

    text_csv = text_csv.drop(columns="geometry")
    text_csv.to_csv(args.output_csv, index=False)
    print("Saved GPKG to:", args.output_gpkg)
    print("Saved CSV to:", args.output_csv)


if __name__ == "__main__":
    main()
