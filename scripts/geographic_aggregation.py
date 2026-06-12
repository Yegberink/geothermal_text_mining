#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd

from shape_resources import load_shapes_parquet, nuts2_shapes

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=str, default=str(DEFAULT_PROJECT_DIR))
    ap.add_argument("--input-gpkg", type=str, default="output/text/sentences_with_categories.gpkg")
    ap.add_argument("--input-layer", type=str, default="sentences_with_categories")
    ap.add_argument("--shapes-parquet", type=str, default="data/shapes.parquet")
    ap.add_argument("--country", type=str, default="")
    ap.add_argument("--output-gpkg", type=str, default="output/text/sentences_with_categories_admin.gpkg")
    ap.add_argument("--output-layer", type=str, default="sentences_with_categories_admin")
    ap.add_argument("--output-csv", type=str, default="output/text/sentences_with_categories_admin.csv")
    ap.add_argument("--location-province-overrides", type=str, default="")
    return ap.parse_args()


def assign_nuts2(text_gdf: gpd.GeoDataFrame, nuts2: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    text_gdf = text_gdf.loc[:, ~text_gdf.columns.duplicated()].copy()
    if text_gdf.crs is None:
        raise ValueError("Input geometries have no CRS.")
    if nuts2.crs is None:
        raise ValueError("NUTS2 geometries have no CRS.")
    if nuts2.crs != text_gdf.crs:
        nuts2 = nuts2.to_crs(text_gdf.crs)
    for col in ["nuts2_id", "nuts2_name", "province_code", "province_name", "country_id"]:
        if col not in text_gdf.columns:
            text_gdf[col] = None

    non_country = (
        text_gdf["geo_level"].astype(str).str.lower().ne("country")
        if "geo_level" in text_gdf.columns
        else pd.Series(True, index=text_gdf.index)
    )
    eligible = non_country & text_gdf.geometry.notna()
    if not eligible.any() or nuts2.empty:
        return text_gdf

    reps = text_gdf.loc[eligible, ["geometry"]].copy()
    reps["geometry"] = reps.geometry.representative_point()
    reps = gpd.GeoDataFrame(reps, geometry="geometry", crs=text_gdf.crs)

    keep = ["country_id", "parent_id", "parent_name", "geometry"]
    joined = gpd.sjoin(reps, nuts2[keep], how="left", predicate="within")
    joined = joined.drop(columns=["index_right"], errors="ignore")
    matched = joined["parent_id"].notna()
    if matched.any():
        idx = joined.loc[matched].index
        text_gdf.loc[idx, "country_id"] = joined.loc[matched, "country_id"].values
        text_gdf.loc[idx, "nuts2_id"] = joined.loc[matched, "parent_id"].values
        text_gdf.loc[idx, "nuts2_name"] = joined.loc[matched, "parent_name"].values
        text_gdf.loc[idx, "province_code"] = joined.loc[matched, "parent_id"].values
        text_gdf.loc[idx, "province_name"] = joined.loc[matched, "parent_name"].values
    return text_gdf


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    os.chdir(project_dir)

    text_gdf = gpd.read_file(args.input_gpkg, layer=args.input_layer)
    if text_gdf.crs is None:
        raise ValueError("Input layer has no CRS.")
    text_gdf = text_gdf.to_crs("EPSG:4326")

    shapes = load_shapes_parquet(args.shapes_parquet)
    nuts2 = nuts2_shapes(shapes, args.country).to_crs(text_gdf.crs)
    text_gdf = assign_nuts2(text_gdf, nuts2)

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
