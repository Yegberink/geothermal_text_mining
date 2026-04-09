import argparse
import os
from pathlib import Path

import geopandas as gpd

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[1]


def pick_col_by_regex(cols, patterns):
    import re

    cols_l = [c.lower() for c in cols]
    for pat in patterns:
        for c, cl in zip(cols, cols_l):
            if re.search(pat, cl):
                return c
    return None


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=str, default=str(DEFAULT_PROJECT_DIR))
    ap.add_argument("--input-gpkg", type=str, default="output/text/paragraphs_with_categories.gpkg")
    ap.add_argument("--input-layer", type=str, default="paragraphs_with_categories")
    ap.add_argument("--municipality-gpkg", type=str, default="data/dutch/admin_areas_municipalities_2025.gpkg")
    ap.add_argument("--province-gpkg", type=str, default="data/dutch/admin_areas_provinces_2025.gpkg")
    ap.add_argument("--output-gpkg", type=str, default="output/text/paragraphs_with_categories_admin.gpkg")
    ap.add_argument("--output-layer", type=str, default="paragraphs_with_categories_admin")
    ap.add_argument("--output-csv", type=str, default="output/text/paragraphs_with_categories_admin.csv")
    return ap.parse_args()


def add_admin_by_representative_point(
    text_gdf: gpd.GeoDataFrame,
    admin_gdf: gpd.GeoDataFrame,
    name_col: str,
    out_name_col: str,
    code_col: str | None = None,
    out_code_col: str | None = None,
    eligible_mask=None,
):
    text_gdf = text_gdf.loc[:, ~text_gdf.columns.duplicated()].copy()

    if eligible_mask is None:
        eligible_mask = text_gdf[out_name_col].isna()
    else:
        eligible_mask = eligible_mask & text_gdf[out_name_col].isna()

    if not eligible_mask.any():
        return text_gdf

    rep = text_gdf.loc[eligible_mask, ["geometry"]].copy()
    rep["geometry"] = rep.geometry.representative_point()
    rep = gpd.GeoDataFrame(rep, geometry="geometry", crs=text_gdf.crs)

    keep_cols = [name_col, "geometry"]
    if code_col is not None:
        keep_cols.insert(0, code_col)

    joined = gpd.sjoin(rep, admin_gdf[keep_cols], how="left", predicate="within")
    joined = joined.loc[:, ~joined.columns.duplicated()]
    text_gdf.loc[joined.index, out_name_col] = joined[name_col]

    if code_col is not None and out_code_col is not None:
        text_gdf.loc[joined.index, out_code_col] = joined[code_col]

    return text_gdf


def main():
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    os.chdir(project_dir)

    text_gdf = gpd.read_file(args.input_gpkg, layer=args.input_layer)
    muni = gpd.read_file(args.municipality_gpkg)
    prov = gpd.read_file(args.province_gpkg)
    if text_gdf.crs is None:
        raise ValueError("Input layer has no CRS.")

    prov_name_col = pick_col_by_regex(prov.columns, [r"statnaam", r"provincie.*naam", r"prov.*name", r"\bnaam\b", r"name"])
    muni_name_col = pick_col_by_regex(muni.columns, [r"statnaam", r"gemeente.*naam", r"com.*name", r"\bnaam\b", r"name"])
    prov_code_col = pick_col_by_regex(prov.columns, [r"statcode", r"pv_.*code", r"provincie.*code", r"prov.*istat.*code", r"prov.*acr", r"\bcode\b"])
    muni_code_col = pick_col_by_regex(muni.columns, [r"statcode", r"gm_.*code", r"gemeente.*code", r"com.*istat.*code", r"com.*code", r"\bcode\b"])

    if prov_name_col is None:
        raise ValueError("Could not detect province name column.")
    if muni_name_col is None:
        raise ValueError("Could not detect municipality name column.")

    if text_gdf.crs != prov.crs:
        prov = prov.to_crs(text_gdf.crs)
    if text_gdf.crs != muni.crs:
        muni = muni.to_crs(text_gdf.crs)

    prov_keep = [prov_name_col, "geometry"]
    if prov_code_col is not None:
        prov_keep.insert(0, prov_code_col)
    text_gdf = gpd.sjoin(text_gdf, prov[prov_keep], how="left", predicate="within")
    rename_dict = {prov_name_col: "province_name"}
    if prov_code_col is not None:
        rename_dict[prov_code_col] = "province_code"
    text_gdf = text_gdf.rename(columns=rename_dict).drop(columns=["index_right"], errors="ignore")
    non_country = (
        text_gdf["geo_level"].astype(str).str.lower().ne("country")
        if "geo_level" in text_gdf.columns
        else None
    )
    text_gdf = add_admin_by_representative_point(
        text_gdf=text_gdf,
        admin_gdf=prov,
        name_col=prov_name_col,
        out_name_col="province_name",
        code_col=prov_code_col,
        out_code_col="province_code",
        eligible_mask=non_country,
    )
    print("Added province columns")

    muni_keep = [muni_name_col, "geometry"]
    if muni_code_col is not None:
        muni_keep.insert(0, muni_code_col)
    text_gdf = gpd.sjoin(text_gdf, muni[muni_keep], how="left", predicate="within")
    rename_dict = {muni_name_col: "municipality_name"}
    if muni_code_col is not None:
        rename_dict[muni_code_col] = "municipality_code"
    text_gdf = text_gdf.rename(columns=rename_dict).drop(columns=["index_right"], errors="ignore")
    eligible_for_muni = (
        text_gdf["geo_level"].astype(str).str.lower().isin(["municipality", "city", "site"])
        if "geo_level" in text_gdf.columns
        else None
    )
    text_gdf = add_admin_by_representative_point(
        text_gdf=text_gdf,
        admin_gdf=muni,
        name_col=muni_name_col,
        out_name_col="municipality_name",
        code_col=muni_code_col,
        out_code_col="municipality_code",
        eligible_mask=eligible_for_muni,
    )
    text_gdf = text_gdf.loc[:, ~text_gdf.columns.duplicated()]
    print("Added municipality columns")

    Path(args.output_gpkg).parent.mkdir(parents=True, exist_ok=True)
    text_gdf.to_file(args.output_gpkg, layer=args.output_layer, driver="GPKG")

    text_csv = text_gdf.to_crs(4326).copy()
    geom_type = text_csv.geometry.geom_type.fillna("")
    if (geom_type == "Point").all():
        text_csv["lon"] = text_csv.geometry.x
        text_csv["lat"] = text_csv.geometry.y
    else:
        centroids = text_csv.geometry.centroid
        text_csv["lon"] = centroids.x
        text_csv["lat"] = centroids.y

    text_csv = text_csv.drop(columns="geometry")
    text_csv.to_csv(args.output_csv, index=False)
    print("Saved GPKG to:", args.output_gpkg)
    print("Saved CSV to:", args.output_csv)


if __name__ == "__main__":
    main()
