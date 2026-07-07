import argparse
import os
import re
from pathlib import Path

import pandas as pd
from shapely import wkt
from helpers.language_resources import load_keyword_csv

try:
    import geopandas as gpd
except ImportError:
    gpd = None

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[2]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=str, default=str(DEFAULT_PROJECT_DIR))
    ap.add_argument("--input-csv", type=str, default="")
    ap.add_argument("--input-gpkg", type=str, default="")
    ap.add_argument("--input-point-layer", type=str, default="sentences_points")
    ap.add_argument("--input-polygon-layer", type=str, default="sentences_polygons")
    ap.add_argument("--keywords-csv", type=str, default="data/vocab/keywords_topics.csv")
    ap.add_argument("--output-gpkg", type=str, default="")
    ap.add_argument("--output-layer", type=str, default="sentences_with_categories")
    ap.add_argument("--output-long-csv", type=str, default="output/workflow/sentences_with_categories_long.csv")
    ap.add_argument("--output-short-csv", type=str, default="output/workflow/sentences_with_categories_short.csv")
    ap.add_argument("--keep-only-matched", type=str, default="False")
    return ap.parse_args()


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def hits_to_keywords_only_str(d):
    if not d:
        return None
    seen = set()
    flat = []
    for kws in d.values():
        for kw in kws:
            if kw not in seen:
                seen.add(kw)
                flat.append(kw)
    return ";".join(flat) if flat else None


def _valid_wkt_value(value: object) -> bool:
    text = str(value or "").strip()
    return bool(text) and text.lower() not in {"nan", "none", "null", "<na>"}


def attach_geometry_from_wkt(df: pd.DataFrame) -> tuple[pd.DataFrame, object, bool]:
    if gpd is None:
        raise ImportError("geopandas is required to write --output-gpkg from WKT columns.")
    if "geom_poly_wkt" not in df.columns and "geom_point_wkt" not in df.columns:
        return df, None, False

    out = df.copy()
    geometry_values = []
    source_layers = []
    for _, row in out.iterrows():
        poly_wkt = row.get("geom_poly_wkt") if "geom_poly_wkt" in out.columns else None
        point_wkt = row.get("geom_point_wkt") if "geom_point_wkt" in out.columns else None
        if _valid_wkt_value(poly_wkt):
            geometry_values.append(wkt.loads(str(poly_wkt)))
            source_layers.append("polygons_wkt")
        elif _valid_wkt_value(point_wkt):
            geometry_values.append(wkt.loads(str(point_wkt)))
            source_layers.append("points_wkt")
        else:
            geometry_values.append(None)
            source_layers.append(None)

    out["source_layer"] = out.get("source_layer", pd.Series([None] * len(out), index=out.index))
    out["source_layer"] = out["source_layer"].where(out["source_layer"].notna(), source_layers)
    has_geometry = any(geom is not None for geom in geometry_values)
    gdf = gpd.GeoDataFrame(out, geometry=geometry_values, crs="EPSG:4326")
    return gdf, gdf.crs, has_geometry


def main():
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    os.chdir(project_dir)

    output_long_csv = Path(args.output_long_csv)
    output_short_csv = Path(args.output_short_csv)
    output_long_csv.parent.mkdir(parents=True, exist_ok=True)
    output_short_csv.parent.mkdir(parents=True, exist_ok=True)
    keep_only_matched = parse_bool(args.keep_only_matched)

    if args.input_csv:
        text_gdf = pd.read_csv(args.input_csv)
        if args.output_gpkg:
            text_gdf, crs, has_geometry = attach_geometry_from_wkt(text_gdf)
        else:
            crs = None
            has_geometry = False
    elif args.input_gpkg:
        if gpd is None:
            raise ImportError("geopandas is required when using --input-gpkg.")
        points_gdf = gpd.read_file(args.input_gpkg, layer=args.input_point_layer)
        polys_gdf = gpd.read_file(args.input_gpkg, layer=args.input_polygon_layer)
        if points_gdf.crs is None:
            raise ValueError(f"Input point layer '{args.input_point_layer}' has no CRS.")
        if polys_gdf.crs is None:
            raise ValueError(f"Input polygon layer '{args.input_polygon_layer}' has no CRS.")
        if points_gdf.crs != polys_gdf.crs:
            raise ValueError(
                f"Input layer CRS mismatch: {args.input_point_layer}={points_gdf.crs}, "
                f"{args.input_polygon_layer}={polys_gdf.crs}."
            )
        points_gdf["source_layer"] = args.input_point_layer
        polys_gdf["source_layer"] = args.input_polygon_layer
        text_gdf = pd.concat([points_gdf, polys_gdf], ignore_index=True)
        text_gdf = gpd.GeoDataFrame(text_gdf, geometry="geometry", crs=points_gdf.crs)
        crs = points_gdf.crs
        has_geometry = True
    else:
        raise ValueError("Provide either --input-csv or --input-gpkg.")

    input_rows = len(text_gdf)
    workflow_stage = "sentence_categories" if args.output_gpkg else "sentence_frames"

    keyword_categories = load_keyword_csv(Path(args.keywords_csv))

    all_keywords = keyword_categories.stack().dropna().str.strip()
    duplicates = all_keywords[all_keywords.duplicated()]
    print(duplicates)

    text_col = "sentence_text" if "sentence_text" in text_gdf.columns else "paragraph_text"
    uid_col = "sentence_uid" if "sentence_uid" in text_gdf.columns else "uid"
    text = text_gdf[text_col].astype(str).str.lower()

    cat2keywords = {}
    for col in keyword_categories.columns:
        category = str(col).strip()
        if not category:
            continue
        keywords = (
            keyword_categories[col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .tolist()
        )
        keywords = [k for k in keywords if k]
        if keywords:
            cat2keywords[category] = keywords

    cat2pattern = {
        cat: re.compile(r"\b(" + "|".join(re.escape(k) for k in kws) + r")\b")
        for cat, kws in cat2keywords.items()
    }

    def get_category_hits(s: str):
        categories = []
        hits = {}
        for cat, pat in cat2pattern.items():
            found = pat.findall(s)
            if found:
                uniq = []
                seen = set()
                for w in found:
                    if w not in seen:
                        seen.add(w)
                        uniq.append(w)
                categories.append(cat)
                hits[cat] = uniq
        return categories, hits

    res = text.apply(get_category_hits)
    text_gdf["matched_categories"] = res.apply(lambda x: x[0])
    text_gdf["matched_keywords"] = res.apply(lambda x: x[1])
    text_gdf["n_categories"] = text_gdf["matched_categories"].str.len()
    matched_rows = int(text_gdf["n_categories"].fillna(0).astype(int).gt(0).sum())
    text_gdf["matched_categories_str"] = text_gdf["matched_categories"].apply(
        lambda lst: ";".join(lst) if lst else None
    )
    text_gdf["matched_keywords_str"] = text_gdf["matched_keywords"].apply(
        hits_to_keywords_only_str
    )
    text_gdf = text_gdf.drop(columns=["matched_categories", "matched_keywords"])

    text_gdf["_text_norm"] = text_gdf[text_col].astype(str).str.strip().str.lower()
    if "source_layer" in text_gdf.columns:
        text_gdf["geom_priority"] = text_gdf["source_layer"].map({
            args.input_polygon_layer: 1,
            args.input_point_layer: 2,
        }).fillna(99)
    else:
        text_gdf["geom_priority"] = 1

    dedupe_col = uid_col if uid_col in text_gdf.columns else "_text_norm"
    text_out = (
        text_gdf.sort_values([dedupe_col, "geom_priority"])
        .drop_duplicates(subset=[dedupe_col], keep="first")
        .copy()
    )
    text_out = text_out.drop(columns=["_text_norm", "geom_priority"])
    if keep_only_matched:
        text_out = text_out[text_out["n_categories"].fillna(0).astype(int) > 0].copy()

    if has_geometry and args.output_gpkg:
        output_gpkg = Path(args.output_gpkg)
        output_gpkg.parent.mkdir(parents=True, exist_ok=True)
        text_out = gpd.GeoDataFrame(
            text_out,
            geometry="geometry",
            crs=crs,
        )
        text_out.to_file(output_gpkg, layer=args.output_layer, driver="GPKG")

    short_cols = [text_col, "matched_keywords_str", "matched_categories_str", "sentiment", "source_layer", "sentence_uid"]
    short_cols = [c for c in short_cols if c in text_out.columns]
    text_small = text_out[short_cols].copy()
    text_out.to_csv(output_long_csv, index=False)
    text_small.to_csv(output_short_csv, index=False)

    print(len(text_out), "unique text units processed and saved.")
    print(f"[workflow_table] {workflow_stage}_input_rows: {input_rows}")
    print(f"[workflow_table] {workflow_stage}_matched_rows: {matched_rows}")
    print(f"[workflow_table] {workflow_stage}_output_rows: {len(text_out)}")


if __name__ == "__main__":
    main()
