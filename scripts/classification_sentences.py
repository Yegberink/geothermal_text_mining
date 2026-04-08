import argparse
import os
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=str, default=str(DEFAULT_PROJECT_DIR))
    ap.add_argument("--input-gpkg", type=str, default="output/text/paragraphs_with_geo.gpkg")
    ap.add_argument("--keywords-csv", type=str, default="vocab/keywords_topics.csv")
    ap.add_argument("--output-gpkg", type=str, default="output/text/paragraphs_with_categories.gpkg")
    ap.add_argument("--output-layer", type=str, default="paragraphs_with_categories")
    ap.add_argument("--output-long-csv", type=str, default="output/text/paragraphs_with_categories_long.csv")
    ap.add_argument("--output-short-csv", type=str, default="output/text/paragraphs_with_categories_short.csv")
    return ap.parse_args()


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


def main():
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    os.chdir(project_dir)

    output_gpkg = Path(args.output_gpkg)
    output_long_csv = Path(args.output_long_csv)
    output_short_csv = Path(args.output_short_csv)
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    output_long_csv.parent.mkdir(parents=True, exist_ok=True)
    output_short_csv.parent.mkdir(parents=True, exist_ok=True)

    points_gdf = gpd.read_file(args.input_gpkg, layer="paragraphs_points")
    polys_gdf = gpd.read_file(args.input_gpkg, layer="paragraphs_polygons")
    points_gdf["source_layer"] = "paragraphs_points"
    polys_gdf["source_layer"] = "paragraphs_polygons"

    text_gdf = pd.concat([points_gdf, polys_gdf], ignore_index=True)
    text_gdf = gpd.GeoDataFrame(text_gdf, geometry="geometry", crs=points_gdf.crs)

    keyword_categories = pd.read_csv(args.keywords_csv)
    keyword_categories = keyword_categories.loc[
        :, ~keyword_categories.columns.astype(str).str.match(r"^Unnamed")
    ]

    all_keywords = keyword_categories.stack().dropna().str.strip()
    duplicates = all_keywords[all_keywords.duplicated()]
    print(duplicates)

    text_col = "paragraph_text" if "paragraph_text" in text_gdf.columns else "sentence_text"
    uid_col = "uid" if "uid" in text_gdf.columns else "sentence_uid"
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
    text_gdf["matched_categories_str"] = text_gdf["matched_categories"].apply(
        lambda lst: ";".join(lst) if lst else None
    )
    text_gdf["matched_keywords_str"] = text_gdf["matched_keywords"].apply(
        hits_to_keywords_only_str
    )
    text_gdf = text_gdf.drop(columns=["matched_categories", "matched_keywords"])

    text_gdf["_text_norm"] = text_gdf[text_col].astype(str).str.strip().str.lower()
    text_gdf["geom_priority"] = text_gdf["source_layer"].map({
        "paragraphs_polygons": 1,
        "paragraphs_points": 2,
    })

    dedupe_col = uid_col if uid_col in text_gdf.columns else "_text_norm"
    text_unique = (
        text_gdf.sort_values([dedupe_col, "geom_priority"])
        .drop_duplicates(subset=[dedupe_col], keep="first")
        .copy()
    )
    text_unique = text_unique.drop(columns=["_text_norm", "geom_priority"])
    text_unique = gpd.GeoDataFrame(
        text_unique,
        geometry="geometry",
        crs=text_gdf.crs,
    )

    text_unique.to_file(output_gpkg, layer=args.output_layer, driver="GPKG")

    short_cols = [text_col, "matched_keywords_str", "matched_categories_str", "sentiment", "source_layer"]
    short_cols = [c for c in short_cols if c in text_unique.columns]
    text_small = text_unique[short_cols].copy()
    text_unique.to_csv(output_long_csv, index=False)
    text_small.to_csv(output_short_csv, index=False)

    print(len(text_unique), "unique text units processed and saved.")


if __name__ == "__main__":
    main()
