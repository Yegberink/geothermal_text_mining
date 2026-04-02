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
    ap.add_argument("--input-gpkg", type=str, default="output/text/sentences_with_absa_and_geo_v2.gpkg")
    ap.add_argument("--keywords-csv", type=str, default="vocab/keywords_topics.csv")
    ap.add_argument("--output-gpkg", type=str, default="output/text/sentences_with_categories.gpkg")
    ap.add_argument("--output-layer", type=str, default="sentences_with_categories")
    ap.add_argument("--output-long-csv", type=str, default="output/text/sentences_with_categories_long.csv")
    ap.add_argument("--output-short-csv", type=str, default="output/text/sentences_with_categories_short.csv")
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

    points_gdf = gpd.read_file(args.input_gpkg, layer="sentences_points")
    polys_gdf = gpd.read_file(args.input_gpkg, layer="sentences_polygons")
    points_gdf["source_layer"] = "sentences_points"
    polys_gdf["source_layer"] = "sentences_polygons"

    sentences_gdf = pd.concat([points_gdf, polys_gdf], ignore_index=True)
    sentences_gdf = gpd.GeoDataFrame(sentences_gdf, geometry="geometry", crs=points_gdf.crs)

    keyword_categories = pd.read_csv(args.keywords_csv)
    keyword_categories = keyword_categories.loc[
        :, ~keyword_categories.columns.astype(str).str.match(r"^Unnamed")
    ]

    all_keywords = keyword_categories.stack().dropna().str.strip()
    duplicates = all_keywords[all_keywords.duplicated()]
    print(duplicates)

    text = sentences_gdf["sentence_text"].astype(str).str.lower()

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
    sentences_gdf["matched_categories"] = res.apply(lambda x: x[0])
    sentences_gdf["matched_keywords"] = res.apply(lambda x: x[1])
    sentences_gdf["n_categories"] = sentences_gdf["matched_categories"].str.len()
    sentences_gdf["matched_categories_str"] = sentences_gdf["matched_categories"].apply(
        lambda lst: ";".join(lst) if lst else None
    )
    sentences_gdf["matched_keywords_str"] = sentences_gdf["matched_keywords"].apply(
        hits_to_keywords_only_str
    )
    sentences_gdf = sentences_gdf.drop(columns=["matched_categories", "matched_keywords"])

    sentences_gdf["sentence_text_norm"] = (
        sentences_gdf["sentence_text"].astype(str).str.strip().str.lower()
    )
    majority_sentiment = (
        sentences_gdf.groupby("sentence_text_norm")["sentiment"]
        .agg(lambda x: x.value_counts().idxmax())
    )
    sentences_gdf["geom_priority"] = sentences_gdf["source_layer"].map({
        "sentences_polygons": 1,
        "sentences_points": 2,
    })

    sentences_unique = (
        sentences_gdf.sort_values(["sentence_text_norm", "geom_priority"])
        .drop_duplicates(subset=["sentence_text_norm"], keep="first")
        .copy()
    )
    sentences_unique["sentiment"] = sentences_unique["sentence_text_norm"].map(majority_sentiment)
    sentences_unique = sentences_unique.drop(columns=["sentence_text_norm", "geom_priority"])
    sentences_unique = gpd.GeoDataFrame(
        sentences_unique,
        geometry="geometry",
        crs=sentences_gdf.crs,
    )

    sentences_unique.to_file(output_gpkg, layer=args.output_layer, driver="GPKG")

    sentences_gdf_small = sentences_unique[
        ["sentence_text", "matched_keywords_str", "matched_categories_str", "sentiment", "source_layer"]
    ]
    sentences_unique.to_csv(output_long_csv, index=False)
    sentences_gdf_small.to_csv(output_short_csv, index=False)

    print(len(sentences_unique), "unique sentences processed and saved.")


if __name__ == "__main__":
    main()
