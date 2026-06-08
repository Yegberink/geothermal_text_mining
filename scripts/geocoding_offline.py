import argparse
import os
import re
import unicodedata
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union

from language_resources import country_aliases

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[1]
USE_WOONPLAATS_FALLBACK = True
USE_ALIAS_MAP = True

PROVINCE_CODE_MAP = {
    "GR": "PV20", "FR": "PV21", "DR": "PV22", "OV": "PV23",
    "FL": "PV24", "GE": "PV25", "UT": "PV26", "NH": "PV27",
    "ZH": "PV28", "ZE": "PV29", "NB": "PV30", "LI": "PV31",
}

ALIAS_MAP = {
    "grubbenvorst": ("municipality", "Horst aan de Maas"),
    "maasland": ("municipality", "Midden-Delfland"),
    "pijnacker": ("municipality", "Pijnacker-Nootdorp"),
    "luttelgeest": ("municipality", "Noordoostpolder"),
    "middenmeer": ("municipality", "Hollands Kroon"),
    "wieringermeer": ("municipality", "Hollands Kroon"),
    "twente": ("province", "Overijssel"),
    "ijmond": ("province", "Noord-Holland"),
    "eemland": ("province", "Utrecht"),
    "zuidplaspolder": ("municipality", "Zuidplas"),
}

EXTERNAL_LOCATION_POINTS = {
    "kenia": ("Kenya", -0.0236, 37.9062),
    "kenya": ("Kenya", -0.0236, 37.9062),
    "calefornia": ("California, United States", 36.7783, -119.4179),
    "california": ("California, United States", 36.7783, -119.4179),
    "californie": ("California, United States", 36.7783, -119.4179),
    "curacao": ("Curaçao", 12.1696, -68.9900),
    "curaçao": ("Curaçao", 12.1696, -68.9900),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=str, default=str(DEFAULT_PROJECT_DIR))
    ap.add_argument("--input-csv", type=str, default="output/text/sentence_sentiment_llm.csv")
    ap.add_argument("--municipality-gpkg", type=str, default="data/dutch/admin_areas_municipalities_2025.gpkg")
    ap.add_argument("--province-gpkg", type=str, default="data/dutch/admin_areas_provinces_2025.gpkg")
    ap.add_argument("--output-gpkg", type=str, default="output/text/sentences_with_geo_offline.gpkg")
    ap.add_argument("--output-csv", type=str, default="output/text/sentence_offline_geocoding.csv")
    ap.add_argument("--country", type=str, default="Netherlands")
    ap.add_argument("--points-layer", type=str, default="sentences_points")
    ap.add_argument("--polygons-layer", type=str, default="sentences_polygons")
    return ap.parse_args()


def norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("’", "'")
    s = s.replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\.'()]", "", s)
    return s.strip()


def clean_loc(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(
        r"^\s*(gemeente|provincie|stad|regio|comune|provincia|citta metropolitana|citt[aà]|"
        r"gemeinde|landkreis|kreis|stadt|bundesland|region)\s+",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def first_candidate(s: str) -> str:
    s = clean_loc(s)
    if not s:
        return ""
    parts = re.split(r"\s*(?:,|/|;| en | & )\s*", s, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    return parts[0] if parts else ""


def pick_col_by_regex(cols, patterns):
    cols_l = [c.lower() for c in cols]
    for pat in patterns:
        for c, cl in zip(cols, cols_l):
            if re.search(pat, cl):
                return c
    return None


def pick_admin_col(cols, exact_names, patterns):
    cols_l = [c.lower() for c in cols]
    exact_l = [name.lower() for name in exact_names]
    for target in exact_l:
        for c, cl in zip(cols, cols_l):
            if cl == target:
                return c
    return pick_col_by_regex(cols, patterns)


def pick_municipality_name_col(cols):
    return pick_admin_col(
        cols,
        exact_names=["statnaam", "name", "com_name", "gen"],
        patterns=[r"gemeente.*naam", r"gm_.*naam", r"com.*name", r"\bname\b", r"\bnaam\b"],
    )


def pick_municipality_code_col(cols):
    return pick_admin_col(
        cols,
        exact_names=["statcode", "com_istat_code", "ags", "ars"],
        patterns=[r"gemeente.*code", r"gm_.*code", r"com.*istat.*code", r"com.*code", r"\bcode\b"],
    )


def pick_province_name_col(cols):
    return pick_admin_col(
        cols,
        exact_names=["statnaam", "prov_name", "name", "gen"],
        patterns=[r"provincie.*naam", r"pv_.*naam", r"prov.*name", r"\bname\b", r"\bnaam\b"],
    )


def pick_province_code_col(cols):
    return pick_admin_col(
        cols,
        exact_names=["statcode", "prov_istat_code", "prov_acr", "lkz", "sn_l", "ags", "ars"],
        patterns=[r"pv_.*code", r"provincie.*code", r"prov.*istat.*code", r"prov.*acr", r"\bcode\b"],
    )


def projected_crs_for(gdf: gpd.GeoDataFrame):
    try:
        return gdf.estimate_utm_crs()
    except Exception:
        return "EPSG:3857"


def with_wgs84_centroids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    proj_crs = projected_crs_for(gdf)
    proj = gdf.to_crs(proj_crs)
    proj["_centroid_proj"] = proj.geometry.centroid

    wgs = gdf.to_crs("EPSG:4326")
    wgs["_centroid"] = gpd.GeoSeries(proj["_centroid_proj"], crs=proj.crs).to_crs("EPSG:4326").values
    return wgs


def prov_join_key(gran, loc_first, loc_norm):
    if gran != "province":
        return loc_norm
    abbr = (loc_first or "").strip().upper()
    pv = PROVINCE_CODE_MAP.get(abbr)
    return norm(pv) if pv else loc_norm


def join_admin(df_in, mask, join_key_col, map_gdf, name_col, level_label):
    if not mask.any():
        return df_in

    tmp = df_in.loc[mask, [join_key_col]].copy()
    tmp["__ix"] = tmp.index
    tmp["__k"] = tmp[join_key_col].astype(str)

    map_df = map_gdf[["_norm_name", "_norm_code", name_col, "geometry"]].copy()
    j = tmp.merge(map_df, left_on="__k", right_on="_norm_name", how="left")
    j = j.drop_duplicates(subset="__ix", keep="first")

    need2 = j["geometry"].isna() & j["__k"].ne("") & map_df["_norm_code"].astype(str).ne("").any()
    if need2.any():
        j2 = j.loc[need2, ["__ix", "__k"]].merge(map_df, left_on="__k", right_on="_norm_code", how="left")
        j2 = j2.drop_duplicates(subset="__ix", keep="first")
        j = j.set_index("__ix")
        j2 = j2.set_index("__ix")
        j.loc[j2.index, ["geometry", name_col]] = j2[["geometry", name_col]]
        j = j.reset_index().rename(columns={"index": "__ix"})

    got = ~j["geometry"].isna()
    if got.any():
        idxs = j.loc[got, "__ix"].astype(int).values
        geoms = j.loc[got, "geometry"].values
        geom_series = gpd.GeoSeries(geoms, crs="EPSG:4326")
        proj_crs = projected_crs_for(gpd.GeoDataFrame(geometry=geom_series, crs="EPSG:4326"))
        centroids = geom_series.to_crs(proj_crs).centroid
        centroids = gpd.GeoSeries(centroids, crs=proj_crs).to_crs("EPSG:4326")

        df_in.loc[idxs, "geo_level"] = level_label
        df_in.loc[idxs, "geo_name_matched"] = j.loc[got, name_col].astype(str).values
        df_in.loc[idxs, "geo_source"] = "cbs_gpkg"
        df_in.loc[idxs, "geo_match_type"] = "name_or_code"
        df_in.loc[idxs, "geo_lat"] = [float(p.y) for p in centroids]
        df_in.loc[idxs, "geo_lon"] = [float(p.x) for p in centroids]
        df_in.loc[idxs, "geom_poly_wkt"] = [g.wkt for g in geoms]
        df_in.loc[idxs, "geom_point_wkt"] = [Point(float(p.x), float(p.y)).wkt for p in centroids]

    return df_in


def main():
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    os.chdir(project_dir)
    Path(args.output_gpkg).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    muni_gdf = gpd.read_file(args.municipality_gpkg)
    prov_gdf = gpd.read_file(args.province_gpkg)
    if muni_gdf.crs is None or prov_gdf.crs is None:
        raise ValueError("CBS layers must have CRS set.")

    muni_name_col = pick_municipality_name_col(muni_gdf.columns)
    muni_code_col = pick_municipality_code_col(muni_gdf.columns)
    prov_name_col = pick_province_name_col(prov_gdf.columns)
    prov_code_col = pick_province_code_col(prov_gdf.columns)
    if muni_name_col is None or prov_name_col is None:
        raise ValueError(
            "Could not detect administrative name columns.\n"
            f"Municipality columns: {list(muni_gdf.columns)}\n"
            f"Province columns: {list(prov_gdf.columns)}"
        )

    muni_gdf = muni_gdf.copy()
    prov_gdf = prov_gdf.copy()
    muni_gdf["_norm_name"] = muni_gdf[muni_name_col].astype(str).map(norm)
    prov_gdf["_norm_name"] = prov_gdf[prov_name_col].astype(str).map(norm)
    muni_gdf["_norm_code"] = muni_gdf[muni_code_col].astype(str).map(norm) if muni_code_col else ""
    prov_gdf["_norm_code"] = prov_gdf[prov_code_col].astype(str).map(norm) if prov_code_col else ""

    muni_wgs84 = with_wgs84_centroids(muni_gdf)
    prov_wgs84 = with_wgs84_centroids(prov_gdf)

    country_poly = unary_union(prov_gdf.geometry.values)
    country_poly_wgs84 = gpd.GeoSeries([country_poly], crs=prov_gdf.crs).to_crs("EPSG:4326").iloc[0]
    country_proj_crs = projected_crs_for(prov_gdf)
    country_proj = gpd.GeoSeries([country_poly], crs=prov_gdf.crs).to_crs(country_proj_crs).iloc[0]
    country_centroid_wgs84 = gpd.GeoSeries(
        [country_proj.centroid],
        crs=country_proj_crs,
    ).to_crs("EPSG:4326").iloc[0]
    country_name_aliases = {norm(name) for name in country_aliases(args.country)}

    df = pd.read_csv(args.input_csv)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    missing = {"llm_location", "llm_granularity"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["_gran"] = df["llm_granularity"].astype(str).str.strip().str.lower()
    df["_loc_first"] = df["llm_location"].map(first_candidate)
    df["_loc_norm"] = df["_loc_first"].map(norm)

    if USE_ALIAS_MAP:
        def apply_alias(gran, loc_norm):
            if loc_norm in ALIAS_MAP:
                new_gran, new_val = ALIAS_MAP[loc_norm]
                return new_gran, norm(new_val)
            return gran, loc_norm

        new_vals = [apply_alias(g, l) for g, l in zip(df["_gran"], df["_loc_norm"])]
        df["_gran"] = [g for g, _ in new_vals]
        df["_loc_norm"] = [l for _, l in new_vals]

    df["_join_key"] = [prov_join_key(g, lf, ln) for g, lf, ln in zip(df["_gran"], df["_loc_first"], df["_loc_norm"])]

    for c in ["geo_level", "geo_name_matched", "geo_source", "geo_match_type"]:
        if c not in df.columns:
            df[c] = None
    df["geo_lat"] = pd.NA
    df["geo_lon"] = pd.NA
    df["geom_point_wkt"] = None
    df["geom_poly_wkt"] = None

    external_key = df["_loc_norm"].isin(EXTERNAL_LOCATION_POINTS)
    if external_key.any():
        for idx, loc_norm in df.loc[external_key, "_loc_norm"].items():
            display_name, lat, lon = EXTERNAL_LOCATION_POINTS[loc_norm]
            df.at[idx, "geo_level"] = "external"
            df.at[idx, "geo_name_matched"] = display_name
            df.at[idx, "geo_source"] = "manual_external_alias"
            df.at[idx, "geo_match_type"] = "manual_point"
            df.at[idx, "geo_lat"] = float(lat)
            df.at[idx, "geo_lon"] = float(lon)
            df.at[idx, "geom_point_wkt"] = Point(float(lon), float(lat)).wkt

    df = join_admin(df, df["_gran"].eq("municipality"), "_join_key", muni_wgs84, muni_name_col, "municipality")
    df = join_admin(df, df["_gran"].eq("province"), "_join_key", prov_wgs84, prov_name_col, "province")

    country_key = df["_gran"].eq("country") & df["_loc_norm"].isin(country_name_aliases)
    if country_key.any():
        df.loc[country_key, "geo_level"] = "country"
        df.loc[country_key, "geo_name_matched"] = df.loc[country_key, "_loc_first"].replace("", "country")
        df.loc[country_key, "geo_source"] = "admin_gpkg_union_provinces"
        df.loc[country_key, "geo_match_type"] = "union"
        df.loc[country_key, "geo_lat"] = float(country_centroid_wgs84.y)
        df.loc[country_key, "geo_lon"] = float(country_centroid_wgs84.x)
        df.loc[country_key, "geom_poly_wkt"] = country_poly_wgs84.wkt
        df.loc[country_key, "geom_point_wkt"] = Point(
            float(country_centroid_wgs84.x),
            float(country_centroid_wgs84.y),
        ).wkt

    if USE_WOONPLAATS_FALLBACK:
        try:
            import fiona
            layers = set(fiona.listlayers(args.municipality_gpkg))
        except Exception:
            layers = set()

        woonplaats_layer = next(
            (cand for cand in ["woonplaats_gegeneraliseerd", "woonplaats", "cbs_woonplaats"] if cand in layers),
            None,
        )

        if woonplaats_layer:
            wp = gpd.read_file(args.municipality_gpkg, layer=woonplaats_layer)
            wp_name_col = pick_col_by_regex(wp.columns, [r"statnaam", r"woonplaats.*naam", r"\bnaam\b"])
            if wp_name_col:
                wp = wp.copy()
                wp["_norm_name"] = wp[wp_name_col].astype(str).map(norm)
                wp_wgs84 = with_wgs84_centroids(wp)

                miss_muni = df["_gran"].eq("municipality") & df["geom_poly_wkt"].isna() & df["_join_key"].ne("")
                if miss_muni.any():
                    tmp = df.loc[miss_muni, ["_join_key"]].copy()
                    tmp["__ix"] = tmp.index
                    tmp = tmp.merge(
                        wp_wgs84[["_norm_name", "geometry"]],
                        left_on="_join_key",
                        right_on="_norm_name",
                        how="left",
                    )
                    tmp = tmp.dropna(subset=["geometry"]).drop_duplicates(subset="__ix", keep="first")
                    if not tmp.empty:
                        wp_series = gpd.GeoSeries(tmp["geometry"], crs="EPSG:4326")
                        wp_proj_crs = projected_crs_for(gpd.GeoDataFrame(geometry=wp_series, crs="EPSG:4326"))
                        wp_cent = wp_series.to_crs(wp_proj_crs).centroid
                        wp_cent = gpd.GeoSeries(wp_cent, crs=wp_proj_crs).to_crs("EPSG:4326")
                        pts = gpd.GeoDataFrame(tmp[["__ix"]].copy(), geometry=wp_cent, crs="EPSG:4326")
                        try:
                            joined = gpd.sjoin(pts, muni_wgs84[["geometry"]].copy(), predicate="within", how="left")
                            ok = joined["index_right"].notna()
                            if ok.any():
                                ix = joined.loc[ok, "__ix"].astype(int).values
                                muni_idx = joined.loc[ok, "index_right"].astype(int).values
                                muni_geom = muni_wgs84.loc[muni_idx, "geometry"].values
                                muni_name = muni_wgs84.loc[muni_idx, muni_name_col].astype(str).values
                                muni_cent = muni_wgs84.loc[muni_idx, "_centroid"].values
                                df.loc[ix, "geo_level"] = "municipality"
                                df.loc[ix, "geo_name_matched"] = muni_name
                                df.loc[ix, "geo_source"] = "cbs_woonplaats->sjoin_muni"
                                df.loc[ix, "geo_match_type"] = "woonplaats_name_then_sjoin"
                                df.loc[ix, "geom_poly_wkt"] = [g.wkt for g in muni_geom]
                                df.loc[ix, "geom_point_wkt"] = [Point(float(c.x), float(c.y)).wkt for c in muni_cent]
                                df.loc[ix, "geo_lat"] = [float(c.y) for c in muni_cent]
                                df.loc[ix, "geo_lon"] = [float(c.x) for c in muni_cent]
                        except Exception as e:
                            print("Woonplaats fallback: spatial join failed, skipping:", repr(e))
            else:
                print("Woonplaats fallback: could not find woonplaats name column.")
        else:
            print("Woonplaats fallback: no woonplaats layer found in the GeoPackage.")

    diag = (
        df.assign(_has_geo=df["geom_point_wkt"].notna() | df["geom_poly_wkt"].notna())
        .groupby(["_gran", "_has_geo"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["_gran", "_has_geo"])
    )
    print("\nCoverage by granularity:")
    print(diag)

    for g in ["municipality", "province"]:
        miss = df["_gran"].eq(g) & df["geom_poly_wkt"].isna()
        if miss.any():
            print(f"\nTop unmatched {g} llm_location values:")
            print(df.loc[miss, "_loc_first"].value_counts().head(20))

    if os.path.exists(args.output_gpkg):
        os.remove(args.output_gpkg)

    df_points = df[df["geom_point_wkt"].notna()].copy()
    gdf_points = gpd.GeoDataFrame(
        df_points,
        geometry=gpd.GeoSeries.from_wkt(df_points["geom_point_wkt"]),
        crs="EPSG:4326",
    )
    gdf_points.to_file(args.output_gpkg, layer=args.points_layer, driver="GPKG")

    df_polys = df[df["geom_poly_wkt"].notna()].copy()
    gdf_polys = gpd.GeoDataFrame(
        df_polys,
        geometry=gpd.GeoSeries.from_wkt(df_polys["geom_poly_wkt"]),
        crs="EPSG:4326",
    )
    gdf_polys.to_file(args.output_gpkg, layer=args.polygons_layer, driver="GPKG")
    df.to_csv(args.output_csv, index=False, encoding="utf-8")

    print("\nWrote GeoPackage:", args.output_gpkg)
    print(f"Layers: {args.points_layer}, {args.polygons_layer}")
    has_geo = df["geom_point_wkt"].notna() | df["geom_poly_wkt"].notna()
    print(f"[workflow_table] paragraphs_after_offline_geocoding: {len(df)}")
    print(f"[workflow_table] paragraphs_geocoded_offline: {int(has_geo.sum())}")
    print(f"[workflow_table] paragraphs_not_geocoded_offline: {int((~has_geo).sum())}")


if __name__ == "__main__":
    main()
