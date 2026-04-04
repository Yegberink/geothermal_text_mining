#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, Optional

import geopandas as gpd
import pandas as pd
from geopy.exc import GeocoderServiceError, GeocoderTimedOut, GeocoderUnavailable
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from shapely.geometry import Point
from shapely.ops import unary_union

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=str, default=str(DEFAULT_PROJECT_DIR))
    ap.add_argument("--input-csv", type=str, default="output/text/sentence_offline_geocoding.csv")
    ap.add_argument("--cbs-gpkg", type=str, default="data/cbsgebiedsindelingen2025.gpkg")
    ap.add_argument("--layer-muni", type=str, default="gemeente_gegeneraliseerd")
    ap.add_argument("--layer-prov", type=str, default="provincie_gegeneraliseerd")
    ap.add_argument("--output-gpkg", type=str, default="output/text/sentences_with_absa_and_geo_v2.gpkg")
    ap.add_argument("--cache-path", type=str, default="cache/nominatim_cache.json")
    ap.add_argument("--country", type=str, default="Netherlands")
    ap.add_argument("--country-codes", type=str, default="")
    ap.add_argument("--user-agent", type=str, default="absa-geo-mapper")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--print-every", type=int, default=25)
    ap.add_argument("--min-delay-seconds", type=float, default=1.1)
    ap.add_argument("--timeout-seconds", type=int, default=10)
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--backoff-base", type=float, default=1.6)
    ap.add_argument("--jitter", type=float, default=0.25)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--attach-admin-polygons-for-city-site", action="store_true")
    return ap.parse_args()


def pick_col_by_regex(cols, patterns):
    cols_l = [c.lower() for c in cols]
    for pat in patterns:
        for c, cl in zip(cols, cols_l):
            if re.search(pat, cl):
                return c
    return None


def with_wgs84_centroids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    try:
        proj = gdf.to_crs("EPSG:28992")
    except Exception:
        proj = gdf.to_crs("EPSG:3857")
    proj["_centroid_proj"] = proj.geometry.centroid

    wgs = gdf.to_crs("EPSG:4326")
    wgs["_centroid"] = gpd.GeoSeries(proj["_centroid_proj"], crs=proj.crs).to_crs("EPSG:4326").values
    return wgs


def load_cache(cache_path: Path) -> Dict[str, Optional[dict]]:
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache_path: Path, cache: Dict[str, Optional[dict]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    os.replace(tmp, cache_path)


def robust_geocode(
    query: str,
    cache: Dict[str, Optional[dict]],
    geocode_rl: RateLimiter,
    country_codes: str,
    max_retries: int,
    backoff_base: float,
    jitter: float,
) -> Optional[dict]:
    query = str(query).strip()
    if not query:
        return None

    if query in cache:
        return cache[query]

    for attempt in range(max_retries):
        try:
            loc = geocode_rl(
                query,
                exactly_one=True,
                addressdetails=False,
                country_codes=country_codes or None,
            )
            if not loc and country_codes:
                loc = geocode_rl(query, exactly_one=True, addressdetails=False)

            if not loc:
                cache[query] = None
                return None

            res = {"lat": float(loc.latitude), "lon": float(loc.longitude), "disp": str(loc)}
            cache[query] = res
            return res
        except (GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError):
            time.sleep((backoff_base ** attempt) + random.random() * jitter)
        except Exception:
            cache[query] = None
            return None

    cache[query] = None
    return None


def bias_query(q: str, country: str) -> str:
    q = str(q).strip()
    if not q:
        return q
    if country and re.search(rf"\b{re.escape(country)}\b", q, flags=re.IGNORECASE):
        return q
    return f"{q}, {country}" if country else q


def build_output_gpkg(df: pd.DataFrame, output_gpkg: Path) -> None:
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    if output_gpkg.exists():
        output_gpkg.unlink()

    df_points = df[df["geom_point_wkt"].notna()].copy()
    gdf_points = gpd.GeoDataFrame(
        df_points,
        geometry=gpd.GeoSeries.from_wkt(df_points["geom_point_wkt"]),
        crs="EPSG:4326",
    )
    gdf_points.to_file(output_gpkg, layer="sentences_points", driver="GPKG")

    df_polys = df[df["geom_poly_wkt"].notna()].copy()
    gdf_polys = gpd.GeoDataFrame(
        df_polys,
        geometry=gpd.GeoSeries.from_wkt(df_polys["geom_poly_wkt"]),
        crs="EPSG:4326",
    )
    gdf_polys.to_file(output_gpkg, layer="sentences_polygons", driver="GPKG")


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    os.chdir(project_dir)

    input_csv = Path(args.input_csv)
    cbs_gpkg = Path(args.cbs_gpkg)
    output_gpkg = Path(args.output_gpkg)
    cache_path = Path(args.cache_path)

    muni_gdf = gpd.read_file(cbs_gpkg, layer=args.layer_muni)
    prov_gdf = gpd.read_file(cbs_gpkg, layer=args.layer_prov)
    muni_wgs84 = with_wgs84_centroids(muni_gdf)
    prov_wgs84 = with_wgs84_centroids(prov_gdf)

    muni_name_col = pick_col_by_regex(
        muni_gdf.columns, [r"statnaam", r"gemeente.*naam", r"gm_.*naam", r"\bnaam\b"]
    )
    prov_name_col = pick_col_by_regex(
        prov_gdf.columns, [r"statnaam", r"provincie.*naam", r"pv_.*naam", r"\bnaam\b"]
    )
    if muni_name_col is None or prov_name_col is None:
        raise ValueError("Could not detect municipality/province name columns in CBS GeoPackage.")

    country_poly = unary_union(prov_gdf.geometry.values)
    country_poly_wgs84 = gpd.GeoSeries([country_poly], crs=prov_gdf.crs).to_crs("EPSG:4326").iloc[0]

    df = pd.read_csv(input_csv, dtype=str)
    df["geo_lat"] = pd.to_numeric(df["geo_lat"], errors="coerce").astype("Float64")
    df["geo_lon"] = pd.to_numeric(df["geo_lon"], errors="coerce").astype("Float64")

    cache = load_cache(cache_path)
    geolocator = Nominatim(user_agent=args.user_agent, timeout=args.timeout_seconds)
    geocode_rl = RateLimiter(
        geolocator.geocode,
        min_delay_seconds=args.min_delay_seconds,
        swallow_exceptions=True,
    )

    def inside_country(lon, lat):
        try:
            return bool(country_poly_wgs84.contains(Point(float(lon), float(lat))))
        except Exception:
            return False

    want_online = (
        df["_gran"].isin(["city", "site"])
        | ((df["_gran"].isin(["municipality", "province"])) & df["geom_poly_wkt"].isna())
    )
    need_point = want_online & df["geom_point_wkt"].isna() & df["_loc_first"].astype(str).str.strip().ne("")

    uniq = df.loc[need_point, "_loc_first"].astype(str).str.strip().unique().tolist()
    if args.max_queries > 0:
        uniq = uniq[: args.max_queries]

    print("Rows eligible for online matching:", int(want_online.sum()))
    print("Rows needing a geocoded point:", int(need_point.sum()))
    print("Unique queries to geocode:", len(uniq))
    print("Cache entries loaded:", len(cache))

    start = time.time()
    new_success = 0
    results = []

    for attempted, q in enumerate(uniq, start=1):
        q_biased = bias_query(q, args.country)
        t0 = time.time()
        res = robust_geocode(
            q_biased,
            cache,
            geocode_rl,
            args.country_codes,
            args.max_retries,
            args.backoff_base,
            args.jitter,
        )
        dt = time.time() - t0

        if isinstance(res, dict):
            results.append({"_loc_first": q, "lat": res["lat"], "lon": res["lon"], "disp": res["disp"]})
            new_success += 1
            if new_success % args.save_every == 0:
                save_cache(cache_path, cache)

        if attempted % args.print_every == 0 or attempted == len(uniq):
            elapsed = time.time() - start
            rate = attempted / elapsed if elapsed > 0 else 0
            remaining = len(uniq) - attempted
            eta = remaining / rate if rate > 0 else float("inf")
            print(
                f"[{attempted}/{len(uniq)}] successes={new_success} last_dt={dt:.2f}s "
                f"rate={rate:.2f}/s ETA={eta/60:.1f} min"
            )

    save_cache(cache_path, cache)
    geo_pts = pd.DataFrame(results)

    if not geo_pts.empty:
        geo_pts_gdf = gpd.GeoDataFrame(
            geo_pts,
            geometry=gpd.points_from_xy(geo_pts["lon"], geo_pts["lat"]),
            crs="EPSG:4326",
        )

        muni_join = gpd.sjoin(
            geo_pts_gdf,
            muni_wgs84[[muni_name_col, "geometry", "_centroid"]].copy(),
            predicate="within",
            how="left",
        ).rename(columns={muni_name_col: "_muni_name"})
        prov_join = gpd.sjoin(
            geo_pts_gdf,
            prov_wgs84[[prov_name_col, "geometry", "_centroid"]].copy(),
            predicate="within",
            how="left",
        ).rename(columns={prov_name_col: "_prov_name"})

        muni_join = muni_join.drop(columns=["index_right"], errors="ignore")
        prov_join = prov_join.drop(columns=["index_right"], errors="ignore")

        def find_col(cols, target):
            for c in cols:
                if c.lower() == target.lower():
                    return c
            for c in cols:
                if target.lower() in c.lower():
                    return c
            return None

        muni_geom_col = find_col(muni_join.columns, "geometry_right") or find_col(muni_join.columns, "geometry")
        prov_geom_col = find_col(prov_join.columns, "geometry_right") or find_col(prov_join.columns, "geometry")

        joined = geo_pts.merge(
            muni_join[["_loc_first", "_muni_name", muni_geom_col]].rename(columns={muni_geom_col: "_muni_geom"}),
            on="_loc_first",
            how="left",
        ).merge(
            prov_join[["_loc_first", "_prov_name", prov_geom_col]].rename(columns={prov_geom_col: "_prov_geom"}),
            on="_loc_first",
            how="left",
        )
        joined_map = joined.set_index("_loc_first").to_dict(orient="index")

        for i, row in df.loc[need_point, ["_loc_first", "_gran"]].iterrows():
            q = str(row["_loc_first"]).strip()
            info = joined_map.get(q)
            if not info:
                continue

            lat, lon = float(info["lat"]), float(info["lon"])
            disp = info.get("disp")

            df.at[i, "geo_lat"] = lat
            df.at[i, "geo_lon"] = lon
            df.at[i, "geom_point_wkt"] = Point(lon, lat).wkt

            if df.at[i, "_gran"] in ["city", "site"]:
                df.at[i, "geo_level"] = df.at[i, "_gran"]
                df.at[i, "geo_source"] = "nominatim"
                df.at[i, "geo_name_matched"] = disp
                df.at[i, "geo_match_type"] = "geocode"
                df.at[i, "municipality_name"] = info.get("_muni_name")
                df.at[i, "province_name"] = info.get("_prov_name")
                if args.attach_admin_polygons_for_city_site and inside_country(lon, lat):
                    if info.get("_muni_geom") is not None and not pd.isna(info.get("_muni_geom")):
                        df.at[i, "geom_poly_wkt"] = info["_muni_geom"].wkt
                continue

            if df.at[i, "_gran"] == "municipality":
                if inside_country(lon, lat) and info.get("_muni_geom") is not None and not pd.isna(info.get("_muni_geom")):
                    muni_geom = info["_muni_geom"]
                    c = gpd.GeoSeries([muni_geom], crs="EPSG:4326").to_crs("EPSG:28992").centroid
                    c = gpd.GeoSeries(c, crs="EPSG:28992").to_crs("EPSG:4326").iloc[0]
                    df.at[i, "geo_level"] = "municipality"
                    df.at[i, "geo_source"] = "nominatim->sjoin(cbs)"
                    df.at[i, "geo_name_matched"] = str(info.get("_muni_name") or disp)
                    df.at[i, "geo_match_type"] = "geocode_then_polygon"
                    df.at[i, "geom_poly_wkt"] = muni_geom.wkt
                    df.at[i, "geo_lat"] = float(c.y)
                    df.at[i, "geo_lon"] = float(c.x)
                    df.at[i, "geom_point_wkt"] = Point(float(c.x), float(c.y)).wkt
                else:
                    df.at[i, "geo_level"] = "municipality"
                    df.at[i, "geo_source"] = "nominatim"
                    df.at[i, "geo_name_matched"] = disp
                    df.at[i, "geo_match_type"] = "geocode_only"
                continue

            if df.at[i, "_gran"] == "province":
                if inside_country(lon, lat) and info.get("_prov_geom") is not None and not pd.isna(info.get("_prov_geom")):
                    prov_geom = info["_prov_geom"]
                    c = gpd.GeoSeries([prov_geom], crs="EPSG:4326").to_crs("EPSG:28992").centroid
                    c = gpd.GeoSeries(c, crs="EPSG:28992").to_crs("EPSG:4326").iloc[0]
                    df.at[i, "geo_level"] = "province"
                    df.at[i, "geo_source"] = "nominatim->sjoin(cbs)"
                    df.at[i, "geo_name_matched"] = str(info.get("_prov_name") or disp)
                    df.at[i, "geo_match_type"] = "geocode_then_polygon"
                    df.at[i, "geom_poly_wkt"] = prov_geom.wkt
                    df.at[i, "geo_lat"] = float(c.y)
                    df.at[i, "geo_lon"] = float(c.x)
                    df.at[i, "geom_point_wkt"] = Point(float(c.x), float(c.y)).wkt
                else:
                    df.at[i, "geo_level"] = "province"
                    df.at[i, "geo_source"] = "nominatim"
                    df.at[i, "geo_name_matched"] = disp
                    df.at[i, "geo_match_type"] = "geocode_only"

    diag = (
        df.assign(_has_geo=df["geom_point_wkt"].notna() | df["geom_poly_wkt"].notna())
        .groupby(["_gran", "_has_geo"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["_gran", "_has_geo"])
    )
    print("\nAfter online matching:")
    print(diag)

    build_output_gpkg(df, output_gpkg)
    print("\nWrote GeoPackage:", output_gpkg)
    print("Layers: sentences_points, sentences_polygons")


if __name__ == "__main__":
    main()
