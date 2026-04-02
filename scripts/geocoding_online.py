# ============================================================
# ONLINE GEOCODING (ROBUST) + PROGRESS + RESUMABLE DISK CACHE
# - Use after offline step (df, muni_wgs84, prov_wgs84, nl_poly_wgs84 exist)
# - Geocodes remaining rows -> point
# - For municipality/province: point -> sjoin to CBS polygons if within NL (European)
# - Writes GeoPackage layers again
# ============================================================

import os, re, json, time, math, random
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError
from geopy.extra.rate_limiter import RateLimiter
from shapely.ops import unary_union


# -----------------------------
# Settings
# -----------------------------
COUNTRY_CODES = "nl,be,bq,aw,cw,sx"
CACHE_PATH = "cache/nominatim_cache.json"
SAVE_EVERY = 50               # save cache every N new successful results
PRINT_EVERY = 25              # print progress every N attempts
MIN_DELAY_SECONDS = 1.1       # be gentle
TIMEOUT_SECONDS = 10          # <-- fix: increase timeout (1 sec was too low)
MAX_RETRIES = 6               # per query
BACKOFF_BASE = 1.6            # exponential backoff
JITTER = 0.25                 # random jitter to avoid thundering herd
MAX_QUERIES = None            # set e.g. 200 to test; None = run all

ATTACH_ADMIN_POLYGONS_FOR_CITY_SITE = False  # keep False if city/site should stay points only

OUT_GPKG = "output/text/sentences_with_absa_and_geo_v2.gpkg"
CBS_GPKG = "data/cbsgebiedsindelingen2025.gpkg"
LAYER_MUNI = "gemeente_gegeneraliseerd"
LAYER_PROV = "provincie_gegeneraliseerd"

muni_gdf = gpd.read_file(CBS_GPKG, layer=LAYER_MUNI)
prov_gdf = gpd.read_file(CBS_GPKG, layer=LAYER_PROV)

def with_wgs84_centroids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    # compute centroids in a projected CRS to avoid geographic centroid issues
    try:
        proj = gdf.to_crs("EPSG:28992")
    except Exception:
        proj = gdf.to_crs("EPSG:3857")
    proj["_centroid_proj"] = proj.geometry.centroid

    wgs = gdf.to_crs("EPSG:4326")
    wgs["_centroid"] = gpd.GeoSeries(proj["_centroid_proj"], crs=proj.crs).to_crs("EPSG:4326").values
    return wgs

muni_wgs84 = with_wgs84_centroids(muni_gdf)
prov_wgs84 = with_wgs84_centroids(prov_gdf)

def pick_col_by_regex(cols, patterns):
    cols_l = [c.lower() for c in cols]
    for pat in patterns:
        for c, cl in zip(cols, cols_l):
            if re.search(pat, cl):
                return c
    return None

muni_name_col = pick_col_by_regex(
    muni_gdf.columns, [r"statnaam", r"gemeente.*naam", r"gm_.*naam", r"\bnaam\b"]
)
muni_code_col = pick_col_by_regex(
    muni_gdf.columns, [r"statcode", r"gemeente.*code", r"gm_.*code", r"\bcode\b"]
)
prov_name_col = pick_col_by_regex(
    prov_gdf.columns, [r"statnaam", r"provincie.*naam", r"pv_.*naam", r"\bnaam\b"]
)
prov_code_col = pick_col_by_regex(
    prov_gdf.columns, [r"statcode", r"provincie.*code", r"pv_.*code", r"\bcode\b"]
)

# Country polygon = union of provinces
nl_poly = unary_union(prov_gdf.geometry.values)
nl_poly_wgs84 = gpd.GeoSeries([nl_poly], crs=prov_gdf.crs).to_crs("EPSG:4326").iloc[0]

# -----------------------------
# Load/resume cache
# Cache format: { "<query>": {"lat":..,"lon":..,"disp":..} or None }
# -----------------------------
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)
else:
    cache = {}

def cache_get(q):
    return cache.get(q, "__MISS__")

def cache_set(q, val):
    cache[q] = val

def cache_save():
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    os.replace(tmp, CACHE_PATH)

# -----------------------------
# Nominatim setup (timeout!)
# -----------------------------
geolocator = Nominatim(user_agent="absa-geo-mapper", timeout=TIMEOUT_SECONDS)

# RateLimiter only handles spacing; we handle retries ourselves
geocode_rl = RateLimiter(
    geolocator.geocode,
    min_delay_seconds=MIN_DELAY_SECONDS,
    swallow_exceptions=True
)

def inside_nl_europe(lon, lat):
    try:
        return bool(nl_poly_wgs84.contains(Point(float(lon), float(lat))))
    except Exception:
        return False

def robust_geocode(query: str):
    """
    Returns dict {lat, lon, disp} or None.
    Uses cache + retries w/backoff on timeouts/unavailability.
    """
    query = str(query).strip()
    if not query:
        return None

    cached = cache_get(query)
    if cached != "__MISS__":
        return cached  # can be dict or None

    # Try with countrycodes first, then without
    for attempt in range(MAX_RETRIES):
        try:
            loc = geocode_rl(
                query,
                exactly_one=True,
                addressdetails=False,
                country_codes=COUNTRY_CODES
            )
            if not loc:
                # fallback without country restriction
                loc = geocode_rl(
                    query,
                    exactly_one=True,
                    addressdetails=False
                )

            if not loc:
                cache_set(query, None)
                return None

            res = {"lat": float(loc.latitude), "lon": float(loc.longitude), "disp": str(loc)}
            cache_set(query, res)
            return res

        except (GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError) as e:
            # exponential backoff + jitter
            sleep_s = (BACKOFF_BASE ** attempt) + random.random() * JITTER
            time.sleep(sleep_s)
        except Exception:
            # unknown failure -> cache as None so we don't spin forever
            cache_set(query, None)
            return None

    # retries exhausted
    cache_set(query, None)
    return None

# -----------------------------
# Pick rows needing online work
# -----------------------------

df = pd.read_csv("output/text/sentence_offline_geocoding.csv", dtype=str)

# convert to nullable floats (so NaNs are allowed)
df["geo_lat"] = pd.to_numeric(df["geo_lat"], errors="coerce").astype("Float64")
df["geo_lon"] = pd.to_numeric(df["geo_lon"], errors="coerce").astype("Float64")

want_online = (
    df["_gran"].isin(["city", "site"])
    | ((df["_gran"].isin(["municipality", "province"])) & df["geom_poly_wkt"].isna())
)

need_point = want_online & df["geom_point_wkt"].isna() & df["_loc_first"].astype(str).str.strip().ne("")

print("Rows eligible for online matching:", int(want_online.sum()))
print("Rows needing a geocoded point:", int(need_point.sum()))
print("Cache entries loaded:", len(cache))

# Unique queries (we geocode by _loc_first to match your pipeline)
uniq = df.loc[need_point, "_loc_first"].astype(str).str.strip().unique().tolist()
if MAX_QUERIES:
    uniq = uniq[:MAX_QUERIES]
print("Unique queries to geocode:", len(uniq))

# Bias query slightly toward NL if user didn’t specify a country (helps ambiguity)
def bias_query(q: str) -> str:
    if re.search(r"\b(netherlands|nederland|belgium|belgie)\b", q, flags=re.I):
        return q
    return f"{q}, Netherlands"

# -----------------------------
# Progress loop
# -----------------------------
start = time.time()
new_success = 0
attempted = 0

results = []  # list of {"_loc_first": original, "lat":.., "lon":.., "disp":..}

for q in uniq:
    attempted += 1
    q_biased = bias_query(q)

    t0 = time.time()
    res = robust_geocode(q_biased)
    dt = time.time() - t0

    if isinstance(res, dict):
        results.append({"_loc_first": q, "lat": res["lat"], "lon": res["lon"], "disp": res["disp"]})
        new_success += 1

        # periodic cache save
        if new_success % SAVE_EVERY == 0:
            cache_save()

    # progress print
    if attempted % PRINT_EVERY == 0 or attempted == len(uniq):
        elapsed = time.time() - start
        rate = attempted / elapsed if elapsed > 0 else 0
        remaining = len(uniq) - attempted
        eta = remaining / rate if rate > 0 else float("inf")
        print(
            f"[{attempted}/{len(uniq)}] "
            f"successes={new_success} "
            f"last_dt={dt:.2f}s "
            f"rate={rate:.2f}/s "
            f"ETA={eta/60:.1f} min"
        )

# final save
cache_save()

geo_pts = pd.DataFrame(results)
print("Geocoded points:", len(geo_pts))
print("Cache saved to:", CACHE_PATH)

# -----------------------------
# Join geocoded points back & optional sjoin to polygons
# -----------------------------
if not geo_pts.empty:
    geo_pts_gdf = gpd.GeoDataFrame(
        geo_pts,
        geometry=gpd.points_from_xy(geo_pts["lon"], geo_pts["lat"]),
        crs="EPSG:4326",
    )

    # Spatial joins (only meaningful for European NL)
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

    # Identify right-geometry columns robustly (depends on geopandas version)
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

    # Fill df rows that needed points
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

        # city/site
        if df.at[i, "_gran"] in ["city", "site"]:
            df.at[i, "geo_level"] = df.at[i, "_gran"]
            df.at[i, "geo_source"] = "nominatim"
            df.at[i, "geo_name_matched"] = disp
            df.at[i, "geo_match_type"] = "geocode"
            df.at[i, "municipality_name"] = info.get("_muni_name")
            df.at[i, "province_name"] = info.get("_prov_name")
            if ATTACH_ADMIN_POLYGONS_FOR_CITY_SITE and inside_nl_europe(lon, lat):
                if info.get("_muni_geom") is not None and not pd.isna(info.get("_muni_geom")):
                    df.at[i, "geom_poly_wkt"] = info["_muni_geom"].wkt
            continue

        # municipality/province: if inside NL and we got polygon via sjoin, prefer polygon
        if df.at[i, "_gran"] == "municipality":
            if inside_nl_europe(lon, lat) and info.get("_muni_geom") is not None and not pd.isna(info.get("_muni_geom")):
                muni_geom = info["_muni_geom"]
                # centroid in projected CRS for stability
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
            if inside_nl_europe(lon, lat) and info.get("_prov_geom") is not None and not pd.isna(info.get("_prov_geom")):
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
            continue

# -----------------------------
# Diagnostics + rewrite GeoPackage
# -----------------------------
diag = (
    df.assign(_has_geo=df["geom_point_wkt"].notna() | df["geom_poly_wkt"].notna())
      .groupby(["_gran", "_has_geo"], dropna=False)
      .size()
      .reset_index(name="n")
      .sort_values(["_gran", "_has_geo"])
)
print("\nAfter online matching:")
print(diag)

if os.path.exists(OUT_GPKG):
    os.remove(OUT_GPKG)

df_points = df[df["geom_point_wkt"].notna()].copy()
gdf_points = gpd.GeoDataFrame(
    df_points,
    geometry=gpd.GeoSeries.from_wkt(df_points["geom_point_wkt"]),
    crs="EPSG:4326",
)
gdf_points.to_file(OUT_GPKG, layer="sentences_points", driver="GPKG")

df_polys = df[df["geom_poly_wkt"].notna()].copy()
gdf_polys = gpd.GeoDataFrame(
    df_polys,
    geometry=gpd.GeoSeries.from_wkt(df_polys["geom_poly_wkt"]),
    crs="EPSG:4326",
)
gdf_polys.to_file(OUT_GPKG, layer="sentences_polygons", driver="GPKG")

print("\nWrote GeoPackage:", OUT_GPKG)
print("Layers: sentences_points, sentences_polygons")
