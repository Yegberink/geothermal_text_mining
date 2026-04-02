import re
import pandas as pd
import geopandas as gpd
import os
os.chdir("/Users/Yannick/Documents/PhD/text_mining/geothermal")


# -----------------------------
# Configuration
# -----------------------------
MAPPING_CSV = "data/newspaper_regions.csv"
CBS_GPKG = "data/cbsgebiedsindelingen2025.gpkg"

LAYER_MUNI = "gemeente_gegeneraliseerd"
LAYER_PROV = "provincie_gegeneraliseerd"

OUT_GPKG = "output/regions_gdf_2025.gpkg"
OUT_GEOJSON = "output/regions_gdf_2025.geojson"


# -----------------------------
# CBS province code mapping
# -----------------------------
PROVINCE_CODE_MAP = {
    "GR": "PV20",  # Groningen
    "FR": "PV21",  # Friesland
    "DR": "PV22",  # Drenthe
    "OV": "PV23",  # Overijssel
    "FL": "PV24",  # Flevoland
    "GE": "PV25",  # Gelderland
    "UT": "PV26",  # Utrecht
    "NH": "PV27",  # Noord-Holland
    "ZH": "PV28",  # Zuid-Holland
    "ZE": "PV29",  # Zeeland
    "NB": "PV30",  # Noord-Brabant
    "LI": "PV31",  # Limburg
}


# -----------------------------
# Helpers
# -----------------------------
def normalize_code(code: str) -> str:
    if code is None:
        return ""
    code = str(code).strip()
    if not code:
        return ""

    if code.lower().startswith("gm"):
        digits = re.sub(r"\D", "", code)
        return "GM" + digits.zfill(4)

    code = code.upper()

    # translate province abbreviations to CBS PV codes
    if code in PROVINCE_CODE_MAP:
        return PROVINCE_CODE_MAP[code]

    return code


def split_classification(location_classification: str):
    if pd.isna(location_classification):
        return []
    parts = str(location_classification).split("+")
    return [normalize_code(p) for p in parts if p.strip()]


def ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("CBS layer has no CRS defined.")
    return gdf.to_crs("EPSG:4326")


# -----------------------------
# Main builder
# -----------------------------
def build_regions_gdf(mapping_csv: str, cbs_gpkg: str) -> gpd.GeoDataFrame:
    mapping = pd.read_csv(mapping_csv, dtype=str)

    muni = ensure_wgs84(gpd.read_file(cbs_gpkg, layer=LAYER_MUNI))
    prov = ensure_wgs84(gpd.read_file(cbs_gpkg, layer=LAYER_PROV))

    muni["_CODE"] = muni["statcode"].astype(str).str.upper()
    prov["_CODE"] = prov["statcode"].astype(str).str.upper()

    # NL = union of all provinces
    nl_geom = prov.geometry.union_all()

    out = []
    problems = []

    for _, r in mapping.iterrows():
        newspaper = r["newspaper"]
        lc_raw = r["location_classification"]
        region_name = r["region_name"]

        codes = split_classification(lc_raw)
        geoms = []

        if not codes:
            geom = None
            problems.append((newspaper, lc_raw, "empty classification"))

        elif codes == ["NL"]:
            geom = nl_geom

        else:
            for code in codes:
                if code == "NL":
                    geoms.append(nl_geom)
                elif code.startswith("GM"):
                    match = muni.loc[muni["_CODE"] == code]
                    if match.empty:
                        problems.append((newspaper, lc_raw, f"municipality not found: {code}"))
                    else:
                        geoms.append(match.geometry.union_all())
                elif code.startswith("PV"):
                    match = prov.loc[prov["_CODE"] == code]
                    if match.empty:
                        problems.append((newspaper, lc_raw, f"province not found: {code}"))
                    else:
                        geoms.append(match.geometry.union_all())
                else:
                    problems.append((newspaper, lc_raw, f"unknown code: {code}"))

            geom = None if not geoms else gpd.GeoSeries(geoms, crs="EPSG:4326").union_all()

        out.append(
            {
                "newspaper": newspaper,
                "location_classification": lc_raw,
                "region_name": region_name,
                "geometry": geom,
            }
        )

    regions_gdf = gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")

    missing = regions_gdf[regions_gdf.geometry.isna()]
    if not missing.empty:
        print("❌ Missing geometries:\n")
        print(missing[["newspaper", "location_classification", "region_name"]])
        print("\nExamples of problems:")
        for p in problems[:20]:
            print(" -", p)
        raise ValueError("Fix the above mappings before continuing.")

    return regions_gdf


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    regions_gdf = build_regions_gdf(MAPPING_CSV, CBS_GPKG)

    regions_gdf.to_file(OUT_GPKG, layer="regions_gdf", driver="GPKG")
    regions_gdf.to_file(OUT_GEOJSON, driver="GeoJSON")

    print("✅ regions_gdf created successfully")
    print("Rows:", len(regions_gdf))
    print("Saved to:")
    print(" -", OUT_GPKG)
    print(" -", OUT_GEOJSON)
