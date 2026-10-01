"""Create the Italian NUTS 2024 level-3 supplement for the shared shape parquet.

Download NUTS_RG_01M_2024_4326_LEVL_3.geojson from Eurostat GISCO, then run:
python scripts/helpers/add_italian_nuts3.py --geojson FILE
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

SOURCE_URL = (
    "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/"
    "NUTS_RG_01M_2024_4326_LEVL_3.geojson"
)


def add_italian_nuts3(shapes_path: Path, geojson_path: Path) -> None:
    shapes = pd.read_parquet(shapes_path)
    source = gpd.read_file(geojson_path).to_crs("EPSG:4326")
    italy = source[source["CNTR_CODE"].eq("IT") & source["LEVL_CODE"].eq(3)].copy()
    # Extra-regio codes have no land geometry and cannot be spatially assigned.
    italy = italy[italy.geometry.notna() & ~italy.geometry.is_empty]
    if len(italy) != 107 or italy["NUTS_ID"].duplicated().any() or not italy.geometry.is_valid.all():
        raise ValueError("Expected 107 valid, unique Italian NUTS 2024 level-3 land regions.")
    additions = pd.DataFrame({
        "shape_id": "ITA_nuts2024_" + italy["NUTS_ID"],
        "country_id": "ITA",
        "shape_class": "land",
        "geometry": italy.geometry.to_wkb(),
        "parent": "nuts",
        "parent_subtype": "3",
        "parent_id": italy["NUTS_ID"],
        "parent_name": italy["NAME_LATN"],
    })
    output = shapes_path.with_name(f"{shapes_path.stem}_italian_nuts3.parquet")
    temporary = output.with_suffix(".tmp.parquet")
    additions[shapes.columns].to_parquet(temporary, index=False)
    temporary.replace(output)
    print(f"Wrote {len(additions)} Italian NUTS3 regions to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes-parquet", type=Path, default=Path("data/shapes.parquet"))
    parser.add_argument("--geojson", type=Path, required=True)
    args = parser.parse_args()
    add_italian_nuts3(args.shapes_parquet, args.geojson)
