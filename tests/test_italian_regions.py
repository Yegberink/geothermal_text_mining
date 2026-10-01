import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from core_workflow import geocoding_offline as offline
from core_workflow import geocoding_online as online
from core_workflow.geographic_aggregation import assign_regions
from helpers.shape_resources import load_shapes_parquet, nuts2_shapes, regional_shapes
from results_tracking import visualize_absa_results as plots
from results_tracking import visualize_overarching_results as overview


class ItalianRegionTests(unittest.TestCase):
    def setUp(self):
        self.shapes = gpd.GeoDataFrame({
            "country_id": ["ITA", "ITA", "ITA", "NLD", "DEU", "AUT", "CHE"],
            "parent_id": ["ITI1", "ITI17", "ITI19", "NL33", "DE11", "AT11", "CH01"],
            "parent_name": ["Toscana", "Pisa", "Siena", "Zuid-Holland", "Stuttgart", "Burgenland", "Leman"],
            "parent_subtype": ["2", "3", "3", "2", "2", "2", "2"],
            "shape_class": ["land"] * 7,
            "parent": ["nuts"] * 7,
            "geometry": [box(10, 42, 12, 44), box(10, 42, 11, 44), box(11, 42, 12, 44),
                         box(4, 51, 5, 53), box(8, 48, 9, 49), box(16, 47, 17, 48), box(6, 46, 7, 47)],
        }, crs="EPSG:4326")

    def test_only_italy_changes_level_in_mixed_scope(self):
        selected = regional_shapes(self.shapes)
        self.assertEqual(set(selected.parent_id), {"ITI17", "ITI19", "NL33", "DE11", "AT11", "CH01"})
        self.assertEqual(regional_shapes(self.shapes, "Italia").parent_subtype.tolist(), ["3", "3"])
        self.assertEqual(nuts2_shapes(self.shapes, "Italy").parent_id.tolist(), ["ITI1"])
        with self.assertRaisesRegex(ValueError, "Italian NUTS3"):
            regional_shapes(self.shapes[self.shapes.parent_subtype.ne("3")], "Italy")

    def test_assignment_preserves_country_and_coarse_mentions(self):
        frame = gpd.GeoDataFrame({
            "geo_level": ["city", "city", "country", "nuts2", "nuts3"],
            "country_id": ["ITA", "NLD", "ITA", "ITA", "ITA"],
            "llm_country": ["Italy", "Netherlands", "Italy", "Italy", "Italy"],
            "geo_name_matched": ["Town", "Delft", "Italy", "Toscana", "Siena"],
            "province_name": [None, None, "stale", "Toscana", None],
            "nuts2_id": ["ITI1", None, "ITI1", "ITI1", None],
            "geometry": [Point(10.5, 43), Point(4.5, 52), box(10, 42, 12, 44),
                         box(10, 42, 12, 44), box(11, 42, 12, 44)],
        }, crs="EPSG:4326")
        result = assign_regions(frame, regional_shapes(self.shapes))
        self.assertEqual(result.admin_level.tolist(), ["nuts3", "nuts2", "country", "nuts2", "nuts3"])
        self.assertEqual(result.loc[0, "nuts3_id"], "ITI17")
        self.assertTrue(pd.isna(result.loc[0, "nuts2_id"]))
        self.assertEqual(result.loc[1, "nuts2_id"], "NL33")
        self.assertTrue(result.loc[[2, 3], "province_name"].isna().all())
        self.assertEqual(result.loc[4, "province_name"], "Siena")
        for module in (plots, overview):
            self.assertEqual(module.province_row_mask(result).tolist(), [True, True, False, False, True])

    def test_offline_named_region_has_correct_level(self):
        result = pd.DataFrame(index=[0])
        offline.apply_geometry(result, 0, self.shapes.iloc[1], "nuts3", "shapes_parquet", "nuts3_name_or_id")
        self.assertEqual(result.loc[0, "geo_level"], "nuts3")
        self.assertEqual(result.loc[0, "nuts3_id"], "ITI17")
        self.assertTrue(pd.isna(result.loc[0, "nuts2_id"]))

    def test_cached_points_and_nuts3_overrides(self):
        frame = online.ensure_output_columns(pd.DataFrame({
            "llm_location": ["Town", "Delft", "alias", "Toscana"],
            "llm_granularity": ["city", "city", "province", "region"],
            "geo_source": ["nominatim_cache", "geonames", None, "manual_override"],
            "geo_level": ["city", "city", None, "nuts2"],
            "geom_point_wkt": [Point(10.5, 43).wkt, Point(4.5, 52).wkt, None, Point(10.5, 43).wkt],
        }))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "overrides.csv"
            pd.DataFrame([{"location": "alias", "action": "nuts3", "nuts3_id": "ITI19"}]).to_csv(path, index=False)
            overrides = online.load_location_geocoding_overrides(path)
        regions = regional_shapes(self.shapes)
        frame = online.apply_location_geocoding_overrides(frame, overrides, regions)
        result = online.apply_region_assignment(frame, regions, attach_polygons=True)
        self.assertEqual(result.loc[0, "nuts3_id"], "ITI17")
        self.assertEqual(result.loc[1, "nuts2_id"], "NL33")
        self.assertEqual(result.loc[2, "nuts3_id"], "ITI19")
        self.assertTrue(pd.isna(result.loc[3, "nuts3_id"]))
        self.assertTrue(result.loc[:2, "geom_poly_wkt"].notna().all())

    def test_repository_boundaries_and_maps_use_selected_levels(self):
        shapes = load_shapes_parquet(ROOT / "data/shapes.parquet")
        regions = regional_shapes(shapes)
        self.assertEqual(regions.groupby("country_id").size().to_dict(),
                         {"ITA": 107, "NLD": 12, "DEU": 38, "AUT": 9, "CHE": 7})
        with patch.object(overview, "load_shapes_parquet", return_value=self.shapes):
            polygons = overview.load_province_polygons(["italian", "dutch"], ["Italy", "Netherlands"], "unused")
        self.assertEqual(set(polygons.province_code), {"ITI17", "ITI19", "NL33"})

    def test_italian_provinces_survive_summary_filters(self):
        frame = pd.DataFrame({
            "province_name": ["Pisa", "Siena", None],
            "admin_level": ["nuts3", "nuts3", "country"],
            "country_id": ["ITA"] * 3,
            "sentiment": ["positive", "negative", "neutral"],
        })
        with patch.object(plots, "MIN_PROVINCE_SENTENCES", 1):
            summary = plots.build_province_summary(frame, "Italy", {})
        provinces = summary[summary.aggregation_level.eq("nuts3")]
        self.assertEqual(set(provinces.province_name), {"Pisa", "Siena"})
        self.assertEqual(provinces.n_text_units.sum(), 2)


if __name__ == "__main__":
    unittest.main()
