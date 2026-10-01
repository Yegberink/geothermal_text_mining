import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from results_tracking import visualize_overarching_results as plots


class ExtremeRegionSentenceExportTests(unittest.TestCase):
    def test_exports_all_sentiments_once_and_only_framed_selected_regions(self):
        provinces = pd.DataFrame({
            "language": ["dutch"] * 7,
            "country": ["Netherlands"] * 7,
            "province_name": list("ABCDEFG"),
            "n_text_units": [plots.MIN_PROVINCE_SENTENCES] * 7,
            "polarity_balance": [-30, -20, -10, 0, 10, 20, 30],
        })
        rows = []
        for region in "ABCDEFG":
            for index, sentiment in enumerate(["negative", "neutral", "positive"]):
                rows.append({
                    "language": "dutch", "country": "Netherlands",
                    "province_name": region, "admin_level": "nuts2",
                    "sentence_uid": f"{region}-{index}", "sentence_text": sentiment,
                    "sentiment": sentiment, "matched_categories_str": "Costs;Operational risk",
                })
        rows.extend([
            rows[0].copy(),
            dict(rows[0], sentence_uid="no-frame", matched_categories_str=None),
            dict(rows[0], sentence_uid="empty-frame", matched_categories_str=" ; "),
            dict(rows[0], sentence_uid="country-level", admin_level="country"),
        ])
        result = plots.build_extreme_region_sentence_table(pd.DataFrame(rows), provinces)
        self.assertEqual(len(result), 18)
        self.assertEqual(result.province_name.unique().tolist(), list("ABCGFE"))
        self.assertEqual(result.groupby("sentiment").size().to_dict(),
                         {"negative": 6, "neutral": 6, "positive": 6})
        self.assertTrue(result.matched_categories_str.eq("Costs;Operational risk").all())
        self.assertFalse(result.sentence_uid.duplicated().any())


if __name__ == "__main__":
    unittest.main()
