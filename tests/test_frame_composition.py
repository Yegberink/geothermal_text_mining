import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from results_tracking import visualize_overarching_results as plots


class FrameCompositionTests(unittest.TestCase):
    def test_missing_frames_do_not_enter_denominator(self):
        admin = pd.DataFrame({
            "language": ["dutch"] * 80,
            "country": ["Netherlands"] * 80,
            "country_color": ["#F58518"] * 80,
            "province_name": ["A"] * 40 + ["B"] * 40,
            "_sent": ["positive"] * 80,
            "matched_categories_str": ["Operational risk;Costs", "Costs", "", []]
            + [None] * 36 + [float("nan")] * 40,
        })
        provinces = pd.DataFrame({
            "language": ["dutch"] * 2,
            "country": ["Netherlands"] * 2,
            "country_color": ["#F58518"] * 2,
            "province_name": ["A", "B"],
            "n_text_units": [40, 40],
            "polarity_balance": [-10, 10],
        })
        extreme = plots.build_country_extreme_province_frame_share_table(admin, provinces, ["dutch"])
        a = extreme.loc[extreme.province_name.eq("A")].set_index("frame")
        self.assertEqual(set(a.index), {"Operational risk", "Costs"})
        self.assertEqual(a.loc["Costs", "n_mentions"], 2)
        self.assertTrue(a.n_total_frame_mentions.eq(3).all())
        self.assertAlmostEqual(a.loc["Costs", "share_pct"], 200 / 3)
        stacked = plots.prepare_stacked_frame_data(extreme)
        totals = stacked.groupby("region").percentage.sum()
        self.assertAlmostEqual(totals["A"], 100)
        self.assertEqual(totals["B"], 0)

    def test_exported_percentages_match_bar_widths_and_labels(self):
        # Previously, a hidden missing-frame category diluted these labels.
        extreme = pd.DataFrame({
            "country": ["Netherlands"] * 3,
            "province_name": ["A"] * 3,
            "province_role": ["Most negative province"] * 3,
            "frame": ["Operational risk", "Costs", "nan"],
            "share_pct": [10, 15, 75],
        })
        stacked = plots.prepare_stacked_frame_data(extreme)
        values = stacked.set_index("frame").percentage
        self.assertEqual(values["Operational risk"], 40)
        self.assertEqual(values["Costs"], 60)
        with tempfile.TemporaryDirectory() as tmp, patch.object(plots.plt, "close"):
            plots.plot_frame_composition_100pct(stacked, Path(tmp) / "composition.png")
            fig = plt.gcf()
            ax = fig.axes[0]
            self.assertEqual([bar.get_width() for bar in ax.patches], [40, 60])
            self.assertEqual([text.get_text() for text in ax.texts], ["40%", "60%"])
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
