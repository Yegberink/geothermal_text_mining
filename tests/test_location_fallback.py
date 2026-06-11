import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import locations_ollama as locations  # noqa: E402


def base_row(uid: str, body_hash: str, location: str, granularity: str = "municipality") -> dict:
    return {
        "uid": uid,
        "body_hash": body_hash,
        "body": "Full document text",
        "source": "Paper",
        "document_title": "Title",
        "publish_date": "2024-01-01",
        "paragraph_id": int(uid[-1]),
        "paragraph_text": f"Paragraph {uid}",
        "region_name": "Region",
        "llm_location": location,
        "llm_granularity": granularity,
        "llm_confidence": 0.8 if location != "NONE" else 0.0,
        "llm_reasoning_short": "paragraph result",
        "llm_status": "ok",
        "llm_error": None,
    }


class LocationFallbackTest(unittest.TestCase):
    def apply_fallback(self, rows, resolver):
        df = pd.DataFrame(rows).set_index("uid")
        return locations.apply_document_location_fallback(
            out=df,
            text_col="paragraph_text",
            region_col="region_name",
            country="Netherlands",
            document_location_resolver=resolver,
        )

    def test_blank_paragraph_copies_single_document_location(self) -> None:
        out = self.apply_fallback(
            [
                base_row("p1", "doc1", "Delft"),
                base_row("p2", "doc1", "NONE", "none"),
            ],
            lambda *_: self.fail("Document LLM should not be called for one document location."),
        )

        self.assertEqual(out.loc["p2", "llm_location"], "Delft")
        self.assertEqual(out.loc["p2", "llm_location_source"], "document_single_location")
        self.assertFalse(bool(out.loc["p2", "llm_location_review_flag"]))
        self.assertFalse(bool(out.loc["p2", "llm_country_fallback_applied"]))
        self.assertEqual(out.loc["p1", "llm_location"], "Delft")
        self.assertEqual(out.loc["p1", "llm_location_source"], "paragraph")

    def test_multiple_document_locations_uses_document_llm_and_flags_review(self) -> None:
        calls = []

        def resolver(document_text, region_name, document_key):
            calls.append((document_text, region_name, document_key))
            return {
                "location": "Rotterdam",
                "granularity": "municipality",
                "confidence": 0.7,
                "reasoning_short": "document result",
            }

        out = self.apply_fallback(
            [
                base_row("p1", "doc1", "Delft"),
                base_row("p2", "doc1", "Utrecht"),
                base_row("p3", "doc1", "NONE", "none"),
            ],
            resolver,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(out.loc["p3", "llm_location"], "Rotterdam")
        self.assertEqual(out.loc["p3", "llm_location_source"], "document_llm")
        self.assertTrue(bool(out.loc["p3", "llm_location_review_flag"]))
        self.assertEqual(out.loc["p3", "llm_location_review_reason"], "multiple_document_locations")
        self.assertEqual(out.loc["p3", "llm_document_location_raw"], "Rotterdam")
        self.assertFalse(bool(out.loc["p3", "llm_document_returned_none"]))
        self.assertEqual(out.loc["p1", "llm_location"], "Delft")
        self.assertEqual(out.loc["p2", "llm_location"], "Utrecht")

    def test_all_blank_document_uses_document_llm(self) -> None:
        out = self.apply_fallback(
            [
                base_row("p1", "doc1", "NONE", "none"),
                base_row("p2", "doc1", "NONE", "none"),
            ],
            lambda *_: {
                "location": "Groningen",
                "granularity": "province",
                "confidence": 0.6,
                "reasoning_short": "document result",
            },
        )

        self.assertEqual(set(out["llm_location"]), {"Groningen"})
        self.assertEqual(set(out["llm_location_source"]), {"document_llm"})
        self.assertEqual(set(out["llm_location_review_reason"]), {"no_document_paragraph_locations"})
        self.assertEqual(set(out["llm_document_location_raw"]), {"Groningen"})
        self.assertFalse(
            out["llm_document_returned_none"]
            .map(
                lambda value: False
                if pd.isna(value)
                else str(value).strip().lower() in {"1", "true", "yes", "y"}
            )
            .any()
        )

    def test_country_fallback_is_last_resort_when_document_llm_returns_none(self) -> None:
        out = self.apply_fallback(
            [base_row("p1", "doc1", "NONE", "none")],
            lambda *_: {
                "location": "NONE",
                "granularity": "none",
                "confidence": 0.0,
                "reasoning_short": "no document location",
            },
        )

        self.assertEqual(out.loc["p1", "llm_location"], "Netherlands")
        self.assertEqual(out.loc["p1", "llm_granularity"], "country")
        self.assertEqual(out.loc["p1", "llm_location_source"], "country_fallback")
        self.assertEqual(out.loc["p1", "llm_document_location_raw"], "NONE")
        self.assertTrue(bool(out.loc["p1", "llm_document_returned_none"]))
        self.assertTrue(bool(out.loc["p1", "llm_country_fallback_applied"]))
        self.assertTrue(bool(out.loc["p1", "llm_location_review_flag"]))
        self.assertEqual(
            out.loc["p1", "llm_location_review_reason"],
            "no_document_paragraph_locations;document_llm_no_location",
        )


if __name__ == "__main__":
    unittest.main()
