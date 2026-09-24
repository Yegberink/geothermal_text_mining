import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from core_workflow import is_geothermal, locations_ollama, sentiment_classification
from helpers.country_scope import CountryScope


class WorkflowOllamaTests(unittest.TestCase):
    def test_sentiment_parse_failure_retries_with_schema_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            kwargs = dict(
                df=pd.DataFrame({"sentence_text": ["1."]}),
                text_col="sentence_text", cache_path=Path(directory) / "cache.jsonl",
                model_name="ministral-3:14b", ollama_url="http://test/api/generate",
                language="dutch", prompt_variant="zero_shot", timeout=120,
                sleep_s=0, think=False,
            )
            with patch.object(sentiment_classification.urllib.request, "urlopen") as request:
                request.return_value.__enter__.return_value.read.side_effect = [
                    json.dumps({"response": '{"error": "No sentence provided."}'}).encode(),
                    json.dumps({"response": json.dumps(dict(
                        sentiment="neutral", confidence=0.9, rationale_short="A number."
                    ))}).encode(),
                ]
                result = sentiment_classification.batch_sentiment_resumable(**kwargs)
                self.assertEqual(result.loc[0, "sentiment_status"], "ok")
                self.assertEqual(result.loc[0, "sentiment"], "neutral")
                self.assertEqual(request.call_count, 2)
                first, retry = [json.loads(c.args[0].data) for c in request.call_args_list]
                self.assertNotIn("format", first)
                self.assertEqual(retry["format"]["properties"]["sentiment"]["enum"],
                                 sentiment_classification.SENTIMENTS)
                self.assertEqual(retry["prompt"], first["prompt"])
                sentiment_classification.batch_sentiment_resumable(**kwargs)
                self.assertEqual(request.call_count, 2)

    def test_sentiment_parse_retry_is_bounded(self):
        with patch.object(sentiment_classification.urllib.request, "urlopen") as request:
            request.return_value.__enter__.return_value.read.return_value = json.dumps(
                {"response": '{"error": "No sentence provided."}'}
            ).encode()
            with self.assertRaisesRegex(ValueError, "parseable sentiment"):
                sentiment_classification.call_ollama_sentiment(
                    "1.", "ministral-3:14b", "http://test/api/generate",
                    "dutch", "zero_shot", 120, think=False,
                )
            self.assertEqual(request.call_count, 2)

    def test_batch_requests_and_cache_isolation(self):
        """Resume identical settings, but rerun after a model or thinking change."""
        raw = dict(is_geothermal="YES", sentiment="positive", location="NONE",
                   granularity="none", confidence=0.9, evidence_short="Test evidence.",
                   reasoning_short="Test evidence.")
        response = {"response": json.dumps(raw)}
        for task in ("geothermal", "location", "sentiment"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                cache = Path(directory) / "cache.jsonl"
                frame = pd.DataFrame([dict(paragraph_text="Aardwarmte in Nederland.",
                                           sentence_text="Aardwarmte is goed.",
                                           region_name="Netherlands", body_hash="doc1")])

                def run(model, think):
                    common = dict(df=frame, cache_path=cache, sleep_s=0,
                                  ollama_url="http://test/api/generate", think=think)
                    if task == "geothermal":
                        return is_geothermal.batch_geothermal_resumable(
                            **common, text_col="paragraph_text", model=model,
                            country="Netherlands", language="dutch",
                            geothermal_lexicon={"strong": ["aardwarmte"],
                                                "contextual": [], "competing": []})
                    if task == "location":
                        return locations_ollama.batch_primary_locations_resumable(
                            **common, text_col="paragraph_text", region_col="region_name",
                            model=model, country_scope=CountryScope(("Netherlands",)))
                    return sentiment_classification.batch_sentiment_resumable(
                        **common, text_col="sentence_text", model_name=model,
                        language="dutch", prompt_variant="zero_shot", timeout=120)

                with patch.object(is_geothermal.requests, "post") as post, \
                        patch.object(sentiment_classification.urllib.request, "urlopen") as urlopen:
                    post.return_value.json.return_value = response
                    urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(response).encode()
                    request = urlopen if task == "sentiment" else post
                    calls_per_run = 2 if task == "location" else 1
                    # NONE forces the location batch through the document fallback too.
                    settings = [("llama3.1:8b", None), ("ministral-3:14b", False),
                                ("ministral-3:14b", True)]
                    for index, (model, think) in enumerate(settings, 1):
                        result = run(model, think)
                        status = "sentiment_status" if task == "sentiment" else "llm_status"
                        self.assertTrue(result[status].eq("ok").all())
                        self.assertEqual(request.call_count, index * calls_per_run)
                        for call in request.call_args_list[-calls_per_run:]:
                            payload = (json.loads(call.args[0].data) if task == "sentiment"
                                       else call.kwargs["json"])
                            self.assertEqual(payload["model"], model)
                            if think is None:
                                self.assertNotIn("think", payload)
                            else:
                                self.assertIs(payload["think"], think)
                            self.assertNotIn("think", payload["options"])
                        run(model, think)
                        self.assertEqual(request.call_count, index * calls_per_run)


if __name__ == "__main__":
    unittest.main()
