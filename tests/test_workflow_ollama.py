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
