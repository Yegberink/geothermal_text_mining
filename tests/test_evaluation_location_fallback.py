import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/model_evaluation'))
import evaluate
from core_workflow import locations_ollama


class LocationFallbackTests(unittest.TestCase):
    def setUp(self):
        self.config = [{'model_id': 'gemma4:12b', 'backend': 'ollama', 'think': False}]
        self.args = SimpleNamespace(tasks=['location'], device='cpu', ollama_url='http://test')

    def frames(self):
        return {'paragraph': pd.DataFrame([
            dict(sample_id='p1', document_id='a', body='Full article A', paragraph_text='Delft', region_name='', split='calibration'),
            dict(sample_id='p2', document_id='a', body='Full article A', paragraph_text='NONE', region_name='', split='calibration'),
            dict(sample_id='p3', document_id='b', body='Full article B', paragraph_text='NONE', region_name='', split='test'),
            dict(sample_id='p4', document_id='b', body='Full article B', paragraph_text='NONE', region_name='', split='test'),
        ])}

    @staticmethod
    def paragraph_prediction(task, row, *args, **kwargs):
        raw = dict(location=row.paragraph_text, confidence=.8,
                   granularity='none' if row.paragraph_text == 'NONE' else 'city')
        return row.paragraph_text.lower(), .8, raw

    def test_peer_then_full_article_fallback_and_resume(self):
        frames = self.frames()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
                patch.object(evaluate, 'ollama_predict', side_effect=self.paragraph_prediction) as paragraph, \
                patch.object(locations_ollama, 'llm_document_primary_location',
                             return_value=dict(location='Rotterdam', granularity='city', confidence=.9)) as document:
            path = Path(directory)
            first = evaluate.predict(frames, self.config, path, self.args)
            self.assertEqual(first.prediction.tolist(), ['delft', 'delft', 'rotterdam', 'rotterdam'])
            self.assertEqual(first.location_source.tolist(), ['paragraph', 'document_single_location', 'document_llm', 'document_llm'])
            self.assertEqual(first.paragraph_prediction.tolist(), ['delft', 'none', 'none', 'none'])
            self.assertEqual(document.call_count, 1)
            self.assertEqual(document.call_args.args[0], 'Full article B')
            self.assertIs(document.call_args.kwargs['think'], False)
            self.assertGreater(first.document_seconds.sum(), 0)
            self.assertEqual(first.iloc[2].document_seconds, first.iloc[3].document_seconds)
            # Labels cannot affect inference or the fallback group.
            frames['paragraph']['gold_location'] = 'changed gold'
            frames['paragraph']['gold_geothermal'] = 'NO'
            second = evaluate.predict(frames, self.config, path, self.args)
            self.assertEqual(paragraph.call_count, 4)
            self.assertEqual(document.call_count, 1)
            self.assertEqual(first.seconds.tolist(), second.seconds.tolist())
            self.assertEqual(len((path / 'predictions.jsonl').read_text().splitlines()), 4)
            frames['paragraph'].loc[0, 'body'] = 'Different article'
            with self.assertRaisesRegex(ValueError, 'Cached input changed'):
                evaluate.predict(frames, self.config, path, self.args)

    def test_failed_document_is_retried_without_repeating_paragraphs(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
                patch.object(evaluate, 'ollama_predict', side_effect=self.paragraph_prediction) as paragraph, \
                patch.object(locations_ollama, 'llm_document_primary_location', side_effect=RuntimeError('offline')) as document:
            first = evaluate.predict(self.frames(), self.config, Path(directory), self.args)
            self.assertTrue(first.iloc[2:].error.str.contains('offline').all())
            self.assertTrue(first.iloc[2:].prediction.eq(evaluate.INVALID).all())
            document.side_effect = None
            document.return_value = dict(location='NONE', granularity='none', confidence=.4)
            second = evaluate.predict(self.frames(), self.config, Path(directory), self.args)
            self.assertTrue(second.error.eq('').all())
            self.assertEqual(second.prediction.tolist(), ['delft', 'delft', 'netherlands', 'netherlands'])
            self.assertEqual(second.iloc[2:].confidence.tolist(), [0, 0])
            self.assertEqual(paragraph.call_count, 4)
            self.assertEqual(document.call_count, 2)

    def test_partial_location_run_preserves_other_predictions_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
                patch.object(evaluate, 'ollama_predict', side_effect=self.paragraph_prediction), \
                patch.object(locations_ollama, 'llm_document_primary_location',
                             return_value=dict(location='Delft', granularity='city', confidence=.9)):
            path = Path(directory)
            unrelated = json.dumps(dict(model='other', task='sentiment', sample_id='s1', signature='old')) + '\n'
            (path / 'predictions.jsonl').write_text(unrelated)
            evaluate.predict(self.frames(), self.config, path, self.args)
            self.assertTrue((path / 'predictions.jsonl').read_text().startswith(unrelated))
            old = pd.DataFrame([dict(model='other', task='sentiment', subset='all', accuracy=.7)])
            old.to_csv(path / 'metrics.csv', index=False)
            evaluate.merge_report(pd.DataFrame([dict(model='gemma', task='location', subset='all', accuracy=.5)]), path / 'metrics.csv')
            result = pd.read_csv(path / 'metrics.csv')
            self.assertEqual(result[result.task.eq('sentiment')].iloc[0].accuracy, .7)

    def test_legacy_alias_preserves_cache_but_rejects_changed_settings(self):
        args = SimpleNamespace(tasks=['sentiment'], device='cpu', ollama_url='http://test')
        frames = {'sentence': pd.DataFrame([dict(sample_id='s1', sentence_text='Zin.')])}
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
                patch.object(evaluate, 'ollama_predict', return_value=('neutral', .8, {})) as call:
            path = Path(directory)
            first = evaluate.predict(frames, self.config, path, args)
            item = first.iloc[0].to_dict()
            item['signature'] = 'legacy'
            (path / 'predictions.jsonl').write_text(json.dumps(item) + '\n')
            (path / 'cache_compatibility.json').write_text(json.dumps([
                dict(model=item['model'], task='sentiment', old_signature='legacy',
                     signature=first.iloc[0].signature)]))
            evaluate.predict(frames, self.config, path, args)
            self.assertEqual(call.call_count, 1)
            changed = [dict(self.config[0], think=True)]
            with self.assertRaisesRegex(ValueError, 'configuration/code changed'):
                evaluate.predict(frames, changed, path, args)


if __name__ == '__main__':
    unittest.main()
