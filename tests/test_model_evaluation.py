import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/model_evaluation'))
import evaluate
import prepare


class EvaluationTests(unittest.TestCase):
    def test_sampling_blind_and_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.csv'
            pd.DataFrame([{'paragraph_text': f'Text {i}', 'body_hash': str(i),
                           'llm_is_geothermal': 'YES', 'sentiment': 'positive',
                           'geo_hits': 8} for i in range(30)]).to_csv(path, index=False)
            first, count = prepare.sample_frame(path, 'paragraph', 12, 42)
            second, _ = prepare.sample_frame(path, 'paragraph', 12, 42)
            self.assertTrue(first.equals(second))
            self.assertEqual(count, 30)
            self.assertEqual(set(first), {'sample_id', 'document_id', 'paragraph_text',
                                         'region_name', 'gold_location', 'gold_geothermal', 'notes'})
            self.assertTrue(first.gold_location.eq('').all())
            with self.assertRaises(ValueError):
                prepare.sample_frame(path, 'paragraph', 31, 42)

    def test_five_class_aggregation_and_question(self):
        prediction, confidence, _ = evaluate.collapse_probabilities(
            [.25, .26, .4, .04, .05], {str(i): str(i + 1) for i in range(5)})
        self.assertEqual(prediction, 'negative')
        self.assertAlmostEqual(confidence, .51)
        prediction, _, _ = evaluate.collapse_probabilities(
            [.7, .1, .1, .1], {'0': 'question', '1': 'negative', '2': 'neutral', '3': 'positive'})
        self.assertEqual(prediction, evaluate.INVALID)

    def test_threshold_constraint_and_fixed_classes(self):
        frame = pd.DataFrame({'gold': ['YES'] * 4 + ['NO'],
                              'prediction': ['YES', 'YES', 'YES', 'NO', 'NO'],
                              'confidence': [.9, .9, .9, .1, .8], 'valid': [True] * 5})
        curve = evaluate.threshold_curve(frame, ['YES', 'NO'])
        self.assertEqual(evaluate.choose_threshold(curve, .8), .8)
        self.assertEqual(evaluate.choose_threshold(curve, 1), 0)
        self.assertEqual(evaluate.scores(['YES'], ['YES'], ['YES', 'NO'])['macro_f1'], .5)
        frame.valid = False
        self.assertIsNone(evaluate.choose_threshold(evaluate.threshold_curve(frame, ['YES', 'NO']), .8))

    def test_unchanged_workflow_calls(self):
        from core_workflow import sentiment_classification, is_geothermal, locations_ollama
        row = pd.Series({'sentence_text': 'Zin.', 'paragraph_text': 'Tekst.', 'region_name': 'Netherlands'})
        with patch.object(sentiment_classification, 'call_ollama_sentiment', return_value={'sentiment': 'neutral', 'confidence': .5}) as call:
            evaluate.ollama_predict('sentiment', row, 'phi3', 'http://test')
            self.assertEqual(call.call_args.args, ('Zin.', 'phi3', 'http://test', 'dutch', 'zero_shot', 120))
        with patch.object(is_geothermal, 'llm_is_geothermal', return_value={'is_geothermal': 'NO', 'confidence': .9}) as call:
            evaluate.ollama_predict('geothermal', row, 'phi3', 'http://test')
            self.assertEqual(call.call_args.args[:5], ('Tekst.', 'Netherlands', 'http://test', 'phi3', 'dutch'))
        with patch.object(locations_ollama, 'llm_primary_location', return_value={'location': 'NONE', 'confidence': .9}) as call:
            self.assertEqual(evaluate.ollama_predict('location', row, 'phi3', 'http://test')[0], 'none')
            self.assertEqual(call.call_args.args[0], 'Tekst.')

    def test_end_to_end_resume_and_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            manifest = {'sources': {}, 'document_splits': {'doc1': 'calibration', 'doc2': 'test'}}
            for kind in ['paragraph', 'sentence']:
                frame = pd.DataFrame({'sample_id': [kind + '1', kind + '2'],
                    'document_id': ['doc1', 'doc2'], kind + '_text': ['Tekst A', 'Tekst B']})
                if kind == 'paragraph':
                    frame['region_name'] = 'Netherlands'
                manifest['sources'][kind] = {'rows': frame.to_dict('records')}
                if kind == 'paragraph':
                    frame['gold_location'] = 'NONE'
                    frame['gold_geothermal'] = 'NO'
                else:
                    frame['gold_sentiment'] = 'neutral'
                frame.to_csv(path / (kind + 's.csv'), index=False)
            (path / 'manifest.json').write_text(json.dumps(manifest))
            frames = evaluate.load_annotations(path)
            args = SimpleNamespace(tasks=['sentiment'], device='cpu', ollama_url='http://test')
            configs = [{'model_id': 'phi3', 'backend': 'ollama'}]
            with patch.object(evaluate, 'ollama_predict', return_value=('neutral', .9, {})) as call:
                first = evaluate.predict(frames, configs, path, args)
                second = evaluate.predict(frames, configs, path, args)
                self.assertEqual(call.call_count, 2)
                self.assertTrue(first.equals(second))
            evaluate.evaluate_predictions(second, frames, path, .8)
            self.assertEqual(pd.read_csv(path / 'metrics.csv').iloc[0].accuracy, 1.)
            frame = pd.read_csv(path / 'sentences.csv')
            frame.loc[0, 'sentence_text'] = 'Tampered text'
            frame.to_csv(path / 'sentences.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'source text/metadata changed'):
                evaluate.load_annotations(path)

    def test_test_labels_do_not_change_threshold(self):
        predictions = pd.DataFrame({'sample_id': ['a', 'b', 'c', 'd'], 'model': ['test'] * 4,
            'task': ['sentiment'] * 4, 'prediction': ['positive', 'negative', 'neutral', 'positive'],
            'confidence': [.9, .7, .9, .8], 'error': [''] * 4, 'seconds': [1.] * 4,
            'setup_seconds': [0.] * 4})
        frame = pd.DataFrame({'sample_id': ['a', 'b', 'c', 'd'],
            'gold_sentiment': ['positive', 'neutral', 'neutral', 'negative'],
            'split': ['calibration', 'calibration', 'test', 'test']})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            evaluate.evaluate_predictions(predictions, {'sentence': frame}, path, .5)
            initial = pd.read_csv(path / 'metrics.csv').iloc[0]
            frame.loc[frame.split.eq('test'), 'gold_sentiment'] = 'positive'
            evaluate.evaluate_predictions(predictions, {'sentence': frame}, path, .5)
            revised = pd.read_csv(path / 'metrics.csv').iloc[0]
            self.assertEqual(initial.threshold, revised.threshold)
            self.assertEqual(initial.runtime_seconds, 4.)


if __name__ == '__main__':
    unittest.main()
