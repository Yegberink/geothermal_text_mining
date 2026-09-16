import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout
import io
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/model_evaluation'))
import evaluate


class MetricsExportTests(unittest.TestCase):
    def details(self):
        return pd.DataFrame([
            dict(model='example', task=task, subset='all', status='ok', n_samples=500,
                 errors=0, unfiltered_accuracy=accuracy, unfiltered_macro_f1=.7,
                 threshold=.9, coverage=.8, runtime_seconds=75000)
            for task, accuracy in [('geothermal', .828), ('sentiment', .714)]])

    def test_both_csv_dialects_keep_fractions_and_missing_tasks(self):
        for decimal, separator in [('.', ','), (',', ';')]:
            with self.subTest(decimal=decimal), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                self.details().to_csv(path / 'metrics_detailed.csv', index=False)
                evaluate.write_metrics_summary(path, decimal=decimal)
                text = (path / 'metrics.csv').read_text()
                self.assertIn('0' + decimal + '828', text)
                self.assertIn('0' + decimal + '714', text)
                frame = pd.read_csv(path / 'metrics.csv', sep=separator, decimal=decimal)
                self.assertEqual(frame.loc[0, 'is_geothermal (accuracy)'], .828)
                self.assertEqual(frame.loc[0, 'sentiment (accuracy)'], .714)
                self.assertTrue(pd.isna(frame.loc[0, 'location (accuracy)']))
                # Runtime is seconds, not a bounded score or percentage.
                self.assertEqual(frame.loc[0, 'average_runtime_seconds'], 150)

    def test_invalid_scores_fail_without_replacing_existing_report(self):
        for column in ['unfiltered_accuracy', 'unfiltered_macro_f1', 'threshold', 'coverage']:
            for value in [-.01, 1.01, 828., float('inf'), 'invalid']:
                with self.subTest(column=column, value=value), tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)
                    frame = self.details().astype({column: object})
                    frame.loc[0, column] = value
                    frame.to_csv(path / 'metrics_detailed.csv', index=False)
                    report = path / 'metrics.csv'
                    report.write_text('existing report\n')
                    with self.assertRaises(ValueError):
                        evaluate.write_metrics_summary(path)
                    self.assertEqual(report.read_text(), 'existing report\n')

    def test_decimal_option_reaches_normal_and_cached_scoring(self):
        frame = pd.DataFrame(dict(sample_id=['a', 'b'], sentence_text=['A', 'B'],
                                  gold_sentiment=['positive', 'negative']))
        predictions = pd.DataFrame(dict(model=['example'] * 2, task=['sentiment'] * 2,
            sample_id=['a', 'b'], prediction=['positive', 'positive'], confidence=[.8, .7],
            error=['', ''], seconds=[1., 1.], setup_seconds=[0., 0.]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            models = path / 'models.json'
            models.write_text(json.dumps([dict(model_id='example', backend='ollama')]))
            argv = ['evaluate.py', '--results-dir', str(path), '--models-file', str(models),
                    '--tasks', 'sentiment', '--summary-decimal', ',']
            with patch.object(evaluate, 'load_annotations', return_value={'sentence': frame}), \
                    patch.object(evaluate, 'check_ollama_models') as check, \
                    patch.object(evaluate, 'predict', return_value=predictions) as predict, \
                    redirect_stdout(io.StringIO()):
                with patch.object(sys, 'argv', argv):
                    evaluate.main()
                predictions.to_json(path / 'predictions.jsonl', orient='records', lines=True)
                with patch.object(sys, 'argv', argv + ['--rescore-only']):
                    evaluate.main()
                self.assertEqual(predict.call_count, 1)
                self.assertEqual(check.call_count, 1)
            summary = pd.read_csv(path / 'metrics.csv', sep=';', decimal=',')
            self.assertEqual(summary.loc[0, 'sentiment (accuracy)'], .5)


if __name__ == '__main__':
    unittest.main()
