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


class SpecialistLoadingTests(unittest.TestCase):
    def test_explicit_loader_options_preserve_existing_checkpoint_format(self):
        import transformers
        with patch.object(transformers.AutoTokenizer, 'from_pretrained') as tokenizer, \
                patch.object(transformers.AutoModelForSequenceClassification, 'from_pretrained') as model:
            tokenizer.return_value.model_max_length = 512
            model.return_value.to.return_value.eval.return_value.config.label2id = {'neutral': 0}
            base = dict(model_id='test', label_names={'0': 'neutral'}, revision='fixed')
            evaluate.HuggingFaceRunner(base, 'cpu')
            self.assertIs(tokenizer.call_args.kwargs['use_fast'], True)
            self.assertIs(model.call_args.kwargs['use_safetensors'], False)
            evaluate.HuggingFaceRunner(dict(base, use_fast=False, use_safetensors=True), 'cpu')
            self.assertIs(tokenizer.call_args.kwargs['use_fast'], False)
            self.assertIs(model.call_args.kwargs['use_safetensors'], True)
            self.assertNotIn('ignore_mismatched_sizes', model.call_args.kwargs)
            self.assertEqual(model.call_args.kwargs['revision'], 'fixed')

    def test_setup_error_reported_once_and_retry_ignores_old_failed_signature(self):
        frames = {'sentence': pd.DataFrame([
            dict(sample_id=str(i), sentence_text=f'Zin {i}', gold_sentiment='neutral',
                 split='calibration' if i < 2 else 'test') for i in range(5)])}
        config = [dict(model_id='test', backend='huggingface', label_names={'0': 'neutral'})]
        args = SimpleNamespace(tasks=['sentiment'], device='cpu', ollama_url='http://test')
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(output), \
                patch.object(evaluate, 'HuggingFaceRunner', side_effect=RuntimeError('broken tokenizer')) as loader:
            path = Path(directory)
            failed = evaluate.predict(frames, config, path, args)
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(output.getvalue().count('SETUP FAILED'), 1)
            self.assertNotIn('sentiment 5/5', output.getvalue())
            self.assertTrue(failed.error_stage.eq('setup').all())
            evaluate.evaluate_predictions(failed, frames, path, .8)
            summary = pd.read_csv(path / 'metrics_detailed.csv').iloc[0]
            self.assertEqual(summary.status, 'model setup failed')
            self.assertTrue(pd.isna(summary.accuracy))
            old = failed.to_dict('records')
            for item in old:
                item['signature'] = 'old incompatible loader'
            (path / 'predictions.jsonl').write_text(''.join(json.dumps(item) + '\n' for item in old))
            loader.side_effect = None
            loader.return_value.return_value = ('neutral', .9, {})
            retried = evaluate.predict(frames, config, path, args)
            self.assertTrue(retried.error.eq('').all())
            self.assertEqual(loader.call_count, 2)


if __name__ == '__main__':
    unittest.main()
