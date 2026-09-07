#!/usr/bin/env python3
"""Evaluate manual annotations independently of Snakemake and workflow caches."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import file_digest  # noqa: E402

LABELS = {'geothermal': ['YES', 'NO'], 'sentiment': ['negative', 'neutral', 'positive']}
INVALID = '__unsupported__'


def normalize_location(text):
    return ' '.join(unicodedata.normalize('NFKC', str(text)).casefold().replace('’', "'").split())


def scores(gold, predicted, labels):
    gold, predicted = np.asarray(gold), np.asarray(predicted)
    if not len(gold):
        return {'accuracy': None, 'macro_f1': None}
    f1 = []
    for label in labels:
        tp = np.sum((gold == label) & (predicted == label))
        fp = np.sum((gold != label) & (predicted == label))
        fn = np.sum((gold == label) & (predicted != label))
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {'accuracy': float(np.mean(gold == predicted)), 'macro_f1': float(np.mean(f1))}


def threshold_curve(frame, labels):
    result = []
    for threshold in sorted({0.0, 1.0, *frame.confidence.astype(float)}):
        kept = frame[(frame.confidence >= threshold) & frame.valid]
        result.append({'threshold': threshold, 'coverage': len(kept) / len(frame),
                       'n_accepted': len(kept), **scores(kept.gold, kept.prediction, labels)})
    return pd.DataFrame(result)


def choose_threshold(curve, min_coverage):
    eligible = curve[(curve.coverage >= min_coverage) & curve.macro_f1.notna()]
    if eligible.empty:
        return None
    return float(eligible.sort_values(['macro_f1', 'coverage', 'threshold'],
                                      ascending=[False, False, True]).iloc[0].threshold)


def load_annotations(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    frames = {}
    for kind in ['paragraph', 'sentence']:
        frame = pd.read_csv(directory / f'{kind}s.csv', dtype=str, keep_default_na=False)
        expected = pd.DataFrame(manifest['sources'][kind]['rows'])
        if frame.sample_id.duplicated().any() or set(frame.sample_id) != set(expected.sample_id):
            raise ValueError(f'{kind}: sample IDs changed or duplicated')
        actual = frame.set_index('sample_id').sort_index()
        original = expected.set_index('sample_id').sort_index()
        if not actual[original.columns].equals(original):
            raise ValueError(f'{kind}: source text/metadata changed; edit only gold_* and notes')
        frame['split'] = frame.document_id.map(manifest['document_splits'])
        for column in [c for c in frame if c.startswith('gold_')]:
            frame[column] = frame[column].str.strip()
            if frame[column].eq('').any():
                raise ValueError(f'{kind}: complete all {column} labels before evaluation')
        if kind == 'paragraph':
            frame.gold_geothermal = frame.gold_geothermal.str.upper()
            if not frame.gold_geothermal.isin(LABELS['geothermal']).all():
                raise ValueError('gold_geothermal must be YES or NO')
            frame.gold_location = frame.gold_location.map(normalize_location)
            if frame.gold_location.isin(['nan', 'null', INVALID]).any():
                raise ValueError('Use NONE for no primary location')
        else:
            frame.gold_sentiment = frame.gold_sentiment.str.lower()
            if not frame.gold_sentiment.isin([*LABELS['sentiment'], 'not_geothermal']).all():
                raise ValueError('gold_sentiment must be negative, neutral, positive, or not_geothermal')
            frame = frame[frame.gold_sentiment.ne('not_geothermal')].copy()
        if set(frame.split) != {'calibration', 'test'}:
            raise ValueError(f'{kind}: both calibration and test need annotated examples')
        frames[kind] = frame
    return frames


def sentiment_class(name):
    label = name.lower()
    if label in ['1', '2', 'very negative', 'negative']:
        return 'negative'
    if label in ['3', 'neutral']:
        return 'neutral'
    if label in ['4', '5', 'very positive', 'positive']:
        return 'positive'
    if label == 'question':
        return INVALID
    raise ValueError(f'Unknown model label: {name}')


def collapse_probabilities(probabilities, names):
    combined = {}
    for index, probability in enumerate(probabilities):
        label = sentiment_class(names[str(index)])
        combined[label] = combined.get(label, 0.0) + float(probability)
    prediction = max(combined, key=combined.get)
    return prediction, combined[prediction], combined


class HuggingFaceRunner:
    def __init__(self, config, device):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = device
        self.names = config['label_names']
        kwargs = {'revision': config.get('revision', 'main'), 'trust_remote_code': False}
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_id'], **kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(config['model_id'], **kwargs).to(device).eval()
        actual = {str(v): k for k, v in self.model.config.label2id.items()}
        if actual != self.names:
            raise ValueError('Model label mapping changed; review models.json before evaluation')
        self.revision = getattr(self.model.config, '_commit_hash', None)
        self.max_length = min(self.tokenizer.model_max_length, 512)
        self.sync()

    def sync(self):
        if self.device.startswith('cuda'):
            self.torch.cuda.synchronize()
        elif self.device == 'mps':
            self.torch.mps.synchronize()

    def __call__(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=self.max_length).to(self.device)
        with self.torch.inference_mode():
            probabilities = self.model(**inputs).logits.softmax(dim=-1)[0].cpu().tolist()
        self.sync()
        prediction, confidence, combined = collapse_probabilities(probabilities, self.names)
        return prediction, confidence, {'probabilities': combined, 'revision': self.revision}


def ollama_predict(task, row, model, url):
    # Direct calls reuse the production prompts, system messages, generation options,
    # parsers and retry behavior. No workflow batch filters, fallbacks or caches.
    from core_workflow import is_geothermal, locations_ollama, sentiment_classification
    from helpers.country_scope import CountryScope
    if task == 'sentiment':
        raw = sentiment_classification.call_ollama_sentiment(
            row.sentence_text, model, url, 'dutch', 'zero_shot', 120)
        return raw['sentiment'], raw['confidence'], raw
    if task == 'geothermal':
        raw = is_geothermal.llm_is_geothermal(
            row.paragraph_text, 'Netherlands', url, model, 'dutch',
            is_geothermal.load_geothermal_lexicon(ROOT, 'dutch'))
        return raw['is_geothermal'], raw['confidence'], raw
    raw = locations_ollama.llm_primary_location(
        row.paragraph_text, row.region_name or 'Netherlands', CountryScope(('Netherlands',)), url, model)
    return normalize_location(raw['location']), raw['confidence'], raw


def predict(frames, configs, output, args):
    prediction_path = output / 'predictions.jsonl'
    context = {
        'models': configs, 'device': args.device, 'ollama_url': args.ollama_url,
        'code': {str(p.relative_to(ROOT)): file_digest(p) for p in [
            Path(__file__), ROOT / 'scripts/core_workflow/is_geothermal.py',
            ROOT / 'scripts/core_workflow/locations_ollama.py',
            ROOT / 'scripts/core_workflow/sentiment_classification.py',
            ROOT / 'scripts/helpers/country_scope.py', ROOT / 'data/vocab/dutch/geo_keywords.yaml']},
        'machine': platform.platform(),
    }
    signature = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    cache = {}
    if prediction_path.exists():
        for line in prediction_path.read_text().splitlines():
            item = json.loads(line)
            if item['signature'] != signature:
                raise ValueError('Evaluation configuration/code changed; choose a new --results-dir')
            if not item['error']:
                cache[(item['model'], item['task'], item['sample_id'])] = item
    (output / 'run.json').write_text(json.dumps(context, indent=2) + '\n')
    records = []
    for config in configs:
        model, backend = config['model_id'], config['backend']
        runner, setup_error = None, ''
        tasks = args.tasks if backend == 'ollama' else [t for t in args.tasks if t == 'sentiment']
        for task in tasks:
            frame = frames['sentence' if task == 'sentiment' else 'paragraph']
            for row in frame.itertuples(index=False):
                key = (model, task, row.sample_id)
                input_hash = hashlib.sha256((getattr(row, 'sentence_text', '') + getattr(row, 'paragraph_text', '') + getattr(row, 'region_name', '')).encode()).hexdigest()
                if key in cache:
                    item = cache[key]
                    if item['input_hash'] != input_hash:
                        raise ValueError('Cached input changed; choose a new results directory')
                    records.append(item)
                    continue
                setup_seconds = 0.0
                if backend == 'huggingface' and runner is None and not setup_error:
                    start = time.perf_counter()
                    try:
                        runner = HuggingFaceRunner(config, args.device)
                    except Exception as exc:
                        setup_error = repr(exc)
                    setup_seconds = time.perf_counter() - start
                start = time.perf_counter()
                try:
                    if setup_error:
                        raise RuntimeError(setup_error)
                    prediction, confidence, raw = (runner(row.sentence_text) if backend == 'huggingface'
                                                  else ollama_predict(task, row, model, args.ollama_url))
                    confidence = float(confidence)
                    if not np.isfinite(confidence) or not 0 <= confidence <= 1:
                        raise ValueError('Invalid confidence')
                    error = ''
                except Exception as exc:
                    prediction, confidence, raw, error = INVALID, 0.0, {}, repr(exc)
                item = dict(model=model, backend=backend, task=task, sample_id=row.sample_id,
                            prediction=prediction, confidence=confidence, raw=raw, error=error,
                            seconds=time.perf_counter() - start, setup_seconds=setup_seconds,
                            signature=signature, input_hash=input_hash)
                with prediction_path.open('a') as handle:
                    handle.write(json.dumps(item, ensure_ascii=False) + '\n')
                records.append(item)
            print(f'{model}: {task} complete', flush=True)
        del runner
        gc.collect()
    return pd.DataFrame(records)


def evaluate_predictions(predictions, frames, output, min_coverage):
    summaries, curves = [], []
    for (model, task), group in predictions.groupby(['model', 'task'], sort=False):
        gold = frames['sentence' if task == 'sentiment' else 'paragraph']
        group = group.merge(gold, on='sample_id', validate='one_to_one')
        group['gold'] = group['gold_' + task]
        group['valid'] = group.error.eq('') & group.prediction.ne(INVALID)
        if task in LABELS:
            group['valid'] &= group.prediction.isin(LABELS[task])
        subsets = {'all': group}
        if task == 'location':
            subsets['human_geothermal'] = group[group.gold_geothermal.eq('YES')]
        for subset, frame in subsets.items():
            calibration = frame[frame.split.eq('calibration')]
            test = frame[frame.split.eq('test')]
            if calibration.empty or test.empty:
                summaries.append(dict(model=model, task=task, subset=subset,
                                      status='insufficient calibration or test rows'))
                continue
            # Fixed class set per partition; absent classes score zero, never disappear
            # from macro F1 when thresholding. Locations are exact normalized strings.
            cal_labels = LABELS.get(task, sorted(set(calibration.gold) | set(calibration.loc[calibration.valid, 'prediction'])))
            test_labels = LABELS.get(task, sorted(set(test.gold) | set(test.loc[test.valid, 'prediction'])))
            curve = threshold_curve(calibration, cal_labels)
            threshold = choose_threshold(curve, min_coverage)
            curve = curve.assign(model=model, task=task, subset=subset)
            curves.append(curve)
            accepted = test.iloc[:0] if threshold is None else test[(test.confidence >= threshold) & test.valid]
            base = scores(test.gold, test.prediction, test_labels)
            selective = scores(accepted.gold, accepted.prediction, test_labels)
            summaries.append(dict(model=model, task=task, subset=subset,
                status='ok' if threshold is not None else 'no threshold meets minimum coverage',
                n_calibration=len(calibration), n_test=len(test), threshold=threshold,
                accuracy=base['accuracy'], macro_f1=base['macro_f1'],
                accepted_accuracy=selective['accuracy'], accepted_macro_f1=selective['macro_f1'],
                test_coverage=len(accepted) / len(test), n_accepted=len(accepted),
                errors=int(frame.error.ne('').sum()),
                runtime_seconds=float(frame.seconds.sum()),
                setup_seconds=float(frame.setup_seconds.sum()),
                mean_seconds_per_sample=float(frame.seconds.mean())))
    pd.DataFrame(summaries).to_csv(output / 'metrics.csv', index=False)
    if curves:
        pd.concat(curves, ignore_index=True).to_csv(output / 'thresholds_calibration.csv', index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--annotation-dir', type=Path, default=ROOT / 'annotation/final')
    ap.add_argument('--results-dir', type=Path, default=ROOT / 'annotation/final/evaluation')
    ap.add_argument('--models-file', type=Path, default=Path(__file__).with_name('models.json'))
    ap.add_argument('--models', nargs='+', help='Exact model IDs; default: all 16 unique models')
    ap.add_argument('--tasks', nargs='+', choices=['location', 'geothermal', 'sentiment'], default=['location', 'geothermal', 'sentiment'])
    ap.add_argument('--device', default='cpu', help='Hugging Face device: cpu, mps, cuda, cuda:0')
    ap.add_argument('--ollama-url', default='http://localhost:11434/api/generate')
    ap.add_argument('--min-coverage', type=float, default=0.8)
    ap.add_argument('--validate-only', action='store_true')
    args = ap.parse_args()
    if not 0 < args.min_coverage <= 1:
        ap.error('--min-coverage must be in (0, 1]')
    frames = load_annotations(args.annotation_dir)
    configs = json.loads(args.models_file.read_text())
    if args.models:
        unknown = set(args.models) - {c['model_id'] for c in configs}
        if unknown:
            ap.error(f'Unknown models: {sorted(unknown)}')
        configs = [c for c in configs if c['model_id'] in args.models]
    if args.validate_only:
        print('All manual labels and source integrity checks passed.')
        return
    args.results_dir.mkdir(parents=True, exist_ok=True)
    predictions = predict(frames, configs, args.results_dir, args)
    if predictions.empty:
        ap.error('No applicable model/task combinations')
    evaluate_predictions(predictions, frames, args.results_dir, args.min_coverage)
    print(f'Results: {args.results_dir / "metrics.csv"}')
    if predictions.error.ne('').any():
        raise SystemExit('Some model calls failed; inspect predictions.jsonl and rerun to retry errors.')


if __name__ == '__main__':
    main()
