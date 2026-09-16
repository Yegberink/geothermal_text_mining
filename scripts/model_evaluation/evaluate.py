#!/usr/bin/env python3
"""Evaluate manual labels, including exact location names and production workflow regions."""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import platform
import sys
import time
import unicodedata
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests

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


def choose_threshold(curve, min_coverage=0.0):
    eligible = curve[(curve.coverage >= min_coverage) & curve.accuracy.notna()]
    if eligible.empty:
        return None
    return float(eligible.sort_values(['accuracy', 'coverage', 'threshold'],
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['model_id'], use_fast=config.get('use_fast', True), **kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config['model_id'], use_safetensors=config.get('use_safetensors', False),
            **kwargs).to(device).eval()
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


def check_ollama_models(configs, url):
    models = [c['model_id'] for c in configs if c['backend'] == 'ollama']
    if not models:
        return
    tags_url = urljoin(url, 'tags')
    print(f'Checking {len(models)} Ollama models at {tags_url}...', flush=True)
    try:
        response = requests.get(tags_url, timeout=10)
        response.raise_for_status()
        installed = response.json()['models']
    except (requests.RequestException, ValueError, KeyError) as exc:
        raise ValueError(f'Could not list Ollama models at {tags_url}: {exc}. '
                         'Check that Ollama is running and --ollama-url is correct.') from exc

    def canonical(name):
        return name if ':' in name.rsplit('/', 1)[-1] else name + ':latest'

    available = {canonical(item['name']) for item in installed}
    missing = [model for model in models if canonical(model) not in available]
    if missing:
        commands = '\n'.join(f'  ollama pull {model}' for model in missing)
        raise ValueError(f'Missing Ollama models. Install them before evaluation:\n{commands}')


def ollama_predict(task, row, model, url, *, think=None):
    # Direct calls reuse the production prompts, system messages, generation options,
    # parsers and retry behavior. No workflow batch filters, fallbacks or caches.
    from core_workflow import is_geothermal, locations_ollama, sentiment_classification
    from helpers.country_scope import CountryScope
    options = {} if think is None else {'think': think}
    if task == 'sentiment':
        raw = sentiment_classification.call_ollama_sentiment(
            row.sentence_text, model, url, 'dutch', 'zero_shot', 120, **options)
        return raw['sentiment'], raw['confidence'], raw
    if task == 'geothermal':
        raw = is_geothermal.llm_is_geothermal(
            row.paragraph_text, 'Netherlands', url, model, 'dutch',
            is_geothermal.load_geothermal_lexicon(ROOT, 'dutch'), **options)
        return raw['is_geothermal'], raw['confidence'], raw
    raw = locations_ollama.llm_primary_location(
        row.paragraph_text, row.region_name or 'Netherlands', CountryScope(('Netherlands',)), url, model,
        **options)
    return normalize_location(raw['location']), raw['confidence'], raw


_OLLAMA_INFERENCE_SOURCE = inspect.getsource(ollama_predict)
_SPECIALIST_INFERENCE_SOURCE = "\n".join(inspect.getsource(obj) for obj in
    [HuggingFaceRunner, sentiment_class, collapse_probabilities])


def prediction_signature(config, task, args):
    """Scope cache identity to this model/task, independent of selected tasks/models."""
    files = []
    if config['backend'] == 'ollama':
        files = [ROOT / 'scripts/core_workflow' / {
            'location': 'locations_ollama.py', 'geothermal': 'is_geothermal.py',
            'sentiment': 'sentiment_classification.py'}[task]]
        if task != 'sentiment':
            files.append(ROOT / 'scripts/helpers/country_scope.py')
        if task == 'geothermal':
            files.extend([ROOT / 'data/vocab/dutch/geo_keywords.yaml',
                          ROOT / 'scripts/helpers/language_resources.py'])
        if task == 'location':
            files.extend([Path(__file__).with_name('location_fallback.py'),
                          ROOT / 'annotation/final/annotation_store.py',
                          ROOT / 'scripts/helpers/shape_resources.py'])
        inference = _OLLAMA_INFERENCE_SOURCE
    else:
        inference = _SPECIALIST_INFERENCE_SOURCE
    context = dict(config=config, task=task, device=args.device, url=args.ollama_url,
                   inference=inference, code={str(p): file_digest(p) for p in files})
    return hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()


def replace_task_predictions(path, records, task):
    """Replace only the selected models' derived task results, atomically."""
    models = {item['model'] for item in records}
    retained = []
    if path.exists():
        for line in path.read_text().splitlines(keepends=True):
            item = json.loads(line)
            if item['task'] != task or item['model'] not in models:
                retained.append(line)
    temporary = path.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(retained) + ''.join(
        json.dumps(item, ensure_ascii=False) + '\n' for item in records))
    temporary.replace(path)


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
    signatures = {(config['model_id'], task): prediction_signature(config, task, args)
                  for config in configs for task in args.tasks
                  if config['backend'] == 'ollama' or task == 'sentiment'}
    context['task_signatures'] = [{'model': model, 'task': task, 'signature': signature}
                                  for (model, task), signature in signatures.items()]
    compatibility_path = output / 'cache_compatibility.json'
    compatibility = json.loads(compatibility_path.read_text()) if compatibility_path.exists() else []
    aliases = {(item['model'], item['task'], item['old_signature']): item['signature']
               for item in compatibility}
    cache = {}
    for path in [prediction_path, output / 'location_paragraphs.jsonl']:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            item = json.loads(line)
            # Final locations are derived from the paragraph and document checkpoints.
            if path == prediction_path and item['task'] == 'location':
                continue
            selected = (item['model'], item['task'])
            if selected not in signatures:
                continue
            # Failed attempts are not reusable predictions, even after a loader fix.
            if item['error']:
                continue
            expected = signatures[selected]
            actual = aliases.get((*selected, item['signature']), item['signature'])
            if actual != expected:
                raise ValueError(f'{selected}: evaluation configuration/code changed; choose a new --results-dir')
            if not item['error']:
                cache[(*selected, item['sample_id'])] = item
    (output / 'run.json').write_text(json.dumps(context, indent=2) + '\n')
    records = []
    for config in configs:
        model, backend = config['model_id'], config['backend']
        runner, setup_error = None, ''
        tasks = args.tasks if backend == 'ollama' else [t for t in args.tasks if t == 'sentiment']
        for task in tasks:
            frame = frames['sentence' if task == 'sentiment' else 'paragraph']
            signature = signatures[(model, task)]
            task_path = output / 'location_paragraphs.jsonl' if task == 'location' else prediction_path
            task_start = len(records)
            cached = sum((model, task, sample_id) in cache for sample_id in frame.sample_id)
            print(f'{model}: {task} starting ({len(frame)} samples, {cached} cached)', flush=True)
            for index, row in enumerate(frame.itertuples(index=False), start=1):
                key = (model, task, row.sample_id)
                input_hash = hashlib.sha256((getattr(row, 'sentence_text', '') + getattr(row, 'paragraph_text', '') + getattr(row, 'region_name', '')).encode()).hexdigest()
                if task == 'location':
                    input_hash = hashlib.sha256(json.dumps([input_hash, row.document_id, row.body]).encode()).hexdigest()
                if key in cache:
                    item = cache[key]
                    if item['input_hash'] != input_hash:
                        raise ValueError('Cached input changed; choose a new results directory')
                    records.append(item)
                    continue
                setup_seconds = 0.0
                if backend == 'huggingface' and runner is None and not setup_error:
                    print(f'{model}: loading model on {args.device} (downloads on first use)...', flush=True)
                    start = time.perf_counter()
                    try:
                        runner = HuggingFaceRunner(config, args.device)
                    except Exception as exc:
                        setup_error = repr(exc)
                        print(f'{model}: SETUP FAILED — {setup_error}. '
                              f'{len(frame) - cached} uncached samples cannot be scored; '
                              'recording failures without further model calls.', flush=True)
                    setup_seconds = time.perf_counter() - start
                start = time.perf_counter()
                try:
                    if setup_error:
                        raise RuntimeError(setup_error)
                    prediction, confidence, raw = (runner(row.sentence_text) if backend == 'huggingface'
                                                  else ollama_predict(task, row, model, args.ollama_url,
                                                                      think=config.get('think')))
                    confidence = float(confidence)
                    if not np.isfinite(confidence) or not 0 <= confidence <= 1:
                        raise ValueError('Invalid confidence')
                    error = ''
                except Exception as exc:
                    prediction, confidence, raw, error = INVALID, 0.0, {}, repr(exc)
                item = dict(model=model, backend=backend, task=task, sample_id=row.sample_id,
                            prediction=prediction, confidence=confidence, raw=raw, error=error,
                            error_stage='setup' if setup_error else ('inference' if error else ''),
                            seconds=time.perf_counter() - start, setup_seconds=setup_seconds,
                            signature=signature, input_hash=input_hash)
                with task_path.open('a') as handle:
                    handle.write(json.dumps(item, ensure_ascii=False) + '\n')
                records.append(item)
                status = f'ERROR: {error}' if error else f'{prediction} (confidence {confidence:.2f})'
                if not setup_error:
                    print(f'{model}: {task} {index}/{len(frame)} — {status} '
                          f'[{item["seconds"]:.1f}s]', flush=True)
            if task == 'location':
                from location_fallback import apply_fallback
                final = apply_fallback(records[task_start:], frame, output, config, args.ollama_url, signature)
                records[task_start:] = final
                replace_task_predictions(prediction_path, final, 'location')
            outcome = 'failed to load' if setup_error else 'complete'
            print(f'{model}: {task} {outcome}', flush=True)
        del runner
        gc.collect()
    return pd.DataFrame(records)


def evaluate_predictions(predictions, frames, output, min_coverage, region_predictions=None, gold_regions=None,
                         *, summary_decimal='.'):
    if region_predictions is not None:
        from geography import UNRESOLVED
        frames = {key: frame.copy() for key, frame in frames.items()}
        frames['paragraph'] = frames['paragraph'].merge(gold_regions, on='sample_id', validate='one_to_one')
        frames['paragraph']['gold_location_region'] = frames['paragraph'].gold_region
        regional = predictions[predictions.task.eq('location')].merge(
            region_predictions, on=['sample_id', 'model'], validate='one_to_one')
        details = regional.merge(frames['paragraph'], on='sample_id', validate='many_to_one')
        details['exact_match'] = details.error.eq('') & details.prediction.eq(details.gold_location)
        details['region_evaluable'] = details.gold_region.ne(UNRESOLVED)
        details['region_match'] = (details.region_evaluable & details.error.eq('')
                                   & details.predicted_region.eq(details.gold_region)).astype('boolean')
        details.loc[~details.region_evaluable, 'region_match'] = pd.NA
        merge_report(details.drop(columns=['raw', 'body'], errors='ignore'),
                     output / 'location_comparison.csv', keys=['model', 'sample_id'])
        regional['prediction'] = regional.predicted_region
        regional['task'] = 'location_region'
        predictions = pd.concat([predictions, regional], ignore_index=True)
    summaries, curves = [], []
    for (model, task), group in predictions.groupby(['model', 'task'], sort=False):
        gold = frames['sentence' if task == 'sentiment' else 'paragraph']
        group = group.merge(gold, on='sample_id', validate='one_to_one')
        group['gold'] = group['gold_' + task]
        group['valid'] = group.error.eq('') & group.prediction.ne(INVALID)
        if task in LABELS:
            group['valid'] &= group.prediction.isin(LABELS[task])
        if task == 'location_region':
            group['valid'] &= group.prediction.ne(UNRESOLVED)
        subsets = {'all': group}
        if task in ['location', 'location_region']:
            subsets['human_geothermal'] = group[group.gold_geothermal.eq('YES')]
        for subset, frame in subsets.items():
            n_excluded = 0
            if task == 'location_region':
                n_excluded = int(frame.gold.eq(UNRESOLVED).sum())
                frame = frame[frame.gold.ne(UNRESOLVED)]
            if frame.empty:
                summaries.append(dict(model=model, task=task, subset=subset,
                                      n_samples=0, n_excluded_gold_regions=n_excluded,
                                      status='no eligible samples'))
                continue
            labels = LABELS.get(task, sorted(set(frame.gold) | set(frame.loc[frame.valid, 'prediction'])))
            curve = threshold_curve(frame, labels)
            threshold = choose_threshold(curve, min_coverage)
            curves.append(curve.assign(model=model, task=task, subset=subset))
            accepted = frame.iloc[:0] if threshold is None else frame[(frame.confidence >= threshold) & frame.valid]
            setup_failed = frame.error.ne('').all()
            base = ({'accuracy': None, 'macro_f1': None} if setup_failed
                    else scores(frame.gold, frame.prediction, labels))
            selective = scores(accepted.gold, accepted.prediction, labels)
            summaries.append(dict(model=model, task=task, subset=subset,
                status=('model setup failed' if setup_failed else
                        'ok' if threshold is not None else 'no threshold meets minimum coverage'),
                n_samples=len(frame), threshold=threshold,
                accuracy=base['accuracy'], macro_f1=base['macro_f1'],
                unfiltered_accuracy=base['accuracy'], unfiltered_macro_f1=base['macro_f1'],
                accepted_accuracy=selective['accuracy'], accepted_macro_f1=selective['macro_f1'],
                coverage=len(accepted) / len(frame), n_accepted=len(accepted),
                errors=int(frame.error.ne('').sum()),
                n_excluded_gold_regions=n_excluded,
                n_unresolved_regions=int(frame.prediction.eq(UNRESOLVED).sum()) if task == 'location_region' else 0,
                n_predicted_none=int(frame.prediction.eq('none').sum()) if task == 'location' else None,
                n_gold_none=int(frame.gold.eq('none').sum()) if task == 'location' else None,
                n_none_misses=int((frame.prediction.eq('none') & frame.gold.ne('none')).sum()) if task == 'location' else None,
                runtime_seconds=float(frame.seconds.sum()),
                setup_seconds=float(frame.setup_seconds.sum()),
                mean_seconds_per_sample=float(frame.seconds.mean())))
    migrate_detailed_metrics(output)
    merge_report(pd.DataFrame(summaries), output / 'metrics_detailed.csv')
    write_metrics_summary(output, decimal=summary_decimal)
    if curves:
        merge_report(pd.concat(curves, ignore_index=True), output / 'thresholds.csv')


def migrate_detailed_metrics(output):
    """Retain the old long-format report before replacing it with the summary."""
    old = output / 'metrics.csv'
    detailed = output / 'metrics_detailed.csv'
    if old.exists() and 'task' in pd.read_csv(old, nrows=0).columns:
        if detailed.exists():
            merge_report(pd.read_csv(old, keep_default_na=False), detailed)
        else:
            detailed.write_bytes(old.read_bytes())


def write_metrics_summary(output, *, decimal='.'):
    if decimal not in {'.', ','}:
        raise ValueError('Summary decimal separator must be . or ,')
    detailed = pd.read_csv(output / 'metrics_detailed.csv', keep_default_na=False)
    all_rows = detailed[detailed.subset.eq('all')]
    # Use a consistent location definition across models in this report.
    location_task = 'location_region' if all_rows.task.eq('location_region').any() else 'location'
    tasks = {'location': location_task, 'is_geothermal': 'geothermal', 'sentiment': 'sentiment'}
    columns = [f'{task} ({metric})' for metric in ['accuracy', 'f1', 'selective threshold', 'selective coverage'] for task in tasks]
    columns.append('average_runtime_seconds')
    records = []
    for model in detailed.model.drop_duplicates():
        row = dict(model=model, **{column: None for column in columns})
        for label, task in tasks.items():
            matched = all_rows[all_rows.model.eq(model) & all_rows.task.eq(task)]
            if matched.empty:
                continue
            result = matched.iloc[-1]
            # Historical setup failures were recorded as zero accuracy/F1.
            total = pd.to_numeric(result.get('n_samples', 0))
            errors = pd.to_numeric(result.get('errors', 0))
            if result.get('status') == 'model setup failed' or (total > 0 and errors >= total):
                continue
            row[f'{label} (accuracy)'] = result.get('unfiltered_accuracy', result.get('accuracy', ''))
            row[f'{label} (f1)'] = result.get('unfiltered_macro_f1', result.get('macro_f1', ''))
            row[f'{label} (selective threshold)'] = result.get('threshold', '')
            row[f'{label} (selective coverage)'] = result.get('coverage', '')
        timings = all_rows[all_rows.model.eq(model) & all_rows.task.isin(['location', 'geothermal', 'sentiment'])]
        if not timings.empty and 'n_samples' in timings:
            counts = pd.to_numeric(timings.n_samples, errors='coerce')
            errors = pd.to_numeric(timings.errors, errors='coerce')
            usable = counts.gt(0) & errors.lt(counts)
            if usable.any():
                row['average_runtime_seconds'] = pd.to_numeric(timings.loc[usable, 'runtime_seconds']).sum() / counts[usable].sum()
        records.append(row)
    summary = pd.DataFrame(records, columns=['model', *columns])
    for column in columns:
        try:
            summary[column] = pd.to_numeric(summary[column], errors='raise')
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Non-numeric value in summary column {column!r}') from exc
        values = summary[column]
        invalid = values.notna() & (~np.isfinite(values) | values.lt(0))
        if column != 'average_runtime_seconds':
            invalid |= values.gt(1)
        if invalid.any():
            models = ', '.join(summary.loc[invalid, 'model'])
            expected = 'finite nonnegative seconds' if column == 'average_runtime_seconds' else 'fractions in [0, 1]'
            raise ValueError(f'Invalid {column} for {models}: expected {expected}; report not written')
    # Keep fractions numeric and locale-explicit. Decimal-comma spreadsheets need
    # a semicolon delimiter so the decimal separator is not mistaken for a field.
    temporary = output / 'metrics.csv.tmp'
    summary.to_csv(temporary, index=False, decimal=decimal,
                   sep=';' if decimal == ',' else ',')
    temporary.replace(output / 'metrics.csv')


def merge_report(new, path, keys=None):
    if path.exists():
        previous = pd.read_csv(path, keep_default_na=False)
        keys = keys or ['model', 'task', 'subset']
        replaced = set(new[keys].itertuples(index=False, name=None))
        keep = [key not in replaced for key in previous[keys].itertuples(index=False, name=None)]
        if any(keep):
            new = pd.concat([previous.loc[keep], new], ignore_index=True)
    new.to_csv(path, index=False)


def rescore_cached(frames, configs, args):
    predictions = pd.read_json(args.results_dir / 'predictions.jsonl', lines=True)
    predictions = predictions[predictions.model.isin([c['model_id'] for c in configs])
                              & predictions.task.isin(args.tasks)]
    predictions = predictions.drop_duplicates(['model', 'task', 'sample_id'], keep='last')
    if predictions.empty:
        raise ValueError('No cached predictions for the requested models/tasks')
    for config in configs:
        for task in args.tasks:
            if config['backend'] != 'ollama' and task != 'sentiment':
                continue
            expected = frames['sentence' if task == 'sentiment' else 'paragraph']
            actual = predictions[predictions.model.eq(config['model_id']) & predictions.task.eq(task)]
            if set(actual.sample_id) != set(expected.sample_id):
                raise ValueError(f'{config["model_id"]}/{task}: cached sample set is incomplete')
    regions = gold_regions = None
    locations = predictions[predictions.task.eq('location')]
    if not locations.empty and args.location_matching == 'both':
        details = pd.read_csv(args.results_dir / 'location_comparison.csv', keep_default_na=False)
        checked = locations.merge(details, on=['model', 'sample_id'], suffixes=('', '_saved'),
                                  how='left', validate='one_to_one')
        if (checked.predicted_region.isna().any()
                or not checked.prediction.eq(checked.prediction_saved).all()
                or not checked.input_hash.eq(checked.input_hash_saved).all()):
            raise ValueError('Saved region matches do not match the current cached location predictions')
        regions = checked[['model', 'sample_id', 'predicted_region', 'geo_source', 'geo_match_type',
                           'geo_lat', 'geo_lon', 'nuts2_id', 'nuts2_name']]
        gold_regions = checked[['sample_id', 'gold_region', 'gold_region_status']].drop_duplicates()
        if gold_regions.sample_id.duplicated().any():
            raise ValueError('Saved gold regions disagree between models')
    evaluate_predictions(predictions, frames, args.results_dir, args.min_coverage, regions, gold_regions,
                         summary_decimal=args.summary_decimal)
    print(f'Rebuilt {args.results_dir / "metrics.csv"} from saved predictions; no inference or geocoding run.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--annotation-dir', type=Path, default=ROOT / 'annotation/final')
    ap.add_argument('--results-dir', type=Path, default=ROOT / 'annotation/final/evaluation')
    ap.add_argument('--models-file', type=Path, default=Path(__file__).with_name('models.json'))
    ap.add_argument('--models', nargs='+', help='Exact model IDs; default: all models in --models-file')
    ap.add_argument('--tasks', nargs='+', choices=['location', 'geothermal', 'sentiment'], default=['location', 'geothermal', 'sentiment'])
    ap.add_argument('--device', default='cpu', help='Hugging Face device: cpu, mps, cuda, cuda:0')
    ap.add_argument('--ollama-url', default='http://localhost:11434/api/generate')
    ap.add_argument('--min-coverage', type=float, default=0.0,
                    help='Optional minimum retained fraction; default: maximize accuracy without a coverage constraint')
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--rescore-only', action='store_true',
                    help='Rebuild metrics from saved predictions and region matches, without model or geocoder calls')
    ap.add_argument('--summary-decimal', choices=['.', ','], default='.',
                    help='Decimal separator for metrics.csv fractions (0–1); comma uses a semicolon delimiter for decimal-comma spreadsheets')
    ap.add_argument('--location-matching', choices=['both', 'exact'], default='both',
                    help='Score normalized exact names and workflow regions, or names only')
    ap.add_argument('--shapes-parquet', type=Path, default=ROOT / 'data/shapes.parquet')
    ap.add_argument('--geonames-dir', type=Path, default=ROOT / 'data/geonames')
    ap.add_argument('--geocoder-cache', type=Path, default=ROOT / 'cache/geocode_unmatched_online.jsonl')
    ap.add_argument('--geocoding-overrides', type=Path,
                    default=ROOT / 'data/vocab/dutch/location_geocoding_overrides.csv')
    ap.add_argument('--geocoder-country-codes', default='nl,be,bq,aw,cw,sx',
                    help='Same cache country-code scope as the Dutch workflow')
    args = ap.parse_args()
    if not 0 <= args.min_coverage <= 1:
        ap.error('--min-coverage must be in [0, 1]')
    print(f'Validating annotations in {args.annotation_dir}...', flush=True)
    frames = load_annotations(args.annotation_dir)
    configs = json.loads(args.models_file.read_text())
    if args.models:
        unknown = set(args.models) - {c['model_id'] for c in configs}
        if unknown:
            ap.error(f'Unknown models: {sorted(unknown)}')
        configs = [c for c in configs if c['model_id'] in args.models]
    if args.rescore_only:
        rescore_cached(frames, configs, args)
        return
    gold_regions = None
    if ('location' in args.tasks and args.location_matching == 'both'
            and any(c['backend'] == 'ollama' for c in configs)):
        sys.path.insert(0, str(ROOT / 'annotation/final'))
        from georeference_store import load_gold_regions
        gold_regions = load_gold_regions(args.annotation_dir, args.shapes_parquet)
    if 'location' in args.tasks and any(c['backend'] == 'ollama' for c in configs):
        from location_fallback import attach_document_context
        frames['paragraph'] = attach_document_context(frames['paragraph'], args.annotation_dir)
    if args.validate_only:
        print('All manual labels and source integrity checks passed.')
        return
    try:
        check_ollama_models(configs, args.ollama_url)
    except ValueError as exc:
        ap.exit(1, f'{exc}\n')
    print(f'Evaluating {len(configs)} models; results: {args.results_dir}', flush=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    predictions = predict(frames, configs, args.results_dir, args)
    if predictions.empty:
        ap.error('No applicable model/task combinations')
    region_predictions = None
    if gold_regions is not None and predictions.task.eq('location').any():
        print('Matching predicted locations to workflow regions...', flush=True)
        from geography import workflow_georeference
        region_predictions = workflow_georeference(
            predictions, frames['paragraph'], args.results_dir, args.shapes_parquet,
            args.geonames_dir, args.geocoder_cache, args.geocoding_overrides, args.geocoder_country_codes)
    evaluate_predictions(predictions, frames, args.results_dir, args.min_coverage,
                         region_predictions, gold_regions, summary_decimal=args.summary_decimal)
    print(f'Results: {args.results_dir / "metrics.csv"}')
    if predictions.error.ne('').any():
        raise SystemExit('Some model calls failed; inspect predictions.jsonl and rerun to retry errors.')


if __name__ == '__main__':
    main()
