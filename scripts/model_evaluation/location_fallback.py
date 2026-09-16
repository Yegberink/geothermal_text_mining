"""Apply production location fallback to the fixed, document-split evaluation sample."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def attach_document_context(frame, directory):
    sys.path.insert(0, str(ROOT / 'annotation/final'))
    from annotation_store import load_documents
    manifest = json.loads((directory / 'manifest.json').read_text())
    source = manifest['sources']['paragraph']
    documents = load_documents(ROOT / source['path'], source['sha256'])
    missing = set(frame.document_id) - documents.keys()
    if missing:
        raise ValueError(f'Missing full article text for {len(missing)} sampled documents')
    result = frame.copy()
    result['body'] = result.document_id.map(lambda key: documents[key]['body'])
    # Use the sampling identity, not titles or row order, to keep documents separate.
    result['body_hash'] = result.document_id
    if result.groupby('document_id').split.nunique().gt(1).any():
        raise ValueError('A document cannot occur in both calibration and test')
    return result


def apply_fallback(records, frame, output, config, url, signature):
    from core_workflow import locations_ollama as workflow
    from helpers.country_scope import CountryScope

    required = {'body', 'document_id'}
    if not required.issubset(frame.columns) or frame.body.str.strip().eq('').any():
        raise ValueError('Location evaluation requires validated full article context')
    scope = CountryScope(('Netherlands',))
    source = frame.set_index('sample_id')
    rows = []
    for item in records:
        row = source.loc[item['sample_id']]
        raw = item['raw']
        rows.append(dict(sample_id=item['sample_id'], body_hash=row.document_id,
                         body=row.body, paragraph_text=row.paragraph_text,
                         region_name=row.region_name or scope.label,
                         llm_location=raw.get('location', 'NONE'),
                         llm_granularity=raw.get('granularity', 'none'),
                         llm_confidence=item['confidence'],
                         llm_reasoning_short=raw.get('reasoning_short', ''),
                         llm_status='error' if item['error'] else 'ok'))
    work = pd.DataFrame(rows)
    cache_path = output / 'location_documents.jsonl'
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            item = json.loads(line)
            if item['model'] == config['model_id']:
                if item['signature'] != signature:
                    raise ValueError('Location document cache changed; reset location results')
                cache[item['key']] = item
    used = {}

    def resolve(text, region, document_id):
        key = hashlib.sha256(json.dumps([document_id, text, region]).encode()).hexdigest()
        item = cache.get(key)
        if item is None or item['error']:
            started = time.perf_counter()
            try:
                raw = workflow.llm_document_primary_location(
                    text, region, scope, url, config['model_id'], think=config.get('think'))
                error = ''
            except Exception as exc:
                raw, error = {}, repr(exc)
            item = dict(key=key, model=config['model_id'], signature=signature,
                        raw=raw, error=error, seconds=time.perf_counter() - started)
            with cache_path.open('a') as handle:
                handle.write(json.dumps(item, ensure_ascii=False) + '\n')
            cache[key] = item
            print(f'{config["model_id"]}: document fallback {document_id[:12]} '
                  f'[{item["seconds"]:.1f}s] {error}', flush=True)
        used[document_id] = item
        # The workflow resolver raises on failure. Continue other documents here,
        # then explicitly mark affected rows failed instead of scoring a fallback.
        return item['raw'] if not item['error'] else {'location': 'NONE', 'granularity': 'none'}

    final = workflow.apply_document_location_fallback(
        work, 'paragraph_text', 'region_name', scope, resolve)
    called_sources = {'document_llm', 'country_fallback'}
    affected = final.llm_location_source.isin(called_sources)
    counts = final[affected].groupby('body_hash').size().to_dict()
    results = []
    for item, (_, row) in zip(records, final.iterrows()):
        result = dict(item)
        raw = workflow.location_payload_from_row(row)
        raw.update({key: value for key, value in row.items() if key.startswith('llm_')})
        raw['paragraph_raw'] = item['raw']
        result['raw'] = raw
        result['paragraph_prediction'] = item['prediction']
        result['location_source'] = row.llm_location_source
        result['document_seconds'] = 0.0
        if row.llm_location_source in called_sources and row.body_hash in used:
            document = used[row.body_hash]
            result['document_seconds'] = document['seconds'] / counts[row.body_hash]
            result['error'] = result['error'] or document['error']
        result['seconds'] += result['document_seconds']
        if result['error']:
            result['prediction'], result['confidence'] = '__unsupported__', 0.0
        else:
            from evaluate import normalize_location
            result['prediction'] = normalize_location(raw['location'])
            result['confidence'] = float(raw['confidence'])
        results.append(result)
    return results
