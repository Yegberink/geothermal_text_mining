"""Independent, reproducible, prediction-blind annotation sampling."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(str(value).encode('utf-8')).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sample_frame(path, kind, size, seed):
    text_col = f'{kind}_text'
    # Read only source text/metadata. Never import predictions into annotation tables.
    allowed = {text_col, 'body_hash', 'source', 'document_title', 'publish_date',
               'region_name', 'language'}
    df = pd.read_csv(path, usecols=lambda c: c in allowed, keep_default_na=False,
                     dtype=str)
    if 'language' in df:
        df = df[df.language.str.lower().isin(['dutch', 'nl', 'nld', 'nederlands'])]
    df = df[df[text_col].str.strip().ne('')].copy()
    df['text_key'] = df[text_col].map(lambda s: digest(' '.join(s.split())))
    df = df.drop_duplicates('text_key').sort_values('text_key')
    pool_size = len(df)
    if pool_size < size:
        raise ValueError(f'{path}: need {size} distinct texts; found {pool_size}')
    df = df.sample(n=size, random_state=seed).reset_index(drop=True)
    def document_key(row):
        value = row.get('body_hash', '') or '||'.join(
            row.get(c, '') for c in ['source', 'document_title', 'publish_date'])
        if not value.strip('|'):
            raise ValueError('Source needs document metadata for grouped evaluation split')
        return digest(value)
    out = pd.DataFrame({'sample_id': kind + '_' + df.text_key,
                        'document_id': df.apply(document_key, axis=1),
                        text_col: df[text_col]})
    if kind == 'paragraph':
        out['region_name'] = df.get('region_name', pd.Series('Netherlands', index=df.index))
        out['gold_location'] = ''
        out['gold_geothermal'] = ''
    else:
        out['gold_sentiment'] = ''
    out['notes'] = ''
    return out, pool_size


def prepare(output_dir=ROOT / 'annotation/final', size=500, seed=42):
    output_dir = Path(output_dir)
    targets = [output_dir / n for n in ['paragraphs.csv', 'sentences.csv', 'manifest.json']]
    if any(p.exists() for p in targets):
        raise FileExistsError('Annotation files already exist; use a new directory to resample.')
    sources = {
        'paragraph': ROOT / 'output/dutch/workflow/newspapers_cleaned_paragraphs.csv',
        'sentence': ROOT / 'output/dutch/workflow/sentence_sentiment_llm.csv',
    }
    frames, manifest = {}, {'seed': seed, 'size': size, 'language': 'dutch', 'sources': {}}
    for kind, path in sources.items():
        frame, pool_size = sample_frame(path, kind, size, seed)
        frames[kind] = frame
        manifest['sources'][kind] = {
            'path': str(path.relative_to(ROOT)), 'sha256': file_digest(path),
            'distinct_pool_size': pool_size,
            'rows': frame.drop(columns=[c for c in frame if c.startswith('gold_') or c == 'notes']).to_dict('records'),
        }
    # Assign whole documents once, shared across both annotation sets.
    documents = sorted(set(pd.concat(list(frames.values())).document_id))
    ranked = sorted(documents, key=lambda d: digest(f'{seed}:{d}'))
    calibration = set(ranked[:max(1, round(len(ranked) * 0.3))])
    manifest['document_splits'] = {d: 'calibration' if d in calibration else 'test' for d in documents}
    output_dir.mkdir(parents=True, exist_ok=True)
    for kind, frame in frames.items():
        frame.to_csv(output_dir / f'{kind}s.csv', index=False)
    targets[2].write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    return {kind: {'sampled': len(frame), 'pool': manifest['sources'][kind]['distinct_pool_size']}
            for kind, frame in frames.items()}
