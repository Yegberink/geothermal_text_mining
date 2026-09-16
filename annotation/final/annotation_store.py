"""Atomic CSV annotation updates with source validation and edit conflict checks."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from filelock import FileLock

FIELDS = {
    'paragraph': ['gold_location', 'gold_geothermal', 'notes'],
    'sentence': ['gold_sentiment', 'notes'],
}
SENTIMENTS = ['negative', 'neutral', 'positive', 'not_geothermal']
ANNOTATORS = ['Egberink', 'Dekker']
JOINT = 'Together'


def load_assignments(directory, frames):
    assignments = json.loads((directory / 'manifest.json').read_text(encoding='utf-8')).get('annotation_assignments')
    if assignments is None:
        return None
    if assignments['annotators'] != ANNOTATORS:
        raise ValueError('Unexpected annotators in the assignment manifest')
    package_owner = assignments.get('package_annotator')
    if package_owner is not None and package_owner not in ANNOTATORS:
        raise ValueError('Unexpected annotator in the work package')
    for task, frame in frames.items():
        owners = assignments['tasks'][task]
        if set(owners) != set(frame.sample_id) or not set(owners.values()).issubset([*ANNOTATORS, JOINT]):
            raise ValueError(f'{task}: assignments do not match the annotation set')
        if package_owner and not set(owners.values()).issubset([package_owner, JOINT]):
            raise ValueError(f'{task}: package contains another annotator\'s work')
    return assignments


def assign_annotators(directory, seed=42):
    """Preserve completed work jointly and divide remaining items once, under the save lock."""
    directory = Path(directory)
    with FileLock(directory / '.annotation.lock', timeout=10):
        frames = {task: load_frame(directory, task) for task in FIELDS}
        existing = load_assignments(directory, frames)
        if existing is not None:
            return existing
        assignments = {'annotators': ANNOTATORS, 'seed': seed, 'tasks': {}}
        for task, frame in frames.items():
            done = completed(frame, task)
            owners = dict.fromkeys(frame.loc[done, 'sample_id'], JOINT)
            remaining = frame.loc[~done, 'sample_id'].sort_values().sample(frac=1, random_state=seed).tolist()
            midpoint = (len(remaining) + 1) // 2
            owners.update(dict.fromkeys(remaining[:midpoint], ANNOTATORS[0]))
            owners.update(dict.fromkeys(remaining[midpoint:], ANNOTATORS[1]))
            assignments['tasks'][task] = owners
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup = directory / 'backups' / f'before_assignment_{stamp}'
        backup.mkdir(parents=True)
        for filename in ['paragraphs.csv', 'sentences.csv', 'manifest.json']:
            shutil.copy2(directory / filename, backup / filename)
        assignments['backup'] = str(backup.relative_to(directory))
        manifest_path = directory / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['annotation_assignments'] = assignments
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=directory,
                                             suffix='.tmp', delete=False) as handle:
                temp_path = Path(handle.name)
                json.dump(manifest, handle, indent=2, ensure_ascii=False)
                handle.write('\n')
                handle.flush()
                os.fsync(handle.fileno())
            temp_path.replace(manifest_path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
        return assignments


def load_frame(directory: Path, task: str) -> pd.DataFrame:
    frame = pd.read_csv(directory / f'{task}s.csv', dtype=str, keep_default_na=False)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    expected = pd.DataFrame(manifest['sources'][task]['rows'])
    if not set(FIELDS[task]).issubset(frame.columns):
        raise ValueError(f'Missing annotation columns in {task}s.csv')
    if frame.sample_id.duplicated().any() or set(frame.sample_id) != set(expected.sample_id):
        raise ValueError(f'{task}s.csv: sample IDs differ from the original sample')
    actual = frame.set_index('sample_id').sort_index()
    original = expected.set_index('sample_id').sort_index()
    if not actual[original.columns].equals(original):
        raise ValueError(f'{task}s.csv: source text or metadata changed')
    if task == 'paragraph':
        frame['gold_location'] = frame.gold_location.str.strip()
    # Explicit allowlist: even unexpected columns can never reach the interface.
    return frame[['sample_id', 'document_id', f'{task}_text', *FIELDS[task]]].copy()


def completed(frame, task):
    if task == 'paragraph':
        return (frame.gold_location.str.strip().ne('')
                & ~frame.gold_location.str.strip().str.lower().isin(['nan', 'null', '__unsupported__'])
                & frame.gold_geothermal.str.strip().str.upper().isin(['YES', 'NO']))
    return frame.gold_sentiment.str.strip().str.lower().isin(SENTIMENTS)


def save_annotation(directory, task, sample_id, values, expected, annotator=None):
    """Merge one row into the latest CSV; refuse stale edits to that row."""
    if set(values) != set(FIELDS[task]):
        raise ValueError('Only annotation fields may be edited')
    values = {k: str(v).strip() for k, v in values.items()}
    if task == 'paragraph':
        values['gold_geothermal'] = values['gold_geothermal'].upper()
        if values['gold_location'].lower() == 'none':
            values['gold_location'] = 'NONE'
    else:
        values['gold_sentiment'] = values['gold_sentiment'].lower()
    if not completed(pd.DataFrame([values]), task).iloc[0]:
        raise ValueError('Complete each annotation field before saving. Use NONE for no clear location.')
    path = directory / f'{task}s.csv'
    with FileLock(directory / '.annotation.lock', timeout=10):
        current = load_frame(directory, task)
        assignments = load_assignments(directory, {task: current})
        if assignments is not None:
            owner = assignments['tasks'][task][sample_id]
            if annotator not in ANNOTATORS or owner != annotator:
                raise ValueError('This item is not assigned to you. Joint annotations are read-only.')
        current = current.set_index('sample_id')
        if current.loc[sample_id, FIELDS[task]].to_dict() != expected:
            raise ValueError('This item changed in another window. Reload the saved annotation before editing again.')
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        index = frame.index[frame.sample_id.eq(sample_id)][0]
        for field, value in values.items():
            frame.at[index, field] = value
        backup_dir = directory / 'backups'
        backup_dir.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        shutil.copy2(path, backup_dir / f'{task}s_{stamp}.csv')
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                             dir=directory, suffix='.tmp', delete=False) as handle:
                temp_path = Path(handle.name)
                frame.to_csv(handle, index=False)
                handle.flush()
                os.fsync(handle.fileno())
            temp_path.replace(path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()


def load_documents(source: Path, expected_sha256: str) -> dict:
    """Read original article text, matching the exact document IDs used by sampling."""
    checksum = hashlib.sha256()
    with source.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(block)
    if checksum.hexdigest() != expected_sha256:
        raise ValueError('The source corpus has changed since sampling. Restore the original source to view document context.')
    allowed = {'body', 'body_hash', 'source', 'document_title', 'publish_date'}
    documents = {}
    for chunk in pd.read_csv(source, usecols=lambda c: c in allowed, dtype=str,
                             keep_default_na=False, chunksize=1000):
        if 'body' not in chunk:
            raise ValueError('The source file does not contain full document text.')
        for row in chunk.drop_duplicates().to_dict('records'):
            identity = row.get('body_hash', '') or '||'.join(
                row.get(c, '') for c in ['source', 'document_title', 'publish_date'])
            document_id = hashlib.sha256(identity.encode('utf-8')).hexdigest()
            if document_id not in documents and row['body'].strip():
                documents[document_id] = {key: row.get(key, '') for key in
                                         ['body', 'body_hash', 'document_title', 'source', 'publish_date']}
    return documents
