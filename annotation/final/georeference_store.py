"""Manual map judgments, stored separately from text labels and production geocoding."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/model_evaluation'))
from geography import load_regions, point_region, UNRESOLVED
from annotation_store import load_frame

COLUMNS = ['sample_id', 'gold_location', 'source_hash', 'status', 'latitude', 'longitude',
           'country_id', 'region_id', 'region_name', 'notes', 'shapes_sha256', 'updated_at']
FILENAME = 'georeferences.csv'


def source_hash(row):
    values = [row['sample_id'], row['document_id'], row['paragraph_text'], row['gold_location'].strip()]
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()


def read_references(directory):
    path = Path(directory) / FILENAME
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if set(frame.columns) != set(COLUMNS) or frame.sample_id.duplicated().any():
        raise ValueError('Invalid manual georeference table')
    return frame


def current_reference(references, row):
    saved = references[references.sample_id.eq(row['sample_id'])]
    return None if saved.empty else saved.iloc[0].to_dict()


def location_key(name):
    """Group spelling-identical names, ignoring case and whitespace, without alias inference."""
    return ' '.join(unicodedata.normalize('NFKC', str(name)).casefold().split())


def group_snapshot(members, references):
    return {
        'sources': {row.sample_id: source_hash(row) for _, row in members.iterrows()},
        'references': references[references.sample_id.isin(members.sample_id)].sort_values(
            'sample_id').to_dict('records'),
    }


def location_groups(paragraphs, references, digest):
    """Reuse existing per-paragraph work without rewriting it; conflicting regions need review."""
    groups = {}
    for key, members in paragraphs.groupby(paragraphs.gold_location.map(location_key), sort=False):
        if key in ['', 'none']:
            continue
        snapshot = group_snapshot(members, references)
        saved = snapshot['references']
        valid = [record for record in saved if record['source_hash'] == snapshot['sources'][record['sample_id']]
                 and location_key(record['gold_location']) == key and record['shapes_sha256'] == digest]
        decisions = {(record['status'], record['country_id'], record['region_id']) for record in valid}
        # Different points inside the same region agree for region evaluation. Keep every original record.
        conflict = len(decisions) > 1
        groups[key] = {'members': members, 'expected': snapshot, 'conflict': conflict,
                       'complete': bool(valid) and not conflict,
                       'valid': valid,
                       'saved': max(valid or saved, key=lambda record: record['updated_at']) if saved else None}
    return groups


def save_reference(directory, row, status, latitude, longitude, notes, expected, shapes, *, unique_name=False):
    directory, shapes = Path(directory), Path(shapes)
    if not row['gold_location'].strip():
        raise ValueError('Annotate the primary location first')
    record = dict.fromkeys(COLUMNS, '')
    record.update(sample_id=row['sample_id'], gold_location=row['gold_location'].strip(),
                  source_hash=source_hash(row), status=status, notes=notes.strip(),
                  shapes_sha256=hashlib.sha256(shapes.read_bytes()).hexdigest(),
                  updated_at=datetime.now(timezone.utc).isoformat())
    is_none = record['gold_location'].casefold() == 'none'
    if is_none != (status == 'none'):
        raise ValueError('No location applies only to a manual location label of NONE')
    if status == 'point':
        record.update(point_region(latitude, longitude, load_regions(shapes)),
                      latitude=str(float(latitude)), longitude=str(float(longitude)))
    elif status == 'country':
        record.update(country_id='NLD', region_name='Netherlands')
    elif status == 'unresolved':
        if not notes.strip():
            raise ValueError('Explain why the location cannot be assigned a Dutch region')
    elif status != 'none':
        raise ValueError('Choose a georeference status')
    with FileLock(directory / '.annotation.lock', timeout=10):
        paragraphs = load_frame(directory, 'paragraph')
        latest = paragraphs.set_index('sample_id', drop=False).loc[row['sample_id']]
        if source_hash(latest) != source_hash(row):
            raise ValueError('The location label changed. Reload before saving.')
        frame = read_references(directory)
        members = paragraphs[paragraphs.gold_location.map(location_key).eq(location_key(row['gold_location']))]
        actual = group_snapshot(members, frame) if unique_name else current_reference(frame, row)
        if actual != expected:
            raise ValueError('This georeference changed in another window. Reload before saving.')
        targets = members if unique_name else paragraphs[paragraphs.sample_id.eq(row['sample_id'])]
        records = [dict(record, sample_id=target.sample_id, gold_location=target.gold_location.strip(),
                        source_hash=source_hash(target)) for _, target in targets.iterrows()]
        frame = frame[~frame.sample_id.isin(targets.sample_id)]
        frame = pd.concat([frame, pd.DataFrame(records)], ignore_index=True)
        path = directory / FILENAME
        if path.exists():
            backup = directory / 'backups'
            backup.mkdir(exist_ok=True)
            shutil.copy2(path, backup / f'georeferences_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.csv')
        with tempfile.TemporaryDirectory(dir=directory) as temp:
            staged = Path(temp) / FILENAME
            with staged.open('w', encoding='utf-8', newline='') as handle:
                frame.to_csv(handle, index=False)
                handle.flush()
                os.fsync(handle.fileno())
            staged.replace(path)


def load_gold_regions(directory, shapes):
    """Expand unique-name judgments back to samples, preserving the original evaluation weighting."""
    paragraphs = load_frame(Path(directory), 'paragraph')
    references = read_references(directory)
    if not set(references.sample_id).issubset(paragraphs.sample_id):
        raise ValueError('Map judgments contain unknown sample IDs')
    digest = hashlib.sha256(Path(shapes).read_bytes()).hexdigest()
    regions = load_regions(shapes)
    groups = location_groups(paragraphs, references, digest)
    for key, group in groups.items():
        if group['conflict']:
            raise ValueError(f'Conflicting saved regions for {key}; choose one judgment in georeference_app.py')
        if not group['complete']:
            if group['saved']:
                raise ValueError(f'Stale map judgment or region boundaries changed for {key}; review it again')
            raise ValueError('Complete map georeferencing for all named locations in georeference_app.py')
    rows = []
    for _, row in paragraphs.iterrows():
        if row.gold_location.casefold() == 'none':
            rows.append({'sample_id': row.sample_id, 'gold_region': 'none', 'gold_region_status': 'none'})
            continue
        group = groups[location_key(row.gold_location)]
        saved = group['saved']
        status = saved['status']
        if (row.gold_location.casefold() == 'none') != (status == 'none'):
            raise ValueError('Manual NONE location and map status disagree')
        if status == 'point':
            assigned = point_region(saved['latitude'], saved['longitude'], regions)
            if any(saved[key] != value for key, value in assigned.items()):
                raise ValueError('Saved point and region disagree; review the map judgment')
            key = 'nuts2:' + assigned['region_id']
        elif status == 'country' and saved['country_id'] == 'NLD':
            key = 'country:NLD'
        elif status == 'none':
            key = 'none'
        elif status == 'unresolved' and saved['notes'].strip():
            key = UNRESOLVED
        else:
            raise ValueError('Invalid manual georeference status')
        rows.append({'sample_id': row.sample_id, 'gold_region': key,
                     'gold_region_status': status})
    return pd.DataFrame(rows)
