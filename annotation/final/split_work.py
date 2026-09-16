"""Prepare portable annotation copies and merge returned labels into the master set."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from filelock import FileLock

from annotation_store import (ANNOTATORS, FIELDS, JOINT, assign_annotators, completed,
                              load_assignments, load_documents, load_frame)

APP_DIR = Path(__file__).resolve().parent


def write_package_app(package, annotator, counts):
    """Install app files and setup instructions without changing annotation data."""
    for filename in ['app.py', 'annotation_store.py']:
        shutil.copy2(APP_DIR / filename, package / filename)
    (package / 'requirements.txt').write_text(
        'streamlit>=1.44,<2\npandas>=2,<4\nfilelock>=3.13,<4\n', encoding='utf-8')
    (package / 'README.md').write_text(
        f'# Annotation copy for {annotator}\n\n'
        f'Your work: {counts["paragraph"]} paragraphs and {counts["sentence"]} sentences. '
        'Previously completed joint annotations are included for read-only reference.\n\n'
        'Extract the ZIP before starting. Install Python 3.11 or later, then open a terminal '
        'in this folder.\n\n'
        'On Windows (PowerShell or Command Prompt):\n\n'
        '```powershell\npy -3 -m venv .venv\n'
        '.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt\n'
        '.\\.venv\\Scripts\\python.exe -m streamlit run app.py\n```\n\n'
        'On macOS/Linux:\n\n'
        '```sh\npython3 -m venv .venv\n'
        '.venv/bin/python -m pip install -r requirements.txt\n'
        '.venv/bin/python -m streamlit run app.py\n```\n\n'
        'With an existing project environment, run `pixi run streamlit run /path/to/this/app.py` '
        'from the project root instead.\n\n'
        'Open the local URL printed by Streamlit. Your name is fixed in this copy. '
        'Use Save & next to continue, Previous to correct saved items, and Save to stay on an item. '
        'Full articles are included; the original workflow files are not needed.\n\n'
        'Labels are saved in paragraphs.csv and sentences.csv in this folder. '
        'To resume later, run the final command for your operating system again. Keep this working folder; '
        'extracting the original ZIP again creates a fresh copy.\n\n'
        'When finished, use Download saved annotations → Download work for merging in the app. '
        'Return that ZIP, which contains your current CSVs and manifest, including joint reference rows.\n',
        encoding='utf-8')


def make_packages(directory, output=None):
    directory = Path(directory).resolve()
    output = Path(output).resolve() if output else directory / 'annotators'
    for annotator in ANNOTATORS:
        if (output / annotator).exists() or (output / f'{annotator}.zip').exists():
            raise FileExistsError(f'{annotator} package already exists; choose a new output directory.')
    assign_annotators(directory)
    with FileLock(directory / '.annotation.lock', timeout=10):
        frames = {task: load_frame(directory, task) for task in FIELDS}
        assignments = load_assignments(directory, frames)
        if assignments.get('package_annotator'):
            raise ValueError('Create packages from the master annotation directory.')
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        origin = manifest.get('document_context', manifest['sources']['paragraph'])
        source = Path(origin['path'])
        if not source.is_absolute():
            source = directory / source if 'document_context' in manifest else APP_DIR.parents[1] / source
        documents = load_documents(source, origin['sha256'])
        output.mkdir(parents=True, exist_ok=True)
        result = {}
        for annotator in ANNOTATORS:
            destination = output / annotator
            # Build the entire copy before making it available under its final name.
            with tempfile.TemporaryDirectory(dir=output) as temp:
                package = Path(temp) / annotator
                package.mkdir()
                exported = copy.deepcopy(manifest)
                exported['annotation_assignments']['package_annotator'] = annotator
                selected = {}
                counts = {}
                for task in FIELDS:
                    owners = assignments['tasks'][task]
                    frame = pd.read_csv(directory / f'{task}s.csv', dtype=str, keep_default_na=False)
                    frame = frame[frame.sample_id.map(owners).isin([annotator, JOINT])].copy()
                    selected[task] = frame
                    frame.to_csv(package / f'{task}s.csv', index=False)
                    exported['sources'][task]['rows'] = [row for row in manifest['sources'][task]['rows']
                                                        if owners[row['sample_id']] in [annotator, JOINT]]
                    exported['annotation_assignments']['tasks'][task] = {
                        sid: owners[sid] for sid in frame.sample_id}
                    counts[task] = sum(owners[sid] == annotator for sid in frame.sample_id)
                document_ids = selected['paragraph'].document_id.unique()
                missing = set(document_ids) - documents.keys()
                if missing:
                    raise ValueError(f'Missing full document context for {len(missing)} sampled documents')
                context = package / 'documents.csv'
                pd.DataFrame([documents[doc_id] for doc_id in document_ids]).to_csv(context, index=False)
                exported['document_context'] = {
                    'path': 'documents.csv', 'sha256': hashlib.sha256(context.read_bytes()).hexdigest()}
                (package / 'manifest.json').write_text(
                    json.dumps(exported, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
                write_package_app(package, annotator, counts)
                package.rename(destination)
            shutil.make_archive(str(output / annotator), 'zip', root_dir=output, base_dir=annotator)
            result[annotator] = counts
        return result


def merge_package(directory, package):
    """Import assigned labels only; reject metadata changes and conflicting saved labels."""
    directory, package = Path(directory).resolve(), Path(package).resolve()
    with FileLock(directory / '.annotation.lock', timeout=10):
        frames = {task: load_frame(directory, task) for task in FIELDS}
        assignments = load_assignments(directory, frames)
        incoming_frames = {task: load_frame(package, task) for task in FIELDS}
        incoming_assignments = load_assignments(package, incoming_frames)
        if assignments is None or assignments.get('package_annotator'):
            raise ValueError('Merge into the master annotation directory after assigning work.')
        annotator = incoming_assignments.get('package_annotator') if incoming_assignments else None
        if annotator not in ANNOTATORS:
            raise ValueError('The returned folder must be an annotator work package.')
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        updates, counts = {}, {}
        for task, incoming in incoming_frames.items():
            owners = assignments['tasks'][task]
            expected_ids = {sid for sid, owner in owners.items() if owner in [annotator, JOINT]}
            if set(incoming.sample_id) != expected_ids:
                raise ValueError(f'{task}: returned sample IDs differ from {annotator}\'s assignment')
            raw = pd.read_csv(package / f'{task}s.csv', dtype=str, keep_default_na=False).set_index('sample_id')
            original = pd.DataFrame(manifest['sources'][task]['rows']).set_index('sample_id')
            original = original.loc[sorted(expected_ids)]
            if not raw.loc[original.index, original.columns].equals(original):
                raise ValueError(f'{task}: returned source text or metadata changed')
            baseline = load_frame(directory / assignments['backup'], task).set_index('sample_id')
            current = pd.read_csv(directory / f'{task}s.csv', dtype=str, keep_default_na=False).set_index('sample_id')
            needs_cleanup = task == 'paragraph' and current.gold_location.ne(current.gold_location.str.strip()).any()
            if task == 'paragraph':
                current['gold_location'] = current.gold_location.str.strip()
            incoming = incoming.set_index('sample_id')
            counts[task] = 0
            for sid, row in incoming.iterrows():
                values = row[FIELDS[task]]
                saved = current.loc[sid, FIELDS[task]]
                if owners[sid] == JOINT:
                    if not values.equals(saved):
                        raise ValueError(f'{task}: joint annotations were changed in the returned package')
                    continue
                initial = baseline.loc[sid, FIELDS[task]]
                if values.equals(initial) or values.equals(saved):
                    continue
                if not completed(pd.DataFrame([values]), task).iloc[0]:
                    raise ValueError(f'{task}: incomplete returned annotation for {sid}')
                if not saved.equals(initial):
                    raise ValueError(f'{task}: conflicting saved annotation for {sid}; no files were merged')
                current.loc[sid, FIELDS[task]] = values
                counts[task] += 1
            if counts[task] or needs_cleanup:
                updates[task] = current.reset_index()
        if not updates:
            return counts
        # Validate both tasks before backing up and replacing either master CSV.
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup = directory / 'backups' / f'before_merge_{annotator}_{stamp}'
        backup.mkdir(parents=True)
        for filename in ['paragraphs.csv', 'sentences.csv', 'manifest.json']:
            shutil.copy2(directory / filename, backup / filename)
        with tempfile.TemporaryDirectory(dir=directory) as temp:
            for task, frame in updates.items():
                path = Path(temp) / f'{task}s.csv'
                with path.open('w', encoding='utf-8', newline='') as handle:
                    frame.to_csv(handle, index=False)
                    handle.flush()
                    os.fsync(handle.fileno())
            for task in updates:
                (Path(temp) / f'{task}s.csv').replace(directory / f'{task}s.csv')
        return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation-dir', type=Path, default=APP_DIR)
    subparsers = parser.add_subparsers(dest='command', required=True)
    prepare_parser = subparsers.add_parser('prepare', help='Create separate folders and ZIPs for each annotator')
    prepare_parser.add_argument('--output', type=Path)
    merge_parser = subparsers.add_parser('merge', help='Merge labels from a returned annotator folder')
    merge_parser.add_argument('package', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(make_packages(args.annotation_dir, args.output), indent=2))
    else:
        print(json.dumps(merge_package(args.annotation_dir, args.package), indent=2))
