import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'annotation/final'))
from annotation_store import (ANNOTATORS, FIELDS, JOINT, assign_annotators, completed,
                              load_assignments, load_frame, save_annotation)
from split_work import make_packages, merge_package, write_package_app


class WorkSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.master = Path(self.temp.name) / 'master'
        self.master.mkdir()
        self.source = Path(self.temp.name) / 'source.csv'
        source = pd.DataFrame([{'body_hash': f'hash{i}', 'body': f'Full article {i}.',
                               'source': 'Newspaper', 'document_title': f'Title {i}',
                               'publish_date': '2020-01-01'} for i in range(10)])
        source.to_csv(self.source, index=False)
        manifest = {'sources': {}, 'document_splits': {}}
        for task, count in [('paragraph', 10), ('sentence', 8)]:
            frame = pd.DataFrame({'sample_id': [f'{task}{i}' for i in range(count)],
                'document_id': [hashlib.sha256(f'hash{i}'.encode()).hexdigest() for i in range(count)],
                f'{task}_text': [f'{task} text {i}. Eén idee voor België.' for i in range(count)]})
            if task == 'paragraph':
                frame['region_name'] = 'Netherlands'
            manifest['sources'][task] = {'path': str(self.source),
                'sha256': hashlib.sha256(self.source.read_bytes()).hexdigest(), 'rows': frame.to_dict('records')}
            for i, document_id in enumerate(frame.document_id):
                manifest['document_splits'][document_id] = 'test' if i % 2 else 'calibration'
            for field in FIELDS[task]:
                frame[field] = ''
            if task == 'paragraph':
                frame.loc[:2, 'gold_location'] = 'NONE'
                frame.loc[:2, 'gold_geothermal'] = 'NO'
                frame.loc[:2, 'notes'] = ['Joint note A', 'Joint note B', 'Joint note C']
            frame.to_csv(self.master / f'{task}s.csv', index=False)
        (self.master / 'manifest.json').write_text(json.dumps(manifest))
        self.original = {task: (self.master / f'{task}s.csv').read_bytes() for task in FIELDS}

    def package(self, annotator):
        return self.master / 'annotators' / annotator

    def owned_ids(self, directory, task, annotator):
        frame = load_frame(directory, task)
        assignments = load_assignments(directory, {task: frame})
        return [sid for sid, owner in assignments['tasks'][task].items() if owner == annotator]

    def annotate(self, directory, task, sid, annotator, location='Delft'):
        frame = load_frame(directory, task).set_index('sample_id')
        values = ({'gold_location': location, 'gold_geothermal': 'YES', 'notes': 'New label'}
                  if task == 'paragraph' else {'gold_sentiment': 'neutral', 'notes': 'New label'})
        save_annotation(directory, task, sid, values, frame.loc[sid, FIELDS[task]].to_dict(), annotator)

    def test_assignment_preserves_files_and_balances_only_unfinished_work(self):
        assignments = assign_annotators(self.master)
        self.assertEqual(assign_annotators(self.master), assignments)
        self.assertEqual(len(list((self.master / 'backups').iterdir())), 1)
        for task, expected in [('paragraph', [4, 3]), ('sentence', [4, 4])]:
            self.assertEqual((self.master / f'{task}s.csv').read_bytes(), self.original[task])
            self.assertEqual((self.master / assignments['backup'] / f'{task}s.csv').read_bytes(), self.original[task])
            owners = assignments['tasks'][task]
            self.assertEqual([list(owners.values()).count(a) for a in ANNOTATORS], expected)
        self.assertEqual({sid for sid, owner in assignments['tasks']['paragraph'].items() if owner == JOINT},
                         {'paragraph0', 'paragraph1', 'paragraph2'})

    def test_portable_packages_round_trip_and_evaluation(self):
        result = make_packages(self.master)
        self.assertEqual(result, {'Egberink': {'paragraph': 4, 'sentence': 4},
                                  'Dekker': {'paragraph': 3, 'sentence': 4}})
        for annotator in ANNOTATORS:
            package = self.package(annotator)
            with zipfile.ZipFile(package.with_suffix('.zip')) as archive:
                self.assertIn(f'{annotator}/documents.csv', archive.namelist())
            for task in FIELDS:
                for sid in self.owned_ids(package, task, annotator):
                    self.annotate(package, task, sid, annotator)
            # Updating an already-used copy must preserve all of its saved data.
            before = {name: (package / name).read_bytes() for name in
                      ['paragraphs.csv', 'sentences.csv', 'manifest.json', 'documents.csv']}
            write_package_app(package, annotator, result[annotator])
            for name, content in before.items():
                self.assertEqual((package / name).read_bytes(), content)
            self.assertIn('filelock', (package / 'requirements.txt').read_text(encoding='utf-8'))
            self.assertIn('.\\.venv\\Scripts\\python.exe', (package / 'README.md').read_text(encoding='utf-8'))
            merged = merge_package(self.master, package)
            self.assertEqual(merged, result[annotator])
            self.assertEqual(merge_package(self.master, package), {'paragraph': 0, 'sentence': 0})
        paragraphs = load_frame(self.master, 'paragraph')
        self.assertEqual(paragraphs.loc[:2, 'notes'].tolist(), ['Joint note A', 'Joint note B', 'Joint note C'])
        self.assertTrue(paragraphs.loc[:2, 'gold_geothermal'].eq('NO').all())
        self.assertTrue(completed(paragraphs, 'paragraph').all())
        self.assertTrue(completed(load_frame(self.master, 'sentence'), 'sentence').all())
        sys.path.insert(0, str(ROOT / 'scripts/model_evaluation'))
        import evaluate
        evaluated = evaluate.load_annotations(self.master)
        self.assertEqual({task: len(frame) for task, frame in evaluated.items()}, {'paragraph': 10, 'sentence': 8})

    def test_split_save_and_merge_when_fcntl_is_unavailable(self):
        # A fresh interpreter prevents an already-imported POSIX module masking the issue.
        script = '''
import sys
from pathlib import Path
sys.modules['fcntl'] = None
sys.path.insert(0, sys.argv[1])
from annotation_store import FIELDS, load_assignments, load_frame, save_annotation
from split_work import make_packages, merge_package
master = Path(sys.argv[2])
make_packages(master)
package = master / 'annotators' / 'Dekker'
frame = load_frame(package, 'paragraph')
owners = load_assignments(package, {'paragraph': frame})['tasks']['paragraph']
sid = next(sid for sid, owner in owners.items() if owner == 'Dekker')
expected = frame.set_index('sample_id').loc[sid, FIELDS['paragraph']].to_dict()
save_annotation(package, 'paragraph', sid,
                dict(gold_location='Delft', gold_geothermal='YES', notes=''), expected, 'Dekker')
assert merge_package(master, package) == {'paragraph': 1, 'sentence': 0}
'''
        result = subprocess.run([sys.executable, '-c', script, str(ROOT / 'annotation/final'), str(self.master)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_utf8_manifests_with_windows_default_text_encoding(self):
        read_text = Path.read_text

        def windows_read_text(path, encoding=None, errors=None):
            return read_text(path, encoding=encoding or 'cp1252', errors=errors)

        with patch.object(Path, 'read_text', windows_read_text):
            make_packages(self.master)
            package = self.package('Egberink')
            sid = self.owned_ids(package, 'paragraph', 'Egberink')[0]
            self.annotate(package, 'paragraph', sid, 'Egberink', location='België')
            self.assertEqual(merge_package(self.master, package), {'paragraph': 1, 'sentence': 0})
            row = load_frame(self.master, 'paragraph').set_index('sample_id').loc[sid]
            self.assertEqual(row.gold_location, 'België')
            self.assertIn('Eén idee voor België.', row.paragraph_text)

    def test_packages_open_independently_and_joint_work_is_read_only(self):
        make_packages(self.master)
        # The copies must continue to show full articles without the original source.
        self.source.rename(self.source.with_suffix('.unavailable'))
        for annotator in ANNOTATORS:
            package = self.package(annotator)
            with patch.dict(os.environ, {'FINAL_ANNOTATION_DIR': str(package)}):
                app = AppTest.from_file(str(package / 'app.py')).run()
                self.assertFalse(app.exception)
                self.assertFalse(any(r.label == 'Annotator' for r in app.radio))
                self.assertTrue(any(t.value.startswith('Full article') for t in app.text))
                app.text_input(key='edit_location').set_value('Delft')
                app.radio(key='edit_geothermal').set_value('YES')
                next(b for b in app.button if b.label == 'Save & next').click().run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertEqual(int(completed(load_frame(package, 'paragraph'), 'paragraph').sum()), 4)
                app.checkbox[0].check().run()
                self.assertEqual(app.text_input(key='edit_location').value, 'NONE')
                self.assertTrue(app.text_input(key='edit_location').disabled)
                self.assertTrue(next(b for b in app.button if b.label == 'Save').disabled)
            for task in FIELDS:
                self.assertEqual((self.master / f'{task}s.csv').read_bytes(), self.original[task])

    def test_backend_refuses_joint_and_other_annotator_edits(self):
        assign_annotators(self.master)
        for sid, annotator in [('paragraph0', 'Egberink'),
                               (self.owned_ids(self.master, 'paragraph', 'Dekker')[0], 'Egberink')]:
            with self.assertRaisesRegex(ValueError, 'not assigned to you'):
                self.annotate(self.master, 'paragraph', sid, annotator)
        self.assertEqual((self.master / 'paragraphs.csv').read_bytes(), self.original['paragraph'])

    def test_merge_refuses_changed_joint_labels_and_source_metadata(self):
        make_packages(self.master)
        package = self.package('Egberink')
        self.annotate(package, 'paragraph', self.owned_ids(package, 'paragraph', 'Egberink')[0], 'Egberink')
        paragraphs = pd.read_csv(package / 'paragraphs.csv', dtype=str, keep_default_na=False)
        paragraphs.loc[paragraphs.sample_id.eq('paragraph0'), 'gold_location'] = 'Changed joint label'
        paragraphs.to_csv(package / 'paragraphs.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'joint annotations were changed'):
            merge_package(self.master, package)
        paragraphs.loc[paragraphs.sample_id.eq('paragraph0'), 'gold_location'] = 'NONE'
        paragraphs.to_csv(package / 'paragraphs.csv', index=False)
        sentences = pd.read_csv(package / 'sentences.csv', dtype=str, keep_default_na=False)
        sentences.loc[0, 'sentence_text'] = 'Changed source'
        sentences.to_csv(package / 'sentences.csv', index=False)
        # Even modifying the package manifest cannot bypass the master metadata check.
        manifest = json.loads((package / 'manifest.json').read_text())
        manifest['sources']['sentence']['rows'][0]['sentence_text'] = 'Changed source'
        (package / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'source text or metadata changed'):
            merge_package(self.master, package)
        for task in FIELDS:
            self.assertEqual((self.master / f'{task}s.csv').read_bytes(), self.original[task])

    def test_merge_refuses_conflicting_master_labels(self):
        make_packages(self.master)
        package = self.package('Egberink')
        sid = self.owned_ids(package, 'paragraph', 'Egberink')[0]
        self.annotate(package, 'paragraph', sid, 'Egberink', location='Delft')
        self.annotate(self.master, 'paragraph', sid, 'Egberink', location='Utrecht')
        before = (self.master / 'paragraphs.csv').read_bytes()
        with self.assertRaisesRegex(ValueError, 'conflicting saved annotation'):
            merge_package(self.master, package)
        self.assertEqual((self.master / 'paragraphs.csv').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
