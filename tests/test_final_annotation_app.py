import json
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'annotation/final'))
from annotation_store import FIELDS, load_frame, save_annotation


class AnnotationAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        manifest = {'sources': {}}
        for task in FIELDS:
            df = pd.DataFrame({'sample_id': [task + '1', task + '2'],
                'document_id': ['doc1', 'doc2'], task + '_text': ['Een eerste tekst.', 'Een tweede tekst.']})
            manifest['sources'][task] = {'rows': df.to_dict('records')}
            for field in FIELDS[task]:
                df[field] = ''
            df.to_csv(self.directory / f'{task}s.csv', index=False)
        (self.directory / 'manifest.json').write_text(json.dumps(manifest))

    def app(self):
        env = patch.dict(os.environ, {'FINAL_ANNOTATION_DIR': str(self.directory)})
        env.start()
        self.addCleanup(env.stop)
        return AppTest.from_file(str(ROOT / 'annotation/final/app.py')).run()

    @staticmethod
    def button(app, label):
        return next(b for b in app.button if b.label == label)

    def test_save_resume_and_sentence_exclusion(self):
        app = self.app()
        self.assertFalse(app.exception)
        self.assertIsNone(app.radio(key='edit_geothermal').value)
        self.button(app, 'Save & next').click().run()
        self.assertTrue(app.error)
        app.text_input(key='edit_location').set_value('NONE')
        app.radio(key='edit_geothermal').set_value('NO')
        self.button(app, 'Save & next').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox[0].value, 'paragraph2')
        self.assertEqual(app.text_input(key='edit_location').value, '')
        self.assertEqual(load_frame(self.directory, 'paragraph').iloc[0].gold_location, 'NONE')
        next(r for r in app.radio if r.label == 'Show').set_value('All').run()
        app.selectbox[0].set_value('paragraph1').run()
        self.assertEqual(app.text_input(key='edit_location').value, 'NONE')
        self.assertEqual(app.radio(key='edit_geothermal').value, 'NO')
        next(r for r in app.radio if r.label == 'Annotation task').set_value('sentence').run()
        self.assertIsNone(app.radio(key='edit_sentiment').value)
        app.radio(key='edit_sentiment').set_value('not_geothermal')
        self.button(app, 'Save & next').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(load_frame(self.directory, 'sentence').iloc[0].gold_sentiment, 'not_geothermal')
        self.assertEqual(len(list((self.directory / 'backups').glob('*.csv'))), 2)
        raw = pd.read_csv(self.directory / 'paragraphs.csv')
        self.assertEqual(raw.document_id.tolist(), ['doc1', 'doc2'])

    def test_full_document_context_and_source_change(self):
        source = self.directory / 'source.csv'
        pd.DataFrame([{'body_hash': 'article_hash', 'body': 'Full original article with extra context.',
            'document_title': 'Article title', 'source': 'Newspaper', 'publish_date': '2020-01-02',
            'llm_prediction': 'SECRET MODEL OUTPUT'}]).to_csv(source, index=False)
        document_id = hashlib.sha256(b'article_hash').hexdigest()
        manifest_path = self.directory / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['sources']['paragraph'].update(path=str(source),
            sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        manifest['sources']['paragraph']['rows'][0]['document_id'] = document_id
        manifest_path.write_text(json.dumps(manifest))
        paragraphs = self.directory / 'paragraphs.csv'
        df = pd.read_csv(paragraphs, keep_default_na=False)
        df.loc[0, 'document_id'] = document_id
        df.to_csv(paragraphs, index=False)
        before = paragraphs.read_bytes()
        app = self.app()
        self.assertFalse(app.exception)
        self.assertTrue(any(e.label == 'Show full document' for e in app.expander))
        self.assertIn('Full original article with extra context.', [t.value for t in app.text])
        self.assertIn('Article title', [t.value for t in app.text])
        self.assertNotIn('SECRET MODEL OUTPUT', str(app))
        self.assertEqual(before, paragraphs.read_bytes())
        # Navigating must not show the previous item's article when no match exists.
        app.selectbox[0].set_value('paragraph2').run()
        self.assertNotIn('Full original article with extra context.', [t.value for t in app.text])
        app.selectbox[0].set_value('paragraph1').run()
        source.write_text(source.read_text() + '\n')
        app.run()
        self.assertTrue(any('source corpus has changed' in i.value for i in app.info))
        self.assertNotIn('Full original article with extra context.', [t.value for t in app.text])
        self.assertTrue(app.text_input)

    def test_conflict_and_other_row_merge(self):
        expected = {field: '' for field in FIELDS['paragraph']}
        values = dict(gold_location='Delft', gold_geothermal='YES', notes='Note')
        save_annotation(self.directory, 'paragraph', 'paragraph1', values, expected)
        with self.assertRaisesRegex(ValueError, 'another window'):
            save_annotation(self.directory, 'paragraph', 'paragraph1', values, expected)
        save_annotation(self.directory, 'paragraph', 'paragraph2', values, expected)
        self.assertTrue(load_frame(self.directory, 'paragraph').gold_location.eq('Delft').all())

    def test_source_integrity_and_no_prediction_display(self):
        path = self.directory / 'paragraphs.csv'
        df = pd.read_csv(path, keep_default_na=False)
        df['llm_prediction'] = 'SECRET MODEL OUTPUT'
        df.to_csv(path, index=False)
        app = self.app()
        self.assertFalse(app.exception)
        self.assertNotIn('SECRET MODEL OUTPUT', str(app))
        df.loc[0, 'paragraph_text'] = 'Changed source'
        df.to_csv(path, index=False)
        app.run()
        self.assertTrue(app.error)
        self.assertFalse(app.text_input)


if __name__ == '__main__':
    unittest.main()
