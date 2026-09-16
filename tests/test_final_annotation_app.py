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
        self.assertFalse(app.selectbox)
        self.assertEqual(app.subheader[0].value, 'Paragraph 2 of 2')
        self.assertEqual(app.text_input(key='edit_location').value, '')
        self.assertEqual(load_frame(self.directory, 'paragraph').iloc[0].gold_location, 'NONE')
        self.button(app, '← Previous').click().run()
        self.assertEqual(app.subheader[0].value, 'Paragraph 1 of 2')
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

    def test_correct_saved_paragraph_in_unfinished_queue(self):
        app = self.app()
        app.text_input(key='edit_location').set_value('NONE')
        app.radio(key='edit_geothermal').set_value('NO')
        self.button(app, 'Save & next').click().run()
        self.assertFalse(self.button(app, '← Previous').disabled)
        self.button(app, '← Previous').click().run()
        self.assertEqual(next(r for r in app.radio if r.label == 'Show').value, 'Unfinished')
        self.assertEqual(app.text_input(key='edit_location').value, 'NONE')
        app.text_input(key='edit_location').set_value('Delft')
        app.radio(key='edit_geothermal').set_value('YES')
        app.text_area(key='edit_notes').set_value('Corrected location and relevance')
        self.button(app, 'Save').click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(app.subheader[0].value, 'Paragraph 1 of 2')
        saved = load_frame(self.directory, 'paragraph').iloc[0]
        self.assertEqual(saved.gold_location, 'Delft')
        self.assertEqual(saved.gold_geothermal, 'YES')
        self.assertEqual(saved.notes, 'Corrected location and relevance')
        self.button(app, 'Save & next').click().run()
        self.assertEqual(app.subheader[0].value, 'Paragraph 2 of 2')
        self.assertEqual(app.text_input(key='edit_location').value, '')
        self.assertIsNone(app.radio(key='edit_geothermal').value)
        # A new session resumes unfinished work and can still go back to correct it.
        resumed = self.app()
        self.assertEqual(resumed.subheader[0].value, 'Paragraph 2 of 2')
        self.button(resumed, '← Previous').click().run()
        self.assertEqual(resumed.text_input(key='edit_location').value, 'Delft')
        self.assertEqual(resumed.radio(key='edit_geothermal').value, 'YES')

    def test_previous_after_save_returns_to_skipped_work(self):
        app = self.app()
        self.button(app, 'Skip / next →').click().run()
        app.text_input(key='edit_location').set_value('NONE')
        app.radio(key='edit_geothermal').set_value('NO')
        self.button(app, 'Save & next').click().run()
        self.assertEqual(app.subheader[0].value, 'Paragraph 1 of 2')
        self.assertFalse(self.button(app, '← Previous').disabled)
        self.button(app, '← Previous').click().run()
        self.assertEqual(app.subheader[0].value, 'Paragraph 2 of 2')
        self.assertEqual(app.radio(key='edit_geothermal').value, 'NO')
        app.radio(key='edit_geothermal').set_value('YES')
        self.button(app, 'Save & next').click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(app.subheader[0].value, 'Paragraph 1 of 2')
        self.assertEqual(load_frame(self.directory, 'paragraph').iloc[1].gold_geothermal, 'YES')

    def test_correct_sentences_after_finishing_task(self):
        app = self.app()
        next(r for r in app.radio if r.label == 'Annotation task').set_value('sentence').run()
        for sentiment in ['negative', 'positive']:
            app.radio(key='edit_sentiment').set_value(sentiment)
            self.button(app, 'Save & next').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any('All items in this task are saved' in s.value for s in app.success))
        self.assertEqual(app.subheader[0].value, 'Sentence 2 of 2')
        self.assertEqual(app.radio(key='edit_sentiment').value, 'positive')
        app.radio(key='edit_sentiment').set_value('neutral')
        self.button(app, 'Save').click().run()
        self.button(app, '← Previous').click().run()
        self.assertEqual(app.radio(key='edit_sentiment').value, 'negative')
        app.radio(key='edit_sentiment').set_value('not_geothermal')
        self.button(app, 'Save & next').click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(load_frame(self.directory, 'sentence').gold_sentiment.tolist(),
                         ['not_geothermal', 'neutral'])
        # Completed and All still support forward/back navigation through saved items.
        for queue in ['Completed', 'All']:
            next(r for r in app.radio if r.label == 'Show').set_value(queue).run()
            self.assertEqual(app.subheader[0].value, 'Sentence 1 of 2')
            self.button(app, 'Skip / next →').click().run()
            self.assertEqual(app.radio(key='edit_sentiment').value, 'neutral')
            self.button(app, '← Previous').click().run()
            self.assertEqual(app.radio(key='edit_sentiment').value, 'not_geothermal')
        resumed = self.app()
        next(r for r in resumed.radio if r.label == 'Annotation task').set_value('sentence').run()
        self.assertEqual(resumed.subheader[0].value, 'Sentence 2 of 2')
        self.button(resumed, '← Previous').click().run()
        self.assertEqual(resumed.radio(key='edit_sentiment').value, 'not_geothermal')

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
        self.button(app, 'Skip / next →').click().run()
        self.assertNotIn('Full original article with extra context.', [t.value for t in app.text])
        self.button(app, '← Previous').click().run()
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
