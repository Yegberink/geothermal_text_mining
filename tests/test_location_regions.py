import json
import hashlib
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from shapely.geometry import box
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/model_evaluation'))
sys.path.insert(0, str(ROOT / 'annotation/final'))
import evaluate
import geography
import georeference_store as store
from georeference_search import load_places, search_places
from annotation_store import load_frame


class RegionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.shapes = self.path / 'shapes.parquet'
        frame = pd.DataFrame({'country_id': ['NLD', 'NLD'], 'parent_id': ['NL33', 'NL31'],
            'parent_name': ['Zuid-Holland', 'Utrecht'], 'shape_class': ['land'] * 2,
            'parent': ['nuts'] * 2, 'parent_subtype': ['2'] * 2,
            'geometry': [box(4, 51, 5, 53).wkb, box(5, 51, 6, 53).wkb]})
        frame.to_parquet(self.shapes)
        source = pd.DataFrame({'sample_id': ['p1', 'p2'], 'document_id': ['d1', 'd2'],
                               'paragraph_text': ['Text one', 'Text two']})
        (self.path / 'manifest.json').write_text(json.dumps({'sources': {
            'paragraph': {'rows': source.to_dict('records')}}}), encoding='utf-8')
        source['gold_location'] = ['  Delft \t', 'NONE']
        source['gold_geothermal'] = ['YES', 'NO']
        source['notes'] = ''
        source.to_csv(self.path / 'paragraphs.csv', index=False)
        self.paragraphs = load_frame(self.path, 'paragraph')
        self.gazetteer = self.path / 'NL.zip'
        with zipfile.ZipFile(self.gazetteer, 'w') as bundle:
            bundle.writestr('NL.txt', '\n'.join('\t'.join([
                str(i), 'Delft', 'Delft', 'Delft City', '52', str(lon), 'P', 'PPL',
                'NL', '', '', '', '', '', '1000', '', '', '', '2026-01-01'])
                for i, lon in enumerate([4.5, 5.5], start=1)) + '\n')

    def save(self, index, status, lat='', lon='', notes='', expected=None):
        store.save_reference(self.path, self.paragraphs.iloc[index], status, lat, lon,
                             notes, expected, self.shapes)

    def test_manual_save_resume_whitespace_conflicts_and_stale_labels(self):
        self.assertEqual(self.paragraphs.iloc[0].gold_location, 'Delft')
        with self.assertRaisesRegex(ValueError, 'all named locations'):
            store.load_gold_regions(self.path, self.shapes)
        self.save(0, 'point', 52, 4.5)
        self.assertEqual(store.load_gold_regions(self.path, self.shapes).gold_region.tolist(),
                         ['nuts2:NL33', 'none'])
        # Existing NONE map records remain compatible, but are no longer required.
        self.save(1, 'none')
        gold = store.load_gold_regions(self.path, self.shapes)
        self.assertEqual(gold.gold_region.tolist(), ['nuts2:NL33', 'none'])
        with self.assertRaisesRegex(ValueError, 'another window'):
            self.save(0, 'point', 52, 5.5)
        with self.assertRaisesRegex(ValueError, 'inside one Dutch region'):
            self.save(0, 'point', 52, 5)
        with self.assertRaises(ValueError):
            self.save(0, 'point', float('nan'), 4)
        with self.assertRaisesRegex(ValueError, 'only'):
            self.save(0, 'none')
        raw = pd.read_csv(self.path / 'paragraphs.csv', keep_default_na=False)
        raw.loc[0, 'gold_location'] = 'Utrecht'
        raw.to_csv(self.path / 'paragraphs.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'Stale'):
            store.load_gold_regions(self.path, self.shapes)
        with self.assertRaisesRegex(ValueError, 'label changed'):
            self.save(0, 'point', 52, 5.5)
        raw.loc[0, 'gold_location'] = 'NONE'
        raw.to_csv(self.path / 'paragraphs.csv', index=False)
        self.assertEqual(store.load_gold_regions(self.path, self.shapes).gold_region.tolist(), ['none', 'none'])

    def test_country_unresolved_and_shapes_changes(self):
        self.save(0, 'country')
        self.save(1, 'none')
        self.assertEqual(store.load_gold_regions(self.path, self.shapes).iloc[0].gold_region, 'country:NLD')
        expected = store.read_references(self.path).iloc[0].to_dict()
        with self.assertRaisesRegex(ValueError, 'Explain'):
            self.save(0, 'unresolved', expected=expected)
        self.save(0, 'unresolved', notes='Outside Dutch scope', expected=expected)
        self.assertEqual(store.load_gold_regions(self.path, self.shapes).iloc[0].gold_region, geography.UNRESOLVED)
        changed = pd.read_parquet(self.shapes)
        changed.loc[0, 'parent_name'] = 'Changed'
        changed.to_parquet(self.shapes)
        with self.assertRaisesRegex(ValueError, 'boundaries changed'):
            store.load_gold_regions(self.path, self.shapes)

    def test_map_app_default_search_fallback_and_resume(self):
        with patch.dict(os.environ, {'FINAL_ANNOTATION_DIR': str(self.path),
                                    'FINAL_ANNOTATION_SHAPES': str(self.shapes),
                                    'FINAL_ANNOTATION_GEONAMES': str(self.gazetteer)}):
            app = AppTest.from_file(str(ROOT / 'annotation/final/georeference_app.py')).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get('component_instance')), 1)
            self.assertFalse(app.radio)
            self.assertEqual(app.text_input(key='geo_query').value, 'Delft')
            next(b for b in app.button if b.label == 'Search').click().run()
            self.assertEqual(len(app.selectbox(key='geo_search_result').options), 2)
            app.selectbox(key='geo_search_result').set_value(0).run()
            self.assertFalse(app.exception)
            self.assertEqual(float(app.text_input(key='geo_lat').value), 52.)
            self.assertEqual(float(app.text_input(key='geo_lon').value), 4.5)
            args = json.loads(app.get('component_instance')[0].proto.json_args)
            self.assertEqual((args['latitude'], args['longitude'], args['focus']), (52., 4.5, 1))
            next(b for b in app.button if b.label == 'Save & next').click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            # NONE is excluded from the queue and requires no new map judgment.
            self.assertEqual(len(app.selectbox(key='geo_sample').options), 1)
            self.assertEqual(store.read_references(self.path).sample_id.tolist(), ['p1'])
            resumed = AppTest.from_file(str(ROOT / 'annotation/final/georeference_app.py')).run()
            self.assertFalse(resumed.exception)
            self.assertEqual(resumed.text_input(key='geo_lat').value, '52.0')
            self.assertEqual(store.load_gold_regions(self.path, self.shapes).gold_region.tolist(), ['nuts2:NL33', 'none'])
            resumed.checkbox(key='geo_use_fallback').check().run()
            self.assertEqual(resumed.radio(key='geo_fallback').options,
                             ['Netherlands as a whole', 'Unresolved / outside Dutch scope'])
            resumed.radio(key='geo_fallback').set_value('country').run()
            next(b for b in resumed.button if b.label == 'Save').click().run()
            self.assertFalse(resumed.exception)
            self.assertFalse(resumed.error)
            self.assertEqual(store.load_gold_regions(self.path, self.shapes).iloc[0].gold_region, 'country:NLD')
            self.assertEqual(len(resumed.get('component_instance')), 1)

    def test_search_aliases_partial_names_and_homonyms(self):
        places = load_places(self.gazetteer)
        for query in ['  DÉLFT CITY  ', 'delf']:
            matches = search_places(places, query)
            self.assertEqual(len(matches), 2)
            self.assertEqual([match['longitude'] for match in matches], [4.5, 5.5])
        self.assertEqual(search_places(places, 'unknown'), [])
        self.assertEqual(search_places(places, '  '), [])

    def duplicate_locations(self):
        frame = pd.read_csv(self.path / 'paragraphs.csv', dtype=str, keep_default_na=False)
        frame.loc[1, 'gold_location'] = '  DELFT  '
        frame.to_csv(self.path / 'paragraphs.csv', index=False)
        self.paragraphs = load_frame(self.path, 'paragraph')

    def group(self):
        digest = hashlib.sha256(self.shapes.read_bytes()).hexdigest()
        return store.location_groups(self.paragraphs, store.read_references(self.path), digest)['delft']

    def test_unique_names_reuse_existing_work_without_rewriting(self):
        self.duplicate_locations()
        # Work saved on a later occurrence must prefill the first occurrence, too.
        self.save(1, 'point', 52, 4.5, notes='Already done')
        before = (self.path / store.FILENAME).read_bytes()
        self.assertTrue(self.group()['complete'])
        self.assertEqual(store.load_gold_regions(self.path, self.shapes).gold_region.tolist(),
                         ['nuts2:NL33', 'nuts2:NL33'])
        with patch.dict(os.environ, {'FINAL_ANNOTATION_DIR': str(self.path),
                                    'FINAL_ANNOTATION_SHAPES': str(self.shapes)}):
            app = AppTest.from_file(str(ROOT / 'annotation/final/georeference_app.py')).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.selectbox(key='geo_sample').options), 1)
            self.assertEqual(app.text_input(key='geo_lon').value, '4.5')
            self.assertEqual(app.text_area(key='geo_notes').value, 'Already done')
            self.assertEqual((self.path / store.FILENAME).read_bytes(), before)
            app.text_input(key='geo_lon').set_value('5.5')
            next(b for b in app.button if b.label == 'Save').click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
        self.assertEqual(store.load_gold_regions(self.path, self.shapes).gold_region.tolist(),
                         ['nuts2:NL31', 'nuts2:NL31'])
        self.assertEqual(set(store.read_references(self.path).sample_id), {'p1', 'p2'})
        self.assertTrue(any(path.read_bytes() == before for path in (self.path / 'backups').glob('*.csv')))
        self.assertNotEqual(store.location_key('Den Haag'), store.location_key('The Hague'))

    def test_unique_conflicts_and_group_edit_protection(self):
        self.duplicate_locations()
        self.save(0, 'point', 52, 4.5)
        self.save(1, 'point', 52, 5.5)
        group = self.group()
        self.assertTrue(group['conflict'])
        self.assertFalse(group['complete'])
        with self.assertRaisesRegex(ValueError, 'Conflicting saved regions'):
            store.load_gold_regions(self.path, self.shapes)
        # An edit to another occurrence must invalidate an open unique-name editor.
        old = store.current_reference(store.read_references(self.path), self.paragraphs.iloc[1])
        self.save(1, 'point', 52, 5.6, expected=old)
        with self.assertRaisesRegex(ValueError, 'another window'):
            store.save_reference(self.path, self.paragraphs.iloc[0], 'point', 52, 4.5, '',
                                 group['expected'], self.shapes, unique_name=True)
        store.save_reference(self.path, self.paragraphs.iloc[0], 'point', 52, 4.5, 'Shared judgment',
                             self.group()['expected'], self.shapes, unique_name=True)
        self.assertTrue(self.group()['complete'])
        self.assertEqual(store.load_gold_regions(self.path, self.shapes).gold_region.tolist(),
                         ['nuts2:NL33', 'nuts2:NL33'])

    def test_actual_workflow_chain_and_distinct_matching(self):
        gazetteer = self.path / 'geonames'
        gazetteer.mkdir()
        with zipfile.ZipFile(gazetteer / 'NL.zip', 'w') as bundle:
            bundle.writestr('NL.txt', '\t'.join(['1', 'Rijswijk', 'Rijswijk', '', '52', '4.4',
                'P', 'PPL', 'NL', '', '', '', '', '', '1000', '', '', '', '2026-01-01']) + '\n')
        (self.path / 'cache.jsonl').write_text('')
        (self.path / 'overrides.csv').write_text('location,action\n')
        names = ['Rijswijk', 'Utrecht', 'NONE', 'Unknown', 'Netherlands', 'NONE']
        predictions = pd.DataFrame([dict(sample_id=f'p{i}', model='test', task='location',
            prediction=name.lower(), raw={'location': name, 'granularity': 'country' if i == 4 else 'city'},
            confidence=.9, error='failure' if i == 5 else '', seconds=1., setup_seconds=0.)
            for i, name in enumerate(names)])
        paragraphs = pd.DataFrame({'sample_id': predictions.sample_id, 'paragraph_text': ['Text'] * 6,
            'gold_location': ['delft', 'delft', 'none', 'delft', 'netherlands', 'none'],
            'gold_geothermal': ['YES'] * 6, 'split': ['calibration', 'test', 'test', 'test', 'test', 'test']})
        result = geography.workflow_georeference(predictions, paragraphs, self.path, self.shapes,
            gazetteer, self.path / 'cache.jsonl', self.path / 'overrides.csv')
        self.assertEqual(result.predicted_region.tolist(), ['nuts2:NL33', 'nuts2:NL31', 'none',
            geography.UNRESOLVED, 'country:NLD', geography.UNRESOLVED])
        gold = pd.DataFrame({'sample_id': predictions.sample_id,
            'gold_region': ['nuts2:NL33', 'nuts2:NL33', 'none', 'nuts2:NL33', 'country:NLD', geography.UNRESOLVED],
            'gold_region_status': ['point', 'point', 'none', 'point', 'country', 'unresolved']})
        evaluate.evaluate_predictions(predictions, {'paragraph': paragraphs}, self.path, .8, result, gold)
        detail = pd.read_csv(self.path / 'location_comparison.csv')
        self.assertFalse(detail.iloc[0].exact_match)
        self.assertTrue(detail.iloc[0].region_match)
        metrics = pd.read_csv(self.path / 'metrics_detailed.csv')
        regional = metrics[metrics.task.eq('location_region') & metrics.subset.eq('all')].iloc[0]
        self.assertEqual(regional.accuracy, .6)
        self.assertEqual(regional.accepted_accuracy, .75)
        self.assertEqual(regional.n_samples, 5)
        self.assertEqual(regional.n_excluded_gold_regions, 1)
        self.assertEqual(regional.n_unresolved_regions, 1)
        self.assertTrue((self.path / 'geocoding/run.json').exists())


if __name__ == '__main__':
    unittest.main()
