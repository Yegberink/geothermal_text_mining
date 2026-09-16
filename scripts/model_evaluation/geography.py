"""Region evaluation using the production geocoding and aggregation entry points."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from core_workflow.geographic_aggregation import assign_nuts2
from helpers.shape_resources import load_shapes_parquet, nuts2_shapes

UNRESOLVED = '__unresolved__'


def load_regions(path):
    regions = nuts2_shapes(load_shapes_parquet(path), 'Netherlands')
    if regions.empty:
        raise ValueError('No Dutch NUTS2 boundaries in the shapes file')
    return regions


def point_region(lat, lon, regions):
    lat, lon = float(lat), float(lon)
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError('Select valid latitude and longitude')
    matches = regions[regions.geometry.contains(Point(lon, lat))]
    if len(matches) != 1:
        raise ValueError('Choose a point inside one Dutch region, away from its boundary; '
                         'use Unresolved / outside scope if no Dutch region applies.')
    row = matches.iloc[0]
    return {'country_id': str(row.country_id), 'region_id': str(row.parent_id),
            'region_name': str(row.parent_name)}


def region_key(row):
    if str(row.get('geo_level', '')).lower() == 'country' and pd.notna(row.get('country_id')):
        return 'country:' + str(row['country_id'])
    if pd.notna(row.get('nuts2_id')) and str(row.get('nuts2_id', '')).strip():
        return 'nuts2:' + str(row['nuts2_id'])
    return UNRESOLVED


def workflow_georeference(predictions, paragraphs, output, shapes, geonames, cache, overrides,
                          country_codes='nl,be,bq,aw,cw,sx'):
    """Run the same offline → overrides/GeoNames/cache → NUTS2 chain as Snakefile.

    All outputs are evaluation-local. The production gazetteer/cache/overrides are read only.
    Preserve raw case and granularity: the workflow cache keys are case sensitive.
    """
    from core_workflow.classification_sentences import attach_geometry_from_wkt
    from prepare import file_digest

    output = Path(output).resolve() / 'geocoding'
    output.mkdir(parents=True, exist_ok=True)
    locations = predictions[predictions.task.eq('location')].copy().reset_index(drop=True)
    sources = paragraphs.set_index('sample_id')
    rows = []
    for index, item in locations.iterrows():
        raw = item['raw']
        if not item.error and not {'location', 'granularity'}.issubset(raw):
            raise ValueError('Location prediction lacks workflow location/granularity; rerun inference.')
        row = {'evaluation_id': str(index), 'sample_id': item.sample_id, 'model': item.model,
               'paragraph_text': sources.loc[item.sample_id, 'paragraph_text'],
               'llm_location': str(raw.get('location', 'NONE')).strip() if not item.error else 'NONE',
               'llm_granularity': raw.get('granularity', 'none') if not item.error else 'none'}
        row.update({key: value for key, value in raw.items() if key.startswith('llm_country')})
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / 'inputs.csv', index=False)
    common = ['--project-dir', str(ROOT), '--shapes-parquet', str(Path(shapes).resolve()),
              '--country', 'Netherlands']
    commands = [
        [sys.executable, str(ROOT / 'scripts/core_workflow/geocoding_offline.py'), *common,
         '--input-csv', str(output / 'inputs.csv'), '--output-csv', str(output / 'shapes.csv'),
         '--output-gpkg', str(output / 'shapes.gpkg')],
        [sys.executable, str(ROOT / 'scripts/core_workflow/geocoding_online.py'), *common,
         '--input-csv', str(output / 'shapes.csv'), '--output-csv', str(output / 'locations.csv'),
         '--output-gpkg', str(output / 'locations.gpkg'),
         '--geonames-dir', str(Path(geonames).resolve()), '--geonames-country-codes', 'nl',
         '--country-codes', country_codes, '--geocoder-cache-path', str(Path(cache).resolve()),
         '--overrides-csv', str(Path(overrides).resolve()),
         '--unmatched-csv', str(output / 'unmatched.csv'),
         '--suggestions-csv', str(output / 'suggestions.csv')],
    ]
    resources = [Path(shapes), Path(geonames) / 'NL.zip', Path(cache), Path(overrides),
                 Path(__file__), *[ROOT / 'scripts/core_workflow' / name for name in
                 ['geocoding_offline.py', 'geocoding_online.py', 'geographic_aggregation.py',
                  'classification_sentences.py']], ROOT / 'scripts/helpers/shape_resources.py',
                 ROOT / 'scripts/helpers/country_scope.py']
    provenance = {'commands': commands, 'sha256': {
        str(path.resolve()): file_digest(path) if path.exists() else None for path in resources}}
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(ROOT / 'scripts') + os.pathsep + environment.get('PYTHONPATH', '')
    for command in commands:
        subprocess.run(command, check=True, cwd=ROOT, env=environment)
    frame = pd.read_csv(output / 'locations.csv', dtype={'sample_id': str})
    geometry, _, _ = attach_geometry_from_wkt(frame)
    assigned = assign_nuts2(geometry, load_regions(shapes))
    assigned['predicted_region'] = assigned.apply(region_key, axis=1)
    assigned.loc[locations.prediction.eq('none').to_numpy() & locations.error.eq('').to_numpy(),
                 'predicted_region'] = 'none'
    assigned.loc[locations.error.ne('').to_numpy(), 'predicted_region'] = UNRESOLVED
    assigned.drop(columns='geometry').to_csv(output / 'regions.csv', index=False)
    (output / 'run.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    return assigned[['sample_id', 'model', 'predicted_region', 'geo_source', 'geo_match_type',
                     'geo_lat', 'geo_lon', 'nuts2_id', 'nuts2_name']]
