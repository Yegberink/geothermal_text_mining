"""Interactive place-name search in the project's local Dutch GeoNames gazetteer."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from core_workflow.geocoding_online import read_geonames_country_zip, geonames_priority
from helpers.shape_resources import normalize_key


def load_places(path):
    frame = read_geonames_country_zip(Path(path), 'NL')
    frame = frame[frame.feature_class.isin(['A', 'P', 'L', 'T', 'V', 'S'])].copy()
    frame['_names'] = frame.apply(lambda row: tuple(dict.fromkeys(
        name for value in [row['name'], row['asciiname'], *row['alternatenames'].split(',')]
        if (name := normalize_key(value)))), axis=1)
    frame['_priority'] = frame.apply(geonames_priority, axis=1)
    return frame


def search_places(places, query, limit=20):
    query = normalize_key(query)
    if not query:
        return []

    def rank(names):
        if query in names:
            return 0
        if any(name.startswith(query) for name in names):
            return 1
        return 2 if any(query in name for name in names) else 3

    matches = places.assign(_rank=places['_names'].map(rank))
    matches = matches[matches._rank.lt(3)].sort_values(
        ['_rank', '_priority', 'population', 'name', 'geonameid'],
        ascending=[True, True, False, True, True])
    # Keep homonymous places as separate choices; aliases of a place share one result.
    return matches.drop_duplicates('geonameid').head(limit)[
        ['geonameid', 'name', 'feature_code', 'latitude', 'longitude']].to_dict('records')
