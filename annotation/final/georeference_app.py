"""Run: pixi run streamlit run annotation/final/georeference_app.py"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
from annotation_store import load_frame
from georeference_store import (FILENAME, current_reference, load_regions, point_region,
                                read_references, save_reference, location_groups)
from app import show_document
from georeference_search import load_places, search_places

DATA_DIR = Path(os.environ.get('FINAL_ANNOTATION_DIR', APP_DIR)).resolve()
SHAPES = Path(os.environ.get('FINAL_ANNOTATION_SHAPES', APP_DIR.parents[1] / 'data/shapes.parquet')).resolve()
GAZETTEER = Path(os.environ.get('FINAL_ANNOTATION_GEONAMES', APP_DIR.parents[1] / 'data/geonames/NL.zip')).resolve()
osm_picker = components.declare_component('manual_osm_location', path=str(APP_DIR / 'osm_component'))


@st.cache_data
def boundaries(path, modified_ns):
    return load_regions(path), hashlib.sha256(Path(path).read_bytes()).hexdigest()


@st.cache_data(show_spinner='Loading place names…')
def places(path, modified_ns):
    return load_places(path)


def choose_place():
    selected = st.session_state.geo_search_result
    if selected is None:
        return
    result = st.session_state.geo_search_results[selected]
    st.session_state.geo_lat = str(result['latitude'])
    st.session_state.geo_lon = str(result['longitude'])
    st.session_state.geo_use_fallback = False
    st.session_state.geo_focus = st.session_state.get('geo_focus', 0) + 1


def show_search(regions):
    with st.form('place_search'):
        st.text_input('Search place names', key='geo_query')
        submitted = st.form_submit_button('Search')
    if submitted:
        st.session_state.geo_search_results = []
        st.session_state.pop('geo_search_result', None)
        try:
            matches = search_places(places(str(GAZETTEER), GAZETTEER.stat().st_mtime_ns),
                                    st.session_state.geo_query)
            for match in matches:
                try:
                    area = point_region(match['latitude'], match['longitude'], regions)['region_name']
                except ValueError:
                    area = 'outside Dutch regions'
                match['label'] = (f'{match["name"]} · {area} '
                                  f'({match["latitude"]:.4f}, {match["longitude"]:.4f})')
            st.session_state.geo_search_results = matches
            if not matches:
                st.info('No places found. Try another spelling or a shorter name, or navigate the map.')
        except (OSError, ValueError, KeyError) as exc:
            st.warning(f'Place-name search is unavailable: {exc}. You can still use the map.')
    results = st.session_state.get('geo_search_results', [])
    if results:
        st.selectbox('Choose a place to show on the map', range(len(results)), index=None,
                     format_func=lambda index: results[index]['label'],
                     key='geo_search_result', on_change=choose_place)


def navigate(sample_id):
    st.session_state.geo_sample = sample_id


def submit_reference(row, expected, destination):
    try:
        save_reference(DATA_DIR, row, st.session_state.geo_status,
                       st.session_state.get('geo_lat', ''), st.session_state.get('geo_lon', ''),
                       st.session_state.geo_notes, expected, SHAPES, unique_name=True)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        st.session_state.geo_error = str(exc)
    else:
        st.session_state.pop('geo_editor', None)
        st.session_state['geo_revision'] = st.session_state.get('geo_revision', 0) + 1
        st.session_state.geo_sample = destination
        st.session_state.geo_notice = 'Map judgment saved for every occurrence of this location name.'


def main():
    st.set_page_config(page_title='Manual location georeferencing', layout='wide')
    st.title('Manual location georeferencing')
    st.caption('Georeference each unique location name once. Search a name or click its position.')
    if 'geo_notice' in st.session_state:
        st.success(st.session_state.pop('geo_notice'))
    if 'geo_error' in st.session_state:
        st.error(st.session_state.pop('geo_error'))
    try:
        manifest = json.loads((DATA_DIR / 'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('annotation_assignments', {}).get('package_annotator'):
            raise ValueError('Open the combined master annotation folder after merging the returned copies.')
        frame = load_frame(DATA_DIR, 'paragraph')
        refs = read_references(DATA_DIR)
        regions, digest = boundaries(str(SHAPES), SHAPES.stat().st_mtime_ns)
    except (OSError, ValueError, KeyError) as exc:
        st.error(str(exc))
        st.stop()
    no_location = frame.gold_location.str.casefold().eq('none')
    missing_location = frame.gold_location.eq('')
    groups = location_groups(frame, refs, digest)
    if not groups:
        st.info('There are no named locations to georeference. Existing NONE labels are handled automatically.')
        st.stop()
    # Keep an active sample as its group's representative so an open editor retains its draft.
    active = st.session_state.get('geo_sample')
    by_id = {}
    for group in groups.values():
        members = group['members']
        sid = active if active in set(members.sample_id) else members.iloc[0].sample_id
        by_id[sid] = group
    done = {sid for sid, group in by_id.items() if group['complete']}
    ids = list(by_id)
    remaining = [sid for sid in ids if sid not in done]
    labels = {sid: f'{i + 1}. {group["members"].iloc[0].gold_location} '
              f'({len(group["members"])} paragraphs) {"✓" if sid in done else ""}'
              for i, (sid, group) in enumerate(by_id.items())}
    if st.session_state.get('geo_sample') not in ids:
        st.session_state.geo_sample = (remaining or ids)[0]
    with st.sidebar:
        st.progress(len(done) / len(groups), text=f'{len(done)} / {len(groups)} unique locations georeferenced')
        st.caption(f'{int(no_location.sum())} NONE labels handled automatically. '
                   f'{int(missing_location.sum())} paragraphs still need a location label.')
        sample_id = st.selectbox('Unique location name', ids, format_func=labels.get, key='geo_sample')
        st.button('Next unfinished', disabled=not remaining, on_click=navigate,
                  args=((remaining or [sample_id])[0],))
        if st.button('Reload saved judgment'):
            st.session_state.pop('geo_editor', None)
            st.session_state['geo_revision'] = st.session_state.get('geo_revision', 0) + 1
        if (DATA_DIR / FILENAME).exists():
            st.download_button('Download map judgments', (DATA_DIR / FILENAME).read_bytes(),
                               file_name=FILENAME, mime='text/csv')
        st.caption('Save before navigating. Map clicks are only stored when you press Save.')
    group = by_id[sample_id]
    row = group['members'].set_index('sample_id', drop=False).loc[sample_id]
    identity = sample_id + ':' + str(st.session_state.get('geo_revision', 0))
    editor = st.session_state.get('geo_editor', {})
    if editor.get('identity') == identity and 'sources' not in (editor.get('expected') or {}):
        if current_reference(refs, row) == editor.get('expected'):
            editor['expected'] = group['expected']
    if st.session_state.get('geo_editor', {}).get('identity') != identity:
        saved = group['saved']
        st.session_state.geo_editor = {'identity': identity, 'expected': group['expected']}
        for key in ['geo_status', 'geo_lat', 'geo_lon', 'geo_notes', 'geo_click', 'geo_use_fallback',
                    'geo_fallback', 'geo_query', 'geo_search_results', 'geo_search_result', 'geo_focus']:
            st.session_state.pop(key, None)
        saved = saved or {}
        fallback = saved.get('status') if saved.get('status') in ['country', 'unresolved'] else None
        st.session_state.geo_status = fallback or 'point'
        st.session_state.geo_use_fallback = fallback is not None
        st.session_state.geo_fallback = fallback
        st.session_state.geo_query = row.gold_location
        st.session_state.geo_lat = saved.get('latitude', '')
        st.session_state.geo_lon = saved.get('longitude', '')
        st.session_state.geo_notes = saved.get('notes', '')
    expected = st.session_state.geo_editor['expected']
    if group['conflict']:
        st.warning('This name has conflicting saved regions. Choose the shared judgment and save to resolve it. '
                   'The previous records will be backed up.')
        st.dataframe([{key: record[key] for key in ['gold_location', 'status', 'region_name',
                                                   'latitude', 'longitude', 'notes']}
                      for record in group['valid']], hide_index=True)
    elif group['saved'] and not group['complete']:
        st.warning('The location label or boundaries changed. Review and save this judgment again.')
    st.subheader(row.gold_location)
    st.caption(f'Used in {len(group["members"])} paragraphs. One saved judgment applies to all occurrences.')
    show_search(regions)
    try:
        lat, lon = float(st.session_state.geo_lat), float(st.session_state.geo_lon)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            lat, lon = None, None
    except ValueError:
        lat, lon = None, None
    click = osm_picker(identity=identity, latitude=lat, longitude=lon,
                       focus=st.session_state.get('geo_focus', 0),
                       selection_event=st.session_state.get('geo_click', {}).get('event'),
                       regions=json.loads(regions[['parent_name', 'geometry']].to_json()),
                       key='map_' + identity, default=None)
    if click and click['identity'] == identity and click != st.session_state.get('geo_click'):
        st.session_state.geo_lat = str(click['latitude'])
        st.session_state.geo_lon = str(click['longitude'])
        st.session_state.geo_click = click
        st.session_state.geo_use_fallback = False
    with st.expander('Coordinates (optional)'):
        left, right = st.columns(2)
        left.text_input('Latitude', key='geo_lat')
        right.text_input('Longitude', key='geo_lon')
    fallback = st.checkbox('Use a fallback instead of a map point', key='geo_use_fallback')
    if fallback:
        statuses = {'country': 'Netherlands as a whole', 'unresolved': 'Unresolved / outside Dutch scope'}
        st.radio('Fallback', list(statuses), index=None, format_func=statuses.get, key='geo_fallback')
    st.session_state.geo_status = st.session_state.get('geo_fallback') if fallback else 'point'
    if not fallback:
        try:
            region = point_region(st.session_state.geo_lat, st.session_state.geo_lon, regions)
            st.info(f'Region: {region["region_name"]} ({region["region_id"]})')
        except (ValueError, TypeError) as exc:
            st.caption('Choose a point on the map. ' + str(exc))
    st.text_area('Notes / reason if unresolved', key='geo_notes')
    left, right = st.columns(2)
    next_ids = [sid for sid in remaining if sid != sample_id]
    left.button('Save & next', type='primary', on_click=submit_reference,
                args=(row, expected, (next_ids or [sample_id])[0]))
    right.button('Save', on_click=submit_reference, args=(row, expected, sample_id))
    left, right = st.columns(2)
    position = ids.index(sample_id)
    left.button('← Previous', disabled=position == 0, on_click=navigate, args=(ids[max(0, position - 1)],))
    right.button('Next →', disabled=position == len(ids) - 1, on_click=navigate,
                 args=(ids[min(len(ids) - 1, position + 1)],))
    with st.expander('Original paragraph (optional)'):
        st.text(row.paragraph_text)
    show_document(row.document_id, DATA_DIR)


if __name__ == '__main__':
    main()
