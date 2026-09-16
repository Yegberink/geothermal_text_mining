"""Run: pixi run streamlit run annotation/final/app.py"""
from __future__ import annotations

import os
import json
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
from annotation_store import (ANNOTATORS, JOINT, FIELDS, SENTIMENTS, completed,
                              load_assignments, load_frame, save_annotation, load_documents)

DATA_DIR = Path(os.environ.get('FINAL_ANNOTATION_DIR', str(APP_DIR))).resolve()


@st.cache_data(show_spinner='Loading source documents…', max_entries=2)
def cached_documents(source, expected_sha256, modified_ns, size):
    # mtime and size invalidate the cache if a workflow run replaces the source.
    return load_documents(Path(source), expected_sha256)


def show_document(document_id, directory=DATA_DIR):
    with st.expander('Show full document', expanded=False):
        try:
            manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
            origin = manifest.get('document_context', manifest['sources']['paragraph'])
            source = Path(origin['path'])
            if not source.is_absolute():
                source = directory / source if 'document_context' in manifest else APP_DIR.parents[1] / source
            stat = source.stat()
            documents = cached_documents(str(source), origin['sha256'], stat.st_mtime_ns, stat.st_size)
            document = documents.get(document_id)
            if document is None:
                st.info('Full document text is unavailable for this paragraph.')
                return
            if document['document_title']:
                st.text(document['document_title'])
            metadata = ' · '.join(document[k] for k in ['source', 'publish_date'] if document[k])
            if metadata:
                st.text(metadata)
            with st.container(height=450, border=False):
                st.text(document['body'])
        except (OSError, ValueError, KeyError) as exc:
            st.info(f'Full document unavailable: {exc}')


def reset_editor():
    st.session_state.pop('editor', None)
    for key in list(st.session_state):
        if key.startswith('edit_'):
            del st.session_state[key]


def select_item(selector, sample_id, back=False):
    current = st.session_state.get(selector)
    if current is not None and current != sample_id:
        history = st.session_state.setdefault(f'{selector}_history', [])
        if back:
            if history:
                history.pop()
        else:
            history.append(current)
    st.session_state[selector] = sample_id


def submit_annotation(task, sample_id, expected, selector, destination, annotator=None):
    keys = {'gold_location': 'edit_location', 'gold_geothermal': 'edit_geothermal',
            'gold_sentiment': 'edit_sentiment', 'notes': 'edit_notes'}
    values = {field: st.session_state[keys[field]] or '' for field in FIELDS[task]}
    try:
        save_annotation(DATA_DIR, task, sample_id, values, expected, annotator=annotator)
    except (ValueError, OSError, KeyError) as exc:
        st.session_state['save_error'] = str(exc)
    else:
        st.session_state.notice = 'Annotation saved.'
        select_item(selector, destination)
        reset_editor()


def main():
    st.set_page_config(page_title='Dutch text annotation', page_icon='✍️', layout='centered')
    st.title('Dutch text annotation')
    st.caption('Read the text, record your judgment, and save to continue.')
    try:
        frames = {task: load_frame(DATA_DIR, task) for task in FIELDS}
        assignments = load_assignments(DATA_DIR, frames)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f'Cannot open the annotation set: {exc}')
        st.info('Prepare the files with annotation/final/prepare_annotation.ipynb first.')
        st.stop()

    annotator, review_joint = None, False
    with st.sidebar:
        if assignments is not None:
            package_owner = assignments.get('package_annotator')
            if package_owner:
                annotator = package_owner
                st.subheader(f'Annotator: {annotator}')
            else:
                annotator = st.radio('Annotator', ANNOTATORS, index=None, key='annotator', on_change=reset_editor)
            joint_count = sum(owner == JOINT for owner in assignments['tasks']['paragraph'].values())
            st.caption(f'{joint_count} paragraphs completed together are preserved.')
            if annotator is None:
                st.info('Select your name to open your assigned items.')
                st.stop()
            review_joint = st.checkbox('Review annotations completed together', on_change=reset_editor)
            owner = JOINT if review_joint else annotator
            frames = {task: frame[frame.sample_id.map(assignments['tasks'][task]).eq(owner)].copy()
                      for task, frame in frames.items()}
        st.header('Your progress')
        for name, frame in frames.items():
            count = int(completed(frame, name).sum())
            st.progress(count / len(frame) if len(frame) else 0,
                        text=f'{name.title()}s: {count} / {len(frame)} saved')
        task = st.radio('Annotation task', ['paragraph', 'sentence'],
                        format_func=lambda t: 'Paragraphs · location & relevance' if t == 'paragraph' else 'Sentences · sentiment')
        queue = st.radio('Show', ['Unfinished', 'All', 'Completed'])
        st.caption('Use Save before changing tasks or navigating. Unsaved edits are discarded.')
        st.button('Reload saved annotation', on_click=reset_editor)
        with st.expander('Download saved annotations'):
            for name in FIELDS:
                if assignments is None or assignments.get('package_annotator'):
                    data = (DATA_DIR / f'{name}s.csv').read_bytes()
                    filename = f'{name}s.csv'
                else:
                    # Keep original metadata while exporting only this work set.
                    export = pd.read_csv(DATA_DIR / f'{name}s.csv', dtype=str, keep_default_na=False)
                    export = export[export.sample_id.isin(frames[name].sample_id)].copy()
                    export['annotator'] = owner
                    data = export.to_csv(index=False).encode('utf-8')
                    filename = f'{owner.lower()}_{name}s.csv'
                st.download_button(f'Download {name}s.csv',
                                   data=data, file_name=filename, mime='text/csv')
            if assignments is not None and assignments.get('package_annotator'):
                archive = io.BytesIO()
                with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
                    for filename in ['paragraphs.csv', 'sentences.csv', 'manifest.json']:
                        bundle.write(DATA_DIR / filename, arcname=f'{annotator}/{filename}')
                st.download_button('Download work for merging', data=archive.getvalue(),
                    file_name=f'{annotator}_completed.zip', mime='application/zip')

    if 'notice' in st.session_state:
        st.success(st.session_state.pop('notice'))
    if 'save_error' in st.session_state:
        st.error(st.session_state.pop('save_error'))
    frame = frames[task]
    done = completed(frame, task)
    filtered = frame[~done] if queue == 'Unfinished' else frame[done] if queue == 'Completed' else frame
    ids = filtered.sample_id.tolist()
    # Previous must still reach saved items while working through unfinished ones.
    navigation_ids = frame.sample_id.tolist() if queue == 'Unfinished' else ids
    if not navigation_ids:
        st.info('No items in this view yet.')
        st.stop()
    if queue == 'Unfinished' and not ids:
        st.success('All items in this task are saved. You can still review and correct your annotations below.')
    positions = {sid: i + 1 for i, sid in enumerate(frame.sample_id)}
    scope = JOINT if review_joint else annotator or 'all'
    selector = f'current_{scope}_{task}_{queue}'
    if st.session_state.get(selector) not in navigation_ids:
        st.session_state[selector] = ids[0] if ids else navigation_ids[-1]
    sample_id = st.session_state[selector]
    index = navigation_ids.index(sample_id)
    history = st.session_state.get(f'{selector}_history', [])
    previous_id = history[-1] if history else navigation_ids[index - 1] if index else None
    next_id = next((sid for sid in ids if positions[sid] > positions[sample_id]), None)
    row = frame.set_index('sample_id').loc[sample_id]
    identity = (scope, task, sample_id)
    if st.session_state.get('editor', {}).get('identity') != identity:
        reset_editor()
        st.session_state.editor = {'identity': identity, 'expected': row[FIELDS[task]].to_dict()}
    saved = st.session_state.editor['expected']
    st.subheader(f'{task.title()} {positions[sample_id]} of {len(frame)}')
    if review_joint:
        st.info('Completed together. These saved labels are read-only.')
    elif done.iloc[positions[sample_id] - 1]:
        st.caption('This annotation is saved. Edit the labels and save again to correct it.')
    with st.container(border=True):
        # Plain text prevents source text from being interpreted as Markdown/HTML.
        st.text(row[f'{task}_text'])
    if task == 'paragraph':
        show_document(row['document_id'])
    with st.expander('Annotation guide'):
        if task == 'paragraph':
            st.markdown('**Location:** choose the one most specific place clearly at the center of the paragraph. '
                        'Use consistent Dutch place names. If there is no clear primary location, enter `NONE`. '
                        'Annotate location even when the paragraph is unrelated to geothermal energy.\n\n'
                        '**Geothermal relevance:** Yes when geothermal energy is the main or a substantial subject. '
                        'No when absent, incidental, or another subject dominates. Resolve uncertainty manually; use notes to explain.')
        else:
            st.markdown('**Negative:** risks, costs, criticism, uncertainty, conflict, harm, or failure.\n\n'
                        '**Neutral:** factual, procedural, descriptive, or mixed without clear polarity.\n\n'
                        '**Positive:** benefits, support, progress, feasibility, opportunity, or success.\n\n'
                        '**Not about geothermal energy:** use this when the sentence is unrelated; '
                        'it will be excluded from sentiment evaluation. Judge the sentence itself.')
    with st.form('annotation_form', enter_to_submit=False):
        if task == 'paragraph':
            st.text_input('Primary location', value=saved['gold_location'],
                placeholder='Place name, or NONE', key='edit_location', disabled=review_joint)
            choices = ['YES', 'NO']
            previous = saved['gold_geothermal'].strip().upper()
            st.radio('Is this paragraph about geothermal energy?', choices,
                index=choices.index(previous) if previous in choices else None,
                format_func=lambda x: 'Yes' if x == 'YES' else 'No', horizontal=True,
                key='edit_geothermal', disabled=review_joint)
        else:
            previous = saved['gold_sentiment'].strip().lower()
            st.radio('Sentiment', SENTIMENTS,
                index=SENTIMENTS.index(previous) if previous in SENTIMENTS else None,
                format_func=lambda x: 'Not about geothermal energy' if x == 'not_geothermal' else x.title(),
                key='edit_sentiment', disabled=review_joint)
        st.text_area('Notes (optional)', value=saved['notes'], key='edit_notes', height=90, disabled=review_joint)
        left, right = st.columns(2)
        destination = next_id or next((sid for sid in ids if sid != sample_id), sample_id)
        left.form_submit_button('Save & next', type='primary', use_container_width=True,
            on_click=submit_annotation, args=(task, sample_id, saved, selector, destination, annotator),
            disabled=review_joint)
        right.form_submit_button('Save', use_container_width=True,
            on_click=submit_annotation, args=(task, sample_id, saved, selector, sample_id, annotator),
            disabled=review_joint)
    left, right = st.columns(2)
    left.button('← Previous', disabled=previous_id is None, use_container_width=True,
        on_click=select_item, args=(selector, previous_id, True))
    right.button('Skip / next →', disabled=next_id is None, use_container_width=True,
        on_click=select_item, args=(selector, next_id))
    st.caption('Saved labels are written directly to the evaluation CSVs. You can close the app and resume later.')


if __name__ == '__main__':
    main()
