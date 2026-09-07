"""Run: pixi run streamlit run annotation/final/app.py"""
from __future__ import annotations

import os
import json
import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
from annotation_store import FIELDS, SENTIMENTS, completed, load_frame, save_annotation, load_documents

DATA_DIR = Path(os.environ.get('FINAL_ANNOTATION_DIR', str(APP_DIR))).resolve()


@st.cache_data(show_spinner='Loading source documents…', max_entries=2)
def cached_documents(source, expected_sha256, modified_ns, size):
    # mtime and size invalidate the cache if a workflow run replaces the source.
    return load_documents(Path(source), expected_sha256)


def show_document(document_id):
    with st.expander('Show full document', expanded=False):
        try:
            manifest = json.loads((DATA_DIR / 'manifest.json').read_text())
            origin = manifest['sources']['paragraph']
            source = Path(origin['path'])
            if not source.is_absolute():
                source = APP_DIR.parents[1] / source
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


def main():
    st.set_page_config(page_title='Dutch text annotation', page_icon='✍️', layout='centered')
    st.title('Dutch text annotation')
    st.caption('Read the text, record your judgment, and save to continue.')
    try:
        frames = {task: load_frame(DATA_DIR, task) for task in FIELDS}
    except (OSError, ValueError, KeyError) as exc:
        st.error(f'Cannot open the annotation set: {exc}')
        st.info('Prepare the files with annotation/final/prepare_annotation.ipynb first.')
        st.stop()

    with st.sidebar:
        st.header('Your progress')
        for name, frame in frames.items():
            count = int(completed(frame, name).sum())
            st.progress(count / len(frame) if len(frame) else 0,
                        text=f'{name.title()}s: {count} / {len(frame)} saved')
        task = st.radio('Annotation task', ['paragraph', 'sentence'],
                        format_func=lambda t: 'Paragraphs · location & relevance' if t == 'paragraph' else 'Sentences · sentiment')
        queue = st.radio('Show', ['Unfinished', 'All', 'Completed'])
        st.caption('Use Save before changing tasks or navigating. Unsaved edits are discarded.')
        if st.button('Reload saved annotation'):
            reset_editor()
            st.rerun()
        with st.expander('Download saved annotations'):
            for name in FIELDS:
                st.download_button(f'Download {name}s.csv',
                                   data=(DATA_DIR / f'{name}s.csv').read_bytes(),
                                   file_name=f'{name}s.csv', mime='text/csv')

    if 'notice' in st.session_state:
        st.success(st.session_state.pop('notice'))
    frame = frames[task]
    done = completed(frame, task)
    filtered = frame[~done] if queue == 'Unfinished' else frame[done] if queue == 'Completed' else frame
    if filtered.empty:
        st.success('All items in this task are saved.' if queue == 'Unfinished' else 'No items in this view yet.')
        st.stop()
    ids = filtered.sample_id.tolist()
    positions = {sid: i + 1 for i, sid in enumerate(frame.sample_id)}
    selector = f'selection_{task}_{queue}'
    pending = st.session_state.pop('next_item', None)
    if pending in ids:
        st.session_state[selector] = pending
    if st.session_state.get(selector) not in ids:
        st.session_state[selector] = ids[0]
    sample_id = st.selectbox('Go to item', ids, key=selector,
                            format_func=lambda sid: f'{task.title()} {positions[sid]} of {len(frame)}')
    row = frame.set_index('sample_id').loc[sample_id]
    identity = (task, sample_id)
    if st.session_state.get('editor', {}).get('identity') != identity:
        reset_editor()
        st.session_state.editor = {'identity': identity, 'expected': row[FIELDS[task]].to_dict()}
    saved = st.session_state.editor['expected']
    st.subheader(f'{task.title()} {positions[sample_id]}')
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
        values = {}
        if task == 'paragraph':
            values['gold_location'] = st.text_input('Primary location', value=saved['gold_location'],
                placeholder='Place name, or NONE', key='edit_location')
            choices = ['YES', 'NO']
            previous = saved['gold_geothermal'].strip().upper()
            values['gold_geothermal'] = st.radio('Is this paragraph about geothermal energy?', choices,
                index=choices.index(previous) if previous in choices else None,
                format_func=lambda x: 'Yes' if x == 'YES' else 'No', horizontal=True, key='edit_geothermal')
        else:
            previous = saved['gold_sentiment'].strip().lower()
            values['gold_sentiment'] = st.radio('Sentiment', SENTIMENTS,
                index=SENTIMENTS.index(previous) if previous in SENTIMENTS else None,
                format_func=lambda x: 'Not about geothermal energy' if x == 'not_geothermal' else x.title(),
                key='edit_sentiment')
        values['notes'] = st.text_area('Notes (optional)', value=saved['notes'], key='edit_notes', height=90)
        left, right = st.columns(2)
        save_next = left.form_submit_button('Save & next', type='primary', use_container_width=True)
        save_only = right.form_submit_button('Save', use_container_width=True)
    if save_next or save_only:
        try:
            save_annotation(DATA_DIR, task, sample_id, {k: v or '' for k, v in values.items()}, saved)
        except (ValueError, OSError, KeyError) as exc:
            st.error(str(exc))
        else:
            st.session_state.notice = f'{task.title()} {positions[sample_id]} saved.'
            if save_next:
                index = ids.index(sample_id)
                st.session_state.next_item = ids[(index + 1) % len(ids)]
            # Reset widget state at the beginning of the next render.
            st.session_state.pop('editor', None)
            st.rerun()
    left, right = st.columns(2)
    index = ids.index(sample_id)
    if left.button('← Previous', disabled=index == 0, use_container_width=True):
        st.session_state.next_item = ids[index - 1]
        st.rerun()
    if right.button('Skip / next →', disabled=index == len(ids) - 1, use_container_width=True):
        st.session_state.next_item = ids[index + 1]
        st.rerun()
    st.caption('Saved labels are written directly to the evaluation CSVs. You can close the app and resume later.')


if __name__ == '__main__':
    main()
