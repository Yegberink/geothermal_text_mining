import json
import os
import zlib
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

APP_DIR = Path(__file__).resolve().parent
DB_URL = os.environ.get("DB_URL", "sqlite:///annotations.db")
engine = create_engine(DB_URL, future=True)

SENTIMENTS = ["positive", "neutral", "negative"]
ANNOTATORS = ["Egberink", "Dekker"]
OVERLAP_MOD = 20
OVERLAP_BUCKETS = {0, 1}


def resolve_data_path() -> str:
    explicit = os.environ.get("ANNOTATION_DATA_PATH")
    if explicit:
        return explicit

    local_default = APP_DIR / "sentences_for_annotation.csv"
    if local_default.exists():
        return str(local_default)

    legacy_local = APP_DIR / "paragraphs_for_annotation.csv"
    if legacy_local.exists():
        return str(legacy_local)

    config_path = APP_DIR.parent / "config" / "config.yaml"
    if yaml is not None and config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        language = config.get("language")
        if language:
            for name in ["sentences_for_annotation.csv", "paragraphs_for_annotation.csv"]:
                candidate = APP_DIR / language / name
                if candidate.exists():
                    return str(candidate)

    return str(local_default)


DATA_PATH = resolve_data_path()


def stable_bucket(value) -> int:
    return zlib.crc32(str(value).encode("utf-8")) % OVERLAP_MOD


def parse_listish(value):
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return []
    return [x.strip() for x in s.split(";") if x.strip()]


def collect_all_categories_from_df(df: pd.DataFrame):
    all_cats = set()
    if "matched_categories_str" not in df.columns:
        return []
    for val in df["matched_categories_str"].dropna():
        all_cats.update(parse_listish(val))
    return sorted(all_cats)


def normalize_sentiment_value(value):
    mapping = {
        "positive": "positive",
        "pos": "positive",
        "negative": "negative",
        "neg": "negative",
        "neutral": "neutral",
        "neu": "neutral",
        "neutral/uncertain": "neutral",
    }
    return mapping.get(str(value).strip().lower(), "")


def init_db():
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS tasks (
            paragraph_uid TEXT PRIMARY KEY,
            paragraph_text TEXT NOT NULL,
            aspect_pred TEXT,
            sentiment_pred TEXT,
            split_bucket INTEGER NOT NULL,
            meta_json TEXT
        )"""))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS annotations (
            paragraph_uid TEXT,
            annotator TEXT,
            geothermal_relevant INTEGER,
            sentiment_correct INTEGER,
            sentiment_true TEXT,
            matched_categories_present INTEGER,
            matched_categories_correct INTEGER,
            matched_categories_true TEXT,
            location_correct INTEGER,
            location_true TEXT,
            keywords_to_add TEXT,
            notes TEXT,
            created_at TEXT,
            PRIMARY KEY (paragraph_uid, annotator)
        )"""))


def migrate_db():
    needed_cols = {
        "geothermal_relevant": "INTEGER",
        "sentiment_correct": "INTEGER",
        "sentiment_true": "TEXT",
        "matched_categories_present": "INTEGER",
        "matched_categories_correct": "INTEGER",
        "matched_categories_true": "TEXT",
        "location_correct": "INTEGER",
        "location_true": "TEXT",
        "keywords_to_add": "TEXT",
        "notes": "TEXT",
        "created_at": "TEXT",
    }
    with engine.begin() as conn:
        task_existing = conn.execute(text("PRAGMA table_info(tasks)")).fetchall()
        task_existing_cols = {row[1] for row in task_existing}
        if "paragraph_uid" not in task_existing_cols and "sentence_id" in task_existing_cols:
            conn.execute(text("ALTER TABLE tasks RENAME COLUMN sentence_id TO paragraph_uid"))
        if "paragraph_text" not in task_existing_cols and "sentence_text" in task_existing_cols:
            conn.execute(text("ALTER TABLE tasks RENAME COLUMN sentence_text TO paragraph_text"))

        ann_existing = conn.execute(text("PRAGMA table_info(annotations)")).fetchall()
        ann_existing_cols = {row[1] for row in ann_existing}
        if "paragraph_uid" not in ann_existing_cols and "sentence_id" in ann_existing_cols:
            conn.execute(text("ALTER TABLE annotations RENAME COLUMN sentence_id TO paragraph_uid"))
            ann_existing_cols.remove("sentence_id")
            ann_existing_cols.add("paragraph_uid")
        for col, coltype in needed_cols.items():
            if col not in ann_existing_cols:
                conn.execute(text(f"ALTER TABLE annotations ADD COLUMN {col} {coltype}"))


def on_annotator_change():
    st.session_state.task_uid = None


def seed_tasks_if_empty(df: pd.DataFrame):
    uid_col = "sentence_uid" if "sentence_uid" in df.columns else ("paragraph_uid" if "paragraph_uid" in df.columns else ("uid" if "uid" in df.columns else None))
    text_col = "sentence_text" if "sentence_text" in df.columns else "paragraph_text"
    if uid_col is None or text_col not in df.columns:
        raise ValueError("Annotation CSV must include a sentence or paragraph uid and text column.")

    required = {uid_col, text_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    with engine.begin() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM tasks")).scalar_one()
        incoming_ids = {str(v) for v in df[uid_col].dropna().astype(str).tolist()}
        if n > 0:
            existing_ids = {str(r[0]) for r in conn.execute(text("SELECT paragraph_uid FROM tasks")).fetchall()}
            if existing_ids == incoming_ids:
                return
            annotation_count = conn.execute(text("SELECT COUNT(*) FROM annotations")).scalar_one()
            if annotation_count == 0:
                conn.execute(text("DELETE FROM tasks"))
            else:
                return

        for _, r in df.iterrows():
            task_uid = str(r[uid_col])
            bucket = stable_bucket(task_uid)
            predicted_frames = parse_listish(r.get("matched_categories_str"))
            meta = {
                k: (None if pd.isna(r[k]) else r[k])
                for k in df.columns
                if k not in {uid_col, text_col, "sentiment"}
            }
            meta["task_uid_col"] = uid_col
            meta["task_text_col"] = text_col
            meta["predicted_frames"] = predicted_frames
            meta["predicted_location"] = r.get("llm_location")
            meta["matched_location"] = r.get("geo_name_matched")
            meta["paragraph_text"] = r.get("paragraph_text")
            meta["sentence_text"] = r.get("sentence_text")

            conn.execute(
                text("""
                    INSERT OR IGNORE INTO tasks(
                        paragraph_uid, paragraph_text, aspect_pred, sentiment_pred, split_bucket, meta_json
                    )
                    VALUES(
                        :paragraph_uid, :paragraph_text, :aspect_pred, :sentiment_pred, :split_bucket, :meta_json
                    )
                """),
                dict(
                    paragraph_uid=task_uid,
                    paragraph_text=str(r[text_col]),
                    aspect_pred=";".join(predicted_frames) if predicted_frames else None,
                    sentiment_pred=None if pd.isna(r.get("sentiment")) else str(r.get("sentiment")),
                    split_bucket=int(bucket),
                    meta_json=json.dumps(meta, ensure_ascii=False),
                ),
            )


def assignment_where_clause(user: str) -> str:
    overlap_list = ",".join(str(b) for b in sorted(OVERLAP_BUCKETS))
    if user not in ANNOTATORS:
        return "1=0"
    if user == "Dekker":
        return f"(t.split_bucket IN ({overlap_list}) OR (t.split_bucket >= 2 AND (t.split_bucket % 2) = 0))"
    return f"(t.split_bucket IN ({overlap_list}) OR (t.split_bucket >= 2 AND (t.split_bucket % 2) = 1))"


def next_unlabeled_task_uid(user: str):
    where_assign = assignment_where_clause(user)
    with engine.begin() as conn:
        row = conn.execute(
            text(f"""
                SELECT t.paragraph_uid
                FROM tasks t
                LEFT JOIN annotations a
                  ON a.paragraph_uid = t.paragraph_uid AND a.annotator = :user
                WHERE a.paragraph_uid IS NULL
                  AND {where_assign}
                ORDER BY t.paragraph_uid
                LIMIT 1
            """),
            dict(user=user),
        ).first()
    return None if row is None else row[0]


def load_task(task_uid: str):
    with engine.begin() as conn:
        return conn.execute(
            text("""
                SELECT paragraph_uid, paragraph_text, aspect_pred, sentiment_pred, meta_json, split_bucket
                FROM tasks
                WHERE paragraph_uid = :paragraph_uid
            """),
            dict(paragraph_uid=str(task_uid)),
        ).first()


def save_annotation(
    task_uid: str,
    user: str,
    geothermal_relevant: bool,
    sentiment_correct: bool | None,
    sentiment_true: str,
    matched_categories_present: bool,
    matched_categories_correct: bool | None,
    matched_categories_true: str,
    location_correct: bool | None,
    location_true: str,
    keywords_to_add: str,
    notes: str,
):
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT OR REPLACE INTO annotations
                (paragraph_uid, annotator, geothermal_relevant, sentiment_correct, sentiment_true,
                 matched_categories_present, matched_categories_correct, matched_categories_true,
                 location_correct, location_true, keywords_to_add, notes, created_at)
                VALUES
                (:paragraph_uid, :annotator, :geothermal_relevant, :sentiment_correct, :sentiment_true,
                 :matched_categories_present, :matched_categories_correct, :matched_categories_true,
                 :location_correct, :location_true, :keywords_to_add, :notes, :created_at)
            """),
            dict(
                paragraph_uid=str(task_uid),
                annotator=user,
                geothermal_relevant=1 if geothermal_relevant else 0,
                sentiment_correct=None if sentiment_correct is None else (1 if sentiment_correct else 0),
                sentiment_true=sentiment_true or None,
                matched_categories_present=1 if matched_categories_present else 0,
                matched_categories_correct=None if matched_categories_correct is None else (1 if matched_categories_correct else 0),
                matched_categories_true=matched_categories_true or None,
                location_correct=None if location_correct is None else (1 if location_correct else 0),
                location_true=location_true or None,
                keywords_to_add=keywords_to_add or None,
                notes=notes or None,
                created_at=datetime.utcnow().isoformat(),
            ),
        )


def get_progress(user: str):
    where_assign = assignment_where_clause(user)
    with engine.begin() as conn:
        assigned_total = conn.execute(text(f"SELECT COUNT(*) FROM tasks t WHERE {where_assign}")).scalar_one()
        done = conn.execute(
            text(f"""
                SELECT COUNT(*)
                FROM tasks t
                JOIN annotations a
                  ON a.paragraph_uid = t.paragraph_uid
                 AND a.annotator = :user
                WHERE {where_assign}
            """),
            dict(user=user),
        ).scalar_one()
    remaining = max(0, assigned_total - done)
    frac = 0.0 if assigned_total == 0 else done / assigned_total
    return assigned_total, done, remaining, frac


def review_state_key(task_uid: str, name: str) -> str:
    return f"review_{task_uid}_{name}"


def get_review_state(task_uid: str, name: str, default=None):
    return st.session_state.get(review_state_key(task_uid, name), default)


def set_review_state(task_uid: str, name: str, value) -> None:
    st.session_state[review_state_key(task_uid, name)] = value


def clear_review_state(task_uid: str) -> None:
    prefix = f"review_{task_uid}_"
    for key in list(st.session_state.keys()):
        if key.startswith(prefix):
            del st.session_state[key]


def answer_button_row(task_uid: str, field_name: str, prompt: str, options: list[tuple[str, object]]) -> None:
    st.markdown(f"**{prompt}**")
    cols = st.columns(len(options))
    for col, (label, value) in zip(cols, options):
        if col.button(label, key=review_state_key(task_uid, f"{field_name}_{label}"), use_container_width=True):
            set_review_state(task_uid, field_name, value)
            st.rerun()


st.set_page_config(page_title="Sentence annotation review", layout="wide")
st.title("Review tool: geothermal, frame, sentiment, and location validation")

init_db()
migrate_db()

df = pd.read_csv(DATA_PATH)
seed_tasks_if_empty(df)
ALL_CATEGORIES = collect_all_categories_from_df(df)

top_left, top_right = st.columns([2, 1], gap="large")

with top_right:
    user = st.selectbox("Annotator", ANNOTATORS, key="annotator", on_change=on_annotator_change)

with top_left:
    assigned_total, done, remaining, frac = get_progress(user)
    overlap_pct = int(round(100 * len(OVERLAP_BUCKETS) / OVERLAP_MOD))
    st.caption(f"Assignment: ~{overlap_pct}% overlap; remaining items split evenly.")
    st.progress(frac)
    st.write(f"**Progress ({user})**: {done}/{assigned_total} done • {remaining} remaining")

if "task_uid" not in st.session_state or st.session_state.task_uid is None:
    st.session_state.task_uid = next_unlabeled_task_uid(user)

task_uid = st.session_state.task_uid
if task_uid is None:
    st.success("You’re done. No remaining text units assigned to you.")
    st.stop()

row = load_task(task_uid)
current_uid, task_text, aspect_pred, sentiment_pred, meta_json, split_bucket = row
meta = json.loads(meta_json) if meta_json else {}

sentence_text = meta.get("sentence_text") or task_text
paragraph_text = meta.get("paragraph_text") or ""
annotation_id = meta.get("annotation_id")
predicted_frames = parse_listish(meta.get("matched_categories_str") or aspect_pred)
predicted_keywords = parse_listish(meta.get("matched_keywords_str"))
predicted_location = meta.get("predicted_location") or meta.get("llm_location")
matched_location = meta.get("matched_location") or meta.get("geo_name_matched")
display_location = matched_location or predicted_location or "None"

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader(f"Sentence UID: {current_uid}")
    if annotation_id is not None:
        st.caption(f"Annotation row: {annotation_id}")
    st.markdown("### Sentence")
    st.write(sentence_text)
    if paragraph_text and paragraph_text != sentence_text:
        with st.expander("Show paragraph context", expanded=True):
            st.write(paragraph_text)

    st.markdown("### Model output")
    st.markdown(f"- **Predicted sentiment:** {sentiment_pred or 'n/a'}")
    st.markdown(f"- **Predicted frame(s):** {'; '.join(predicted_frames) if predicted_frames else 'none'}")
    st.markdown(f"- **Matched keyword(s):** {'; '.join(predicted_keywords) if predicted_keywords else 'none'}")
    st.markdown(f"- **Extracted location from paragraph:** {predicted_location or 'none'}")
    st.markdown(f"- **Matched geocoded location:** {matched_location or 'none'}")
    st.markdown(f"- **Split bucket:** {split_bucket} " + ("(overlap)" if split_bucket in OVERLAP_BUCKETS else ""))

    with st.expander("Metadata", expanded=False):
        st.json(meta)

with right:
    st.subheader("Your evaluation")
    if st.button("Reset Current Review", use_container_width=True, key=f"reset_{current_uid}"):
        clear_review_state(current_uid)
        st.rerun()

    geothermal_relevant = get_review_state(current_uid, "geothermal_relevant")
    sentiment_correct = get_review_state(current_uid, "sentiment_correct")
    sentiment_true = get_review_state(current_uid, "sentiment_true", "")
    matched_categories_correct = get_review_state(current_uid, "matched_categories_correct")
    matched_categories_true = get_review_state(current_uid, "matched_categories_true", "")
    keywords_to_add = get_review_state(current_uid, "keywords_to_add", "")
    location_choice = get_review_state(current_uid, "location_choice")
    location_true = get_review_state(current_uid, "location_true", "")
    notes = st.session_state.get(review_state_key(current_uid, "notes"), "")

    matched_categories_present = bool(predicted_frames)
    form_complete = False

    if geothermal_relevant is None:
        answer_button_row(
            current_uid,
            "geothermal_relevant",
            "1. Is this text about geothermal?",
            [("Yes", True), ("No", False)],
        )
    else:
        st.caption(f"1. Geothermal: {'Yes' if geothermal_relevant else 'No'}")

    if geothermal_relevant is True:
        if sentiment_correct is None:
            answer_button_row(
                current_uid,
                "sentiment_correct",
                f"2. Is the sentiment correct? Predicted sentiment: {sentiment_pred or 'none'}",
                [("Yes", True), ("No", False)],
            )
        else:
            st.caption(f"2. Sentiment correct: {'Yes' if sentiment_correct else 'No'}")

        if sentiment_correct is False and not sentiment_true:
            answer_button_row(
                current_uid,
                "sentiment_true",
                "Correct sentiment",
                [(label.capitalize(), label) for label in SENTIMENTS],
            )
        elif sentiment_correct is False and sentiment_true:
            st.caption(f"Correct sentiment: {sentiment_true}")

        sentiment_step_complete = sentiment_correct is True or bool(sentiment_true)

        if sentiment_step_complete:
            if matched_categories_correct is None:
                answer_button_row(
                    current_uid,
                    "matched_categories_correct",
                    (
                        f"3. Are the predicted frame(s) correct? Predicted frame(s): {'; '.join(predicted_frames)}"
                        if predicted_frames
                        else "3. No frame was predicted. Is that correct?"
                    ),
                    [("Yes", True), ("No", False)],
                )
            else:
                st.caption(f"3. Frame correct: {'Yes' if matched_categories_correct else 'No'}")

        if matched_categories_correct is False and not matched_categories_true:
            default_categories = [
                cat
                for cat in parse_listish(st.session_state.get(review_state_key(current_uid, "frame_selection"), ""))
                if cat in ALL_CATEGORIES
            ]
            selected_categories = st.multiselect(
                "Correct frame(s)",
                options=ALL_CATEGORIES,
                default=default_categories,
                key=review_state_key(current_uid, "frame_selection"),
            )
            keyword_value = st.text_area(
                "Keyword(s) to add to the frame list",
                height=80,
                key=review_state_key(current_uid, "frame_keywords_input"),
                placeholder="e.g. vergunning;subsidie;aardbeving",
            )
            if st.button("Confirm Frame Correction", use_container_width=True, key=f"confirm_frame_{current_uid}"):
                set_review_state(current_uid, "matched_categories_true", ";".join(selected_categories))
                set_review_state(current_uid, "keywords_to_add", keyword_value)
                st.rerun()
        elif matched_categories_correct is False and matched_categories_true:
            st.caption(f"Correct frame(s): {matched_categories_true}")

        frame_step_complete = matched_categories_correct is True or bool(matched_categories_true)
        next_location_number = 4
    elif geothermal_relevant is False:
        sentiment_step_complete = True
        frame_step_complete = True
        next_location_number = 2
    else:
        sentiment_step_complete = False
        frame_step_complete = False
        next_location_number = 0

    if geothermal_relevant is not None and sentiment_step_complete and frame_step_complete:
        if location_choice is None:
            answer_button_row(
                current_uid,
                "location_choice",
                f"{next_location_number}. Is the extracted paragraph location correct? Predicted location: {predicted_location or 'none'}",
                [("Yes", "yes"), ("No", "no"), ("No Location", "na")],
            )
        else:
            location_labels = {"yes": "Yes", "no": "No", "na": "No location available"}
            st.caption(f"{next_location_number}. Location correct: {location_labels.get(location_choice, location_choice)}")

    if location_choice == "no" and not location_true:
        entered_location = st.text_input(
            "Correct location",
            key=review_state_key(current_uid, "location_input"),
            placeholder="e.g. Westland; Zuid-Holland; Nederland",
        )
        if st.button("Confirm Location Correction", use_container_width=True, key=f"confirm_location_{current_uid}"):
            if entered_location.strip():
                set_review_state(current_uid, "location_true", entered_location.strip())
                st.rerun()
    elif location_choice == "no" and location_true:
        st.caption(f"Correct location: {location_true}")

    location_complete = location_choice in {"yes", "na"} or (location_choice == "no" and bool(location_true))

    if geothermal_relevant is not None and sentiment_step_complete and frame_step_complete and location_complete:
        notes_number = 5 if geothermal_relevant else 3
        st.markdown(f"**{notes_number}. Notes**")
        notes = st.text_area(
            "Optional notes",
            height=120,
            key=review_state_key(current_uid, "notes"),
            placeholder="Optional notes on geothermal relevance, frame, sentiment, or location.",
            label_visibility="collapsed",
        )
        form_complete = True

    if form_complete:
        location_correct = True if location_choice == "yes" else (False if location_choice == "no" else None)
        save_payload = dict(
            task_uid=current_uid,
            user=user,
            geothermal_relevant=geothermal_relevant,
            sentiment_correct=sentiment_correct,
            sentiment_true=sentiment_true,
            matched_categories_present=matched_categories_present,
            matched_categories_correct=matched_categories_correct,
            matched_categories_true=matched_categories_true,
            location_correct=location_correct,
            location_true=location_true,
            keywords_to_add=keywords_to_add,
            notes=notes,
        )

        b1, b2 = st.columns(2)
        with b1:
            if st.button("Save", use_container_width=True, key=f"save_{current_uid}"):
                save_annotation(**save_payload)
                st.success("Saved.")
        with b2:
            if st.button("Save & Next", use_container_width=True, key=f"save_next_{current_uid}"):
                save_annotation(**save_payload)
                clear_review_state(current_uid)
                st.session_state.task_uid = next_unlabeled_task_uid(user)
                st.rerun()
