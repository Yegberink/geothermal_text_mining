from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = APP_DIR / "sentences_for_annotation.csv"
DEFAULT_DB_PATH = APP_DIR / "databases" / "annotations.db"
ANNOTATORS = ["Egberink", "Dekker"]
SENTIMENTS = ["negative", "neutral", "positive"]

DATA_PATH = Path(os.environ.get("ANNOTATION_DATA_PATH", DEFAULT_DATA_PATH))
DB_URL = os.environ.get("DB_URL", f"sqlite:///{DEFAULT_DB_PATH}")
engine = create_engine(DB_URL, future=True)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Annotation CSV not found at {DATA_PATH}. Run scripts/assess_sentiment_models.py first."
        )
    df = pd.read_csv(DATA_PATH)
    required = {"sample_id", "sentence_text", "assignment_group"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in annotation CSV: {sorted(missing)}")
    return df


def init_db() -> None:
    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    sample_id TEXT PRIMARY KEY,
                    sentence_text TEXT NOT NULL,
                    assignment_group TEXT NOT NULL,
                    meta_json TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS annotations (
                    sample_id TEXT,
                    annotator TEXT,
                    gold_sentiment TEXT,
                    notes TEXT,
                    created_at TEXT,
                    PRIMARY KEY (sample_id, annotator)
                )
                """
            )
        )


def seed_tasks_if_empty(df: pd.DataFrame) -> None:
    with engine.begin() as conn:
        task_count = conn.execute(text("SELECT COUNT(*) FROM tasks")).scalar_one()
        incoming_ids = {str(v) for v in df["sample_id"].dropna().astype(str).tolist()}
        if task_count > 0:
            existing_ids = {str(r[0]) for r in conn.execute(text("SELECT sample_id FROM tasks")).fetchall()}
            if existing_ids == incoming_ids:
                return
            annotation_count = conn.execute(text("SELECT COUNT(*) FROM annotations")).scalar_one()
            if annotation_count == 0:
                conn.execute(text("DELETE FROM tasks"))
            else:
                return

        for _, row in df.iterrows():
            meta = {
                column: (None if pd.isna(row[column]) else row[column])
                for column in df.columns
                if column not in {"sample_id", "sentence_text", "assignment_group"}
            }
            conn.execute(
                text(
                    """
                    INSERT OR IGNORE INTO tasks(sample_id, sentence_text, assignment_group, meta_json)
                    VALUES(:sample_id, :sentence_text, :assignment_group, :meta_json)
                    """
                ),
                {
                    "sample_id": str(row["sample_id"]),
                    "sentence_text": str(row["sentence_text"]),
                    "assignment_group": str(row["assignment_group"]),
                    "meta_json": json.dumps(meta, ensure_ascii=False),
                },
            )


def assignment_where_clause(user: str) -> str:
    if user == "Dekker":
        return "t.assignment_group IN ('overlap', 'dekker_only')"
    if user == "Egberink":
        return "t.assignment_group IN ('overlap', 'egberink_only')"
    return "1=0"


def order_column_for(user: str) -> str:
    if user == "Dekker":
        return "annotation_order_dekker"
    return "annotation_order_egberink"


def next_unlabeled_task_uid(user: str):
    where_assign = assignment_where_clause(user)
    order_column = order_column_for(user)
    with engine.begin() as conn:
        row = conn.execute(
            text(
                f"""
                SELECT t.sample_id
                FROM tasks t
                LEFT JOIN annotations a
                  ON a.sample_id = t.sample_id AND a.annotator = :user
                WHERE a.sample_id IS NULL
                  AND {where_assign}
                ORDER BY json_extract(t.meta_json, '$.{order_column}') ASC, t.sample_id ASC
                LIMIT 1
                """
            ),
            {"user": user},
        ).first()
    return None if row is None else row[0]


def load_task(sample_id: str):
    with engine.begin() as conn:
        return conn.execute(
            text(
                """
                SELECT sample_id, sentence_text, assignment_group, meta_json
                FROM tasks
                WHERE sample_id = :sample_id
                """
            ),
            {"sample_id": str(sample_id)},
        ).first()


def save_annotation(sample_id: str, annotator: str, gold_sentiment: str, notes: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT OR REPLACE INTO annotations(sample_id, annotator, gold_sentiment, notes, created_at)
                VALUES(:sample_id, :annotator, :gold_sentiment, :notes, :created_at)
                """
            ),
            {
                "sample_id": str(sample_id),
                "annotator": annotator,
                "gold_sentiment": gold_sentiment,
                "notes": notes or None,
                "created_at": datetime.utcnow().isoformat(),
            },
        )


def get_progress(user: str):
    where_assign = assignment_where_clause(user)
    with engine.begin() as conn:
        assigned_total = conn.execute(text(f"SELECT COUNT(*) FROM tasks t WHERE {where_assign}")).scalar_one()
        done = conn.execute(
            text(
                f"""
                SELECT COUNT(*)
                FROM tasks t
                JOIN annotations a
                  ON a.sample_id = t.sample_id AND a.annotator = :user
                WHERE {where_assign}
                """
            ),
            {"user": user},
        ).scalar_one()
    remaining = max(0, assigned_total - done)
    frac = 0.0 if assigned_total == 0 else done / assigned_total
    return assigned_total, done, remaining, frac


st.set_page_config(page_title="Sentiment annotation", layout="wide")
st.title("Sentiment annotation")
st.caption("Sentence-level sentiment review with overlap between Dekker and Egberink.")

df = load_data()
init_db()
seed_tasks_if_empty(df)

user = st.selectbox("Annotator", ANNOTATORS)
assigned_total, done, remaining, frac = get_progress(user)
st.progress(frac)
st.write(f"**Progress ({user})**: {done}/{assigned_total} done • {remaining} remaining")

task_uid = next_unlabeled_task_uid(user)
if task_uid is None:
    st.success("You’re done. No remaining sentences assigned to you.")
    st.stop()

row = load_task(task_uid)
current_uid, sentence_text, assignment_group, meta_json = row
meta = json.loads(meta_json) if meta_json else {}

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader(f"Sentence ID: {meta.get('annotation_id', current_uid)}")
    meta_bits = [
        str(meta.get("source", "") or "").strip(),
        str(meta.get("document_title", "") or "").strip(),
        str(meta.get("publish_date", "") or "").strip(),
    ]
    meta_bits = [item for item in meta_bits if item]
    if meta_bits:
        st.caption(" | ".join(meta_bits))
    st.markdown("### Sentence")
    st.write(sentence_text)

    paragraph_text = str(meta.get("paragraph_text", "") or "").strip()
    if paragraph_text and paragraph_text != str(sentence_text).strip():
        with st.expander("Show paragraph context", expanded=True):
            st.write(paragraph_text)

    with st.expander("Metadata", expanded=False):
        st.markdown(f"- **Assignment group:** {assignment_group}")
        if meta.get("matched_categories_str"):
            st.markdown(f"- **Matched frame(s):** {meta.get('matched_categories_str')}")
        if meta.get("matched_keywords_str"):
            st.markdown(f"- **Matched keyword(s):** {meta.get('matched_keywords_str')}")

with right:
    st.subheader("Gold sentiment")
    with st.form(key=f"annotate_{current_uid}"):
        gold_sentiment = st.radio(
            "What is the correct sentiment of this sentence?",
            options=SENTIMENTS,
            horizontal=True,
        )
        notes = st.text_area(
            "Notes",
            height=120,
            placeholder="Optional note for ambiguity or difficult cases.",
        )
        submitted = st.form_submit_button("Save and continue", use_container_width=True)
        if submitted:
            save_annotation(
                sample_id=str(current_uid),
                annotator=user,
                gold_sentiment=gold_sentiment,
                notes=notes,
            )
            st.rerun()
