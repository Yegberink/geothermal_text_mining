# app.py
import os
import json
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

# --- Config ---
APP_DIR = Path(__file__).resolve().parent


def resolve_data_path() -> str:
    explicit = os.environ.get("ANNOTATION_DATA_PATH")
    if explicit:
        return explicit

    local_default = APP_DIR / "paragraphs_for_annotation.csv"
    if local_default.exists():
        return str(local_default)

    legacy_local = APP_DIR / "sentences_for_annotation.csv"
    if legacy_local.exists():
        return str(legacy_local)

    config_path = APP_DIR.parent / "config" / "config.yaml"
    if yaml is not None and config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        language = config.get("language")
        if language:
            candidate = APP_DIR / language / "paragraphs_for_annotation.csv"
            if candidate.exists():
                return str(candidate)
            legacy_candidate = APP_DIR / language / "sentences_for_annotation.csv"
            if legacy_candidate.exists():
                return str(legacy_candidate)

    language_dirs = sorted(
        p for p in APP_DIR.iterdir()
        if p.is_dir() and (
            (p / "paragraphs_for_annotation.csv").exists()
            or (p / "sentences_for_annotation.csv").exists()
        )
    )
    if len(language_dirs) == 1:
        paragraph_candidate = language_dirs[0] / "paragraphs_for_annotation.csv"
        if paragraph_candidate.exists():
            return str(paragraph_candidate)
        return str(language_dirs[0] / "sentences_for_annotation.csv")

    return str(local_default)


DATA_PATH = resolve_data_path()
DB_URL = os.environ.get("DB_URL", "sqlite:///annotations.db")
engine = create_engine(DB_URL, future=True)

SENTIMENTS = ["Positive", "Neutral", "Negative"]
ANNOTATORS = ["Egberink", "Dekker"]

# --- Balanced split + 10% overlap ---
OVERLAP_MOD = 20
OVERLAP_BUCKETS = {0, 1}


def stable_bucket(value) -> int:
    s = str(value).encode("utf-8")
    return zlib.crc32(s) % OVERLAP_MOD


def parse_listish(value):
    """
    Parse semicolon-separated strings like:
    'Knowledge availability;Costs'
    """
    if value is None:
        return []

    if isinstance(value, float) and pd.isna(value):
        return []

    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return []

    parts = [x.strip() for x in s.split(";")]
    return [x for x in parts if x]


def collect_all_categories_from_df(df: pd.DataFrame):
    if "matched_categories_str" not in df.columns:
        return []

    all_cats = set()
    for val in df["matched_categories_str"].dropna():
        for cat in parse_listish(val):
            all_cats.add(cat)

    return sorted(all_cats)


# --- DB setup ---
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
            sentiment_correct INTEGER,
            matched_categories_present INTEGER,
            matched_categories_correct INTEGER,
            matched_categories_true TEXT,
            keywords_to_add TEXT,
            notes TEXT,
            created_at TEXT,
            PRIMARY KEY (paragraph_uid, annotator)
        )"""))


def migrate_db():
    """
    Safe migration for older annotation DBs.
    Old unused columns can remain in place.
    """
    needed_cols = {
        "sentiment_correct": "INTEGER",
        "matched_categories_present": "INTEGER",
        "matched_categories_correct": "INTEGER",
        "matched_categories_true": "TEXT",
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

        existing = conn.execute(text("PRAGMA table_info(annotations)")).fetchall()
        existing_cols = {row[1] for row in existing}
        if "paragraph_uid" not in existing_cols and "sentence_id" in existing_cols:
            conn.execute(text("ALTER TABLE annotations RENAME COLUMN sentence_id TO paragraph_uid"))
            existing_cols.remove("sentence_id")
            existing_cols.add("paragraph_uid")

        for col, coltype in needed_cols.items():
            if col not in existing_cols:
                conn.execute(text(f"ALTER TABLE annotations ADD COLUMN {col} {coltype}"))


def on_annotator_change():
    st.session_state.paragraph_uid = None


def seed_tasks_if_empty(df: pd.DataFrame):
    """
    Seeds tasks once.

    Important:
    - Uses a paragraph-level uid column as the unique task key
    """
    uid_col = "paragraph_uid" if "paragraph_uid" in df.columns else ("uid" if "uid" in df.columns else None)
    text_col = "paragraph_text" if "paragraph_text" in df.columns else ("sentence_text" if "sentence_text" in df.columns else None)
    required = {uid_col, text_col, "sentiment"}
    required.discard(None)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    with engine.begin() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM tasks")).scalar_one()
        incoming_ids = {str(v) for v in df[uid_col].dropna().astype(str).tolist()}
        if n > 0:
            existing_ids = {
                str(row[0]) for row in conn.execute(text("SELECT paragraph_uid FROM tasks")).fetchall()
            }
            if existing_ids == incoming_ids:
                return

            annotation_count = conn.execute(text("SELECT COUNT(*) FROM annotations")).scalar_one()
            if annotation_count == 0:
                conn.execute(text("DELETE FROM tasks"))
            else:
                return

        for _, r in df.iterrows():
            paragraph_uid = r[uid_col]
            bucket = stable_bucket(paragraph_uid)

            meta = {
                k: (None if pd.isna(r[k]) else r[k])
                for k in df.columns
                if k not in [uid_col, text_col, "aspect", "sentiment"]
            }

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
                    paragraph_uid=str(paragraph_uid),
                    paragraph_text=str(r[text_col]),
                    aspect_pred=None if "aspect" not in df.columns or pd.isna(r.get("aspect")) else str(r["aspect"]),
                    sentiment_pred=None if pd.isna(r["sentiment"]) else str(r["sentiment"]),
                    split_bucket=int(bucket),
                    meta_json=json.dumps(meta, ensure_ascii=False),
                )
            )


def assignment_where_clause(user: str) -> str:
    overlap_list = ",".join(str(b) for b in sorted(OVERLAP_BUCKETS))

    if user not in ANNOTATORS:
        return "1=0"

    if user == "Dekker":
        return f"(t.split_bucket IN ({overlap_list}) OR (t.split_bucket >= 2 AND (t.split_bucket % 2) = 0))"
    else:
        return f"(t.split_bucket IN ({overlap_list}) OR (t.split_bucket >= 2 AND (t.split_bucket % 2) = 1))"


def next_unlabeled_paragraph_uid(user: str):
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
            dict(user=user)
        ).first()

    return None if row is None else row[0]


def load_task(paragraph_uid: str):
    with engine.begin() as conn:
        row = conn.execute(
            text("""
                SELECT paragraph_uid, paragraph_text, aspect_pred, sentiment_pred, meta_json, split_bucket
                FROM tasks
                WHERE paragraph_uid = :paragraph_uid
            """),
            dict(paragraph_uid=str(paragraph_uid))
        ).first()
    return row


def save_annotation(
    paragraph_uid: str,
    user: str,
    sent_ok: bool,
    matched_categories_present: bool,
    matched_categories_correct: bool,
    matched_categories_true: str,
    keywords_to_add: str,
    notes: str
):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT OR REPLACE INTO annotations
            (paragraph_uid, annotator, sentiment_correct,
             matched_categories_present, matched_categories_correct, matched_categories_true,
             keywords_to_add, notes, created_at)
            VALUES
            (:paragraph_uid, :annotator, :sentiment_correct,
             :matched_categories_present, :matched_categories_correct, :matched_categories_true,
             :keywords_to_add, :notes, :created_at)
        """), dict(
            paragraph_uid=str(paragraph_uid),
            annotator=user,
            sentiment_correct=1 if sent_ok else 0,
            matched_categories_present=1 if matched_categories_present else 0,
            matched_categories_correct=1 if matched_categories_correct else 0,
            matched_categories_true=matched_categories_true or None,
            keywords_to_add=keywords_to_add or None,
            notes=notes or None,
            created_at=datetime.utcnow().isoformat(),
        ))


def get_progress(user: str):
    where_assign = assignment_where_clause(user)
    with engine.begin() as conn:
        assigned_total = conn.execute(
            text(f"SELECT COUNT(*) FROM tasks t WHERE {where_assign}")
        ).scalar_one()

        done = conn.execute(
            text(f"""
                SELECT COUNT(*)
                FROM tasks t
                JOIN annotations a
                  ON a.paragraph_uid = t.paragraph_uid
                 AND a.annotator = :user
                WHERE {where_assign}
            """),
            dict(user=user)
        ).scalar_one()

    remaining = max(0, assigned_total - done)
    frac = 0.0 if assigned_total == 0 else done / assigned_total
    return assigned_total, done, remaining, frac


# --- App ---
st.set_page_config(page_title="Sentiment + category validation", layout="wide")
st.title("Review tool: sentiment + matched category validation")

init_db()
migrate_db()

df = pd.read_csv(DATA_PATH)
seed_tasks_if_empty(df)

ALL_CATEGORIES = collect_all_categories_from_df(df)

# Top bar
top_left, top_right = st.columns([2, 1], gap="large")

with top_right:
    user = st.selectbox(
        "Annotator",
        ANNOTATORS,
        key="annotator",
        on_change=on_annotator_change
    )

with top_left:
    assigned_total, done, remaining, frac = get_progress(user)
    overlap_pct = int(round(100 * len(OVERLAP_BUCKETS) / OVERLAP_MOD))
    st.caption(f"Assignment: ~{overlap_pct}% overlap; remaining items split evenly.")
    st.progress(frac)
    st.write(f"**Progress ({user})**: {done}/{assigned_total} done • {remaining} remaining")

# Find next task
if "paragraph_uid" not in st.session_state or st.session_state.paragraph_uid is None:
    st.session_state.paragraph_uid = next_unlabeled_paragraph_uid(user)

paragraph_uid = st.session_state.paragraph_uid
if paragraph_uid is None:
    st.success("You’re done — no remaining text units assigned to you.")
    st.stop()

row = load_task(paragraph_uid)
current_uid, paragraph_text, aspect_pred, sentiment_pred, meta_json, split_bucket = row

meta = json.loads(meta_json) if meta_json else {}
display_paragraph_id = meta.get("paragraph_id")
matched_categories_str = meta.get("matched_categories_str")
matched_keywords_str = meta.get("matched_keywords_str")

matched_categories = parse_listish(matched_categories_str)
matched_keywords = parse_listish(matched_keywords_str)
has_matched_category = len(matched_categories) > 0

# Two-column main layout
left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader(f"Paragraph UID: {current_uid}")
    if display_paragraph_id is not None:
        st.caption(f"Original paragraph_id: {display_paragraph_id}")

    st.write(paragraph_text)

    st.markdown("### Model output")
    st.markdown(f"- **Sentiment (pred):** {sentiment_pred}")
    st.markdown(
        f"- **Split bucket:** {split_bucket} " + ("(overlap)" if split_bucket in OVERLAP_BUCKETS else "")
    )

    st.markdown("### Category matching helper")
    if has_matched_category:
        st.markdown(f"- **Matched categories:** {'; '.join(matched_categories)}")
    else:
        st.markdown("- **Matched categories:** none")

    if matched_keywords:
        st.markdown(f"- **Matched keywords:** {'; '.join(matched_keywords)}")
    else:
        st.markdown("- **Matched keywords:** none")

    st.markdown(f"- **Available categories in selector:** {len(ALL_CATEGORIES)}")
    if ALL_CATEGORIES:
        with st.expander("Show all available categories", expanded=False):
            st.write(ALL_CATEGORIES)

    with st.expander("Metadata (optional)", expanded=False):
        if meta:
            st.json(meta)
        else:
            st.caption("No metadata stored for this paragraph.")

with right:
    st.subheader("Your evaluation")

    sent_ok = st.checkbox("Sentiment is correct", value=True, key=f"sent_ok_{current_uid}")

    sent_true = ""
    if not sent_ok:
        sent_true = st.radio(
            "Correct sentiment",
            SENTIMENTS,
            index=1,
            horizontal=True,
            key=f"sent_true_{current_uid}"
        )

    st.markdown("### Matched category check")

    if has_matched_category:
        matched_cat_ok = st.radio(
            "A matched category was found. Is that correct?",
            ["Yes", "No"],
            horizontal=True,
            key=f"matched_cat_ok_{current_uid}"
        )
        matched_categories_present = True
        matched_categories_correct = (matched_cat_ok == "Yes")
    else:
        no_match_ok = st.radio(
            "No matched category was found. Is that correct?",
            ["Yes", "No"],
            horizontal=True,
            key=f"no_match_ok_{current_uid}"
        )
        matched_categories_present = False
        matched_categories_correct = (no_match_ok == "Yes")

    matched_categories_true = ""
    keywords_to_add = ""

    if not matched_categories_correct:
        st.caption("Select the correct category/categories from the predefined list.")

        selected_categories = st.multiselect(
            "Correct category/categories",
            options=ALL_CATEGORIES,
            default=[],
            key=f"correct_categories_{current_uid}",
            help="Choose one or more existing categories."
        )
        matched_categories_true = ";".join(selected_categories)

        keywords_to_add = st.text_area(
            "Keyword(s) that should be added to the keywords list",
            height=80,
            key=f"keywords_to_add_{current_uid}",
            placeholder="e.g. drilling;water pollution;seismic risk"
        )

    notes = st.text_area(
        "Notes (optional)",
        height=90,
        key=f"notes_{current_uid}",
        placeholder="Optional notes, including corrected sentiment if needed."
    )

    if not sent_ok and sent_true:
        notes_prefix = f"[Correct sentiment: {sent_true}]"
    else:
        notes_prefix = ""

    notes_to_save = f"{notes_prefix}\n{notes}".strip() if notes_prefix else notes

    b1, b2 = st.columns(2)

    with b1:
        if st.button("Save", use_container_width=True, key=f"save_{current_uid}"):
            save_annotation(
                current_uid,
                user,
                sent_ok,
                matched_categories_present,
                matched_categories_correct,
                matched_categories_true,
                keywords_to_add,
                notes_to_save
            )
            st.success("Saved.")

    with b2:
        if st.button("Save & Next ➜", use_container_width=True, key=f"save_next_{current_uid}"):
            save_annotation(
                current_uid,
                user,
                sent_ok,
                matched_categories_present,
                matched_categories_correct,
                matched_categories_true,
                keywords_to_add,
                notes_to_save
            )
            st.session_state.paragraph_uid = next_unlabeled_paragraph_uid(user)
            st.rerun()
