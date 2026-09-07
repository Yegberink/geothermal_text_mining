# Annotation app

From the repository root:

```sh
pixi run streamlit run annotation/final/app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501).
The app opens `paragraphs.csv`, `sentences.csv` and `manifest.json` alongside it, regardless of your working directory. Generate these with `prepare_annotation.ipynb` if missing.

Choose paragraphs or sentences in the sidebar. The default queue shows unfinished items. Read the text, complete the labels, and use **Save & next**. Labels have no default selection. Use **All** or **Completed** to revisit saved work, or jump to an item with **Go to item**. Save before switching tasks, filters, or items: navigation discards unsaved edits. **Save** persists the current item; in the unfinished queue it then disappears because it is complete.

For paragraphs, expand **Show full document** below the selected text to read the original article, with its title, source, and date. The article appears in a scrollable panel, and opening it preserves your unsaved form entries. Context is read from the original source recorded in the sampling manifest; no model output is loaded. If that source is missing or changed, the app explains why context is unavailable and annotation remains usable.

Location accepts one place name or `NONE`. Geothermal relevance is Yes/No. Sentiment is negative, neutral, positive, or “Not about geothermal energy.” The last option excludes an unrelated sentence from sentiment evaluation. The annotation guide is available beneath each text.

The app writes manual labels and notes directly into the two CSVs consumed by the evaluator. It never displays model predictions or calibration/test assignments. CSV saves are atomic, retain all original metadata, and create timestamped copies of the previous CSV in `backups/`. App sessions share one set of labels; conflicting edits to the same item are rejected. Use **Reload saved annotation** to discard your stale edit and load the current labels. Avoid editing the CSVs externally while the app is saving.

Downloads contain the saved CSVs. Backups, labels and sample data remain local and ignored by Git. No connection to the old annotation database or Snakemake is needed. This local app uses file locking available on macOS/Linux.

After finishing both tasks:

```sh
pixi run python scripts/model_evaluation/evaluate.py --validate-only
pixi run python scripts/model_evaluation/evaluate.py
```

See [the evaluation protocol](../../scripts/model_evaluation/README.md) for label definitions and scoring details.
