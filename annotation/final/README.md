# Annotation app

From the repository root:

```sh
pixi run streamlit run annotation/final/app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501).
The app opens `paragraphs.csv`, `sentences.csv` and `manifest.json` alongside it, regardless of your working directory. Generate these with `prepare_annotation.ipynb` if missing.

Work is divided between **Egberink** and **Dekker**. The 119 saved paragraphs are retained as **Together**; paragraph 120 had no saved labels when the split was made. Egberink has 191 remaining paragraphs and 250 sentences; Dekker has 190 remaining paragraphs and 250 sentences. Assignment is random with seed 42 and fixed once created. Splitting never rewrites the annotation CSVs or changes the evaluation sample or document splits. The original CSVs and manifest are backed up together under `backups/before_assignment_…/`.

For separate computers, use the prepared folders and ZIPs in `annotators/`. Each ZIP includes the app, its own annotation files, the joint reference labels, and full article context. Each copy has its annotator fixed, and changes are saved only in that copy. Send `annotators/Dekker.zip` to Dekker and keep the Egberink copy. Extract a ZIP before using it; see its README for setup on Windows, macOS, or Linux with Python 3.11 or later. With the existing project environment, start Egberink's copy from the project root:

```sh
pixi run streamlit run annotation/final/annotators/Egberink/app.py
```

In the master app, select your name to see your assignments. **Review annotations completed together** opens the preserved joint labels as read-only. Personal progress and navigation include only your own assigned items. Previously saved personal labels remain editable with **Previous**.

After working separately, each person can use **Download saved annotations → Download work for merging** to export a ZIP containing their latest CSVs and manifest. Extract the returned ZIPs, then merge each annotator folder into the master set:

```sh
pixi run python annotation/final/split_work.py merge /path/to/returned/Egberink
pixi run python annotation/final/split_work.py merge /path/to/returned/Dekker
```

The merge validates all sample IDs and source metadata, preserves joint labels, and imports only the returning person's assigned labels. Partial progress can be merged; identical repeated imports are harmless. A returned label that conflicts with an existing master label stops the merge for manual review. Both tasks are validated before either CSV is changed, and the master files are backed up before an import. The original evaluation commands below continue to use the combined master CSVs.

To prepare new portable copies explicitly, use `pixi run python annotation/final/split_work.py prepare --output /path/to/new/folder`. Existing assignments are reused, and existing packages are never overwritten. Keep working in the same extracted folder to retain your progress.

Choose paragraphs or sentences in the sidebar. The default queue starts at the first unfinished item. Read the text, complete the labels, and use **Save & next**. Labels have no default selection. Use the **Previous** and **Skip / next** buttons below the form to navigate. In the unfinished queue, **Previous** also visits saved items so you can correct a mistake immediately after **Save & next**; forward navigation continues to the next unfinished item. **Save** persists changes and keeps the current item open. When every item is saved, the current item stays open for corrections. Use **All** or **Completed** to browse saved work. Save before switching tasks, filters, or items: navigation discards unsaved edits.

For paragraphs, expand **Show full document** below the selected text to read the original article, with its title, source, and date. The article appears in a scrollable panel, and opening it preserves your unsaved form entries. Context is read from the original source recorded in the sampling manifest; no model output is loaded. If that source is missing or changed, the app explains why context is unavailable and annotation remains usable.

Location accepts one place name or `NONE`. Geothermal relevance is Yes/No. Sentiment is negative, neutral, positive, or “Not about geothermal energy.” The last option excludes an unrelated sentence from sentiment evaluation. The annotation guide is available beneath each text.

The app writes manual labels and notes directly into the two CSVs alongside it. The evaluator uses the master CSVs after merging separate copies. The app never displays model predictions or calibration/test assignments. CSV saves are atomic, retain all original metadata, and create timestamped copies of the previous CSV in `backups/`. Sessions using the same folder share one set of labels; conflicting edits to the same item are rejected. Use **Reload saved annotation** to discard your stale edit and load the current labels. Avoid editing the CSVs externally while the app is saving.

Downloads contain the saved CSVs. Backups, labels and sample data remain local and ignored by Git. No connection to the old annotation database or Snakemake is needed. The app uses `filelock` for locking on Windows, macOS, and Linux; it is included in the package requirements. Manifest text is always read and written as UTF-8.

After merging the returned copies, georeference the **combined master paragraphs**:

```sh
pixi run streamlit run annotation/final/georeference_app.py
```

The queue shows each **unique location name once**, ignoring differences in case and whitespace. Different spellings and aliases remain separate names. The map opens immediately, and the selected location name is prefilled in **Search place names**: press **Search**, then choose a result to place a marker and centre the map on it. Search accepts partial and alternative names and uses the local Dutch GeoNames file (`data/geonames/NL.zip`). Search results for different places sharing a name remain separate choices, labelled with their region and coordinates. You can also click the map, drag the marker, or enter coordinates in the optional expander. Check the displayed NUTS2 region and save. The region boundaries come from the same `data/shapes.parquet` as the workflow.

If no map point applies, enable **Use a fallback instead of a map point** and choose **Netherlands as a whole** or **Unresolved / outside Dutch scope** (with a reason). Unresolved judgments are counted and excluded from region scoring; they still receive exact-name scoring. Existing `NONE` text labels are handled automatically in evaluation and skipped in the map queue. There is no need to confirm the original location extraction again. Paragraph and document context remain available below the map if needed. The map tool never merges copies automatically.

Use **Save & next**, **Previous**, or the unique-name selector to review and correct judgments. Existing per-paragraph georeferences are retained and reused for every occurrence of their name, so previously completed names need no extra work. Opening the app never rewrites these records. Earlier judgments that disagree on region or fallback status are flagged for resolution; different points within the same region agree for this evaluation. A new save applies the selected judgment to every occurrence of that name in one atomic update to `georeferences.csv`, retaining sample IDs for evaluation. All previous records are backed up before a save, and conflicting edits to any occurrence are rejected.

Each judgment records the manual location, coordinates, region code and boundary-file hash. Changed boundaries invalidate old judgments. A newly added or changed location label can reuse a current judgment for the same name; names with no current judgment remain pending. Evaluation expands the shared judgment to every paragraph, preserving the original sample sizes and document splits. No model predictions are shown. Leading/trailing whitespace in manual location labels is stripped on loading, saving and merging.

The map uses [Leaflet](https://leafletjs.com/examples/quick-start/) with OpenStreetMap tiles and requires internet access to display the basemap; place-name search, coordinate entry and region assignment use local files. No additional Python dependencies are required. `FINAL_ANNOTATION_DIR` selects another master folder, `FINAL_ANNOTATION_GEONAMES` selects another Dutch GeoNames ZIP, and `FINAL_ANNOTATION_SHAPES` selects another shapes file; use the same shapes file with the evaluator's `--shapes-parquet` option.

After finishing the text labels and map judgments:

```sh
pixi run python scripts/model_evaluation/evaluate.py --validate-only
pixi run python scripts/model_evaluation/evaluate.py
```

The default evaluation reports both normalized exact-name matching (`location`) and workflow region matching (`location_region`). For text-only evaluation before map review, add `--location-matching exact`. `evaluation/location_comparison.csv` contains both matches for each model/paragraph; intermediate geocoding and its resource hashes are under `evaluation/geocoding/`.

See [the evaluation protocol](../../scripts/model_evaluation/README.md) for label definitions and scoring details.
