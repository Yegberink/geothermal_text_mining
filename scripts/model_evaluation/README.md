# Manual Dutch model evaluation

Open `annotation/final/prepare_annotation.ipynb` and run it once. It samples 500 distinct Dutch paragraphs uniformly from the cleaned paragraph corpus, before keyword/model filtering, and 500 distinct sentences uniformly from current Dutch sentiment results. The corpus itself may reflect the original newspaper retrieval criteria. No model predictions, confidence values, keyword hits, geographic matches, or rationales enter the annotation CSVs. Sampling uses seed 42 and refuses to overwrite existing files. This is independent of Snakemake: it only reads existing source files and never changes workflow results or caches.

Launch the annotation interface with `pixi run streamlit run annotation/final/app.py`. It saves labels directly to the CSVs below, tracks progress, and supports revisiting completed items. See [app instructions](../../annotation/final/README.md).

Edit `annotation/final/paragraphs.csv`:

- `gold_location`: ONE primary geographic location, using the most specific clearly supported place name. Write `NONE` if there is no clear primary location. Annotate every paragraph, including those unrelated to geothermal energy. Use consistent Dutch place names. Exact-name scoring normalizes case, Unicode, whitespace, and curly apostrophes without inferring aliases; region scoring separately geocodes model outputs through the workflow. Manual strings are stripped of leading/trailing whitespace when read, saved and merged.
- `gold_geothermal`: `YES` if geothermal energy is the main or substantial subject; `NO` if absent, incidental, or another subject dominates. Resolve uncertain cases manually and record decisions in `notes`.

Edit `annotation/final/sentences.csv`:

- `gold_sentiment`: `negative`, `neutral`, or `positive`. Negative emphasizes risks, costs, criticism, uncertainty, conflict, harm, or failure; positive emphasizes benefits, support, progress, feasibility, opportunity, or success; neutral is factual, procedural, descriptive, or mixed without clear polarity. Judge the sentence itself, matching the workflow input.
- If a sampled sentence is actually unrelated to geothermal energy, use `not_geothermal`. Such rows are counted in the annotation file but excluded from sentiment evaluation; do not silently replace them. Membership in a geothermal paragraph does not guarantee sentence-level relevance.

Only edit `gold_*` and `notes`. The evaluator checks all original IDs, texts, and metadata against the manifest and requires complete labels. The manifest retains the original source hashes and historical document split for provenance. Current scoring pools all eligible annotated examples; the historical calibration/test split is not used to select thresholds or calculate scores.

After merging the returned annotator copies into the master set, use `pixi run streamlit run annotation/final/georeference_app.py` to georeference each unique location name once on an OpenStreetMap map. Names are grouped by Unicode-normalized spelling, ignoring case and whitespace; aliases are not inferred. The map opens by default, with place-name search using the local Dutch GeoNames data. Select a search result to centre the map and place a marker, or click the map directly. This creates `georeferences.csv` independently of model predictions. Choose a point inside the appropriate Dutch NUTS2 region (province); the only fallbacks are the whole Netherlands or explicitly unresolved/outside scope with a reason. For a province-level location, choose an interior point of that province. Each name shares one judgment across all occurrences. Existing per-paragraph work is preserved and reused for repeated names without rewriting files; conflicting saved regions or fallback statuses must be resolved in the app. A save updates all occurrences atomically, with backups and checks for concurrent edits. Evaluation expands these shared judgments back to individual paragraphs, so sample weighting and calibration/test splits remain unchanged. Existing `NONE` labels are skipped in the map queue and evaluated automatically as no location. Every named location needs a current judgment; changed boundaries invalidate old judgments, while newly added occurrences can reuse an existing judgment for the same name. See [map instructions](../../annotation/final/README.md).

From the repository root:

```sh
pixi run python scripts/model_evaluation/evaluate.py --validate-only
pixi run python scripts/model_evaluation/evaluate.py --results-dir annotation/final/evaluation_all_models
# Optional smaller run / Apple GPU for specialist models:
pixi run python scripts/model_evaluation/evaluate.py --models qwen2.5:7b --results-dir annotation/final/qwen7
pixi run python scripts/model_evaluation/evaluate.py --device mps --results-dir annotation/final/mps
# Before manual map review, evaluate exact names only:
pixi run python scripts/model_evaluation/evaluate.py --location-matching exact
```

The default evaluation includes 20 models. Nine Ollama models run location extraction, geothermal classification, and sentiment: the original `qwen2.5:14b`, `qwen2.5:7b`, `llama3.1:8b`, `mistral`, and `phi3`, plus `qwen3.5:9b`, `ministral-3:14b`, `gemma4:12b`, and `qwen3.5:4b`. The existing eleven Hugging Face specialists run sentiment only.

Start Ollama and install the local models before evaluation:

```sh
for model in qwen2.5:14b qwen2.5:7b llama3.1:8b mistral phi3 qwen3.5:9b ministral-3:14b gemma4:12b qwen3.5:4b; do
  ollama pull "$model" || break
done
```

The evaluator checks selected Ollama models before inference and exits with installation commands if any are missing. `--validate-only` checks annotations without contacting Ollama. Progress reports show each model/task starting, cached sample counts, specialist loading, and each completed sample with elapsed time or an error. A single request can still take time, including the workflow's retries.

Use a new results directory when switching from the previous model set: existing prediction logs deliberately reject changed code or configurations. The command above uses `annotation/final/evaluation_all_models`; repeating that command resumes successful predictions and retries failed ones. Hugging Face weights download on first use. Existing Pixi dependencies provide pandas, numpy, torch, transformers, requests, tenacity, and the workflow helper imports. Model IDs, pinned revisions and explicit label mappings are in `models.json`.

Ollama calls directly reuse the workflow functions, prompts, system messages, generation options, response parsers and retry behavior. The evaluator additionally sets the top-level API field `think: false` for Qwen3.5 and Gemma4, as recorded in `models.json` and `run.json`. This reserves their short 120–200 token output budgets for the final answer. Other evaluated models omit this field. The shared workflow functions accept an optional keyword argument; the production workflow defaults to `ministral-3:14b` with thinking disabled. See [Ollama thinking controls](https://docs.ollama.com/capabilities/thinking). Sentiment always selects `zero_shot`; no examples are supplied. Location evaluation first extracts each sampled paragraph independently, then reuses the production `apply_document_location_fallback`: missing or country-level locations can inherit the sole specific location found in another sampled paragraph from the same article; otherwise the model reads the full article, followed by the workflow country fallback when necessary. Full article text comes from the original corpus recorded in the sampling manifest, with its SHA-256 verified before inference. Only sampled paragraphs supply peer predictions; unsampled paragraphs remain available as full article context. No manual relevance/location labels or pre-existing workflow predictions enter this process. There is no relevance gate, so the fixed evaluation sample and both subsets remain comparable. This measures extraction plus fallback, while the manual target remains the annotated paragraph location; a country fallback is not automatically a correct answer. The extracted location and granularity are then passed through the production geocoding entry points for region scoring. The unchanged location prompt assumes geothermal relevance, so results include both all paragraphs and the subset manually labeled geothermal. Geothermal `MAYBE` counts as an incorrect binary prediction and cannot pass a confidence threshold. Workflow parser fallbacks (including confidence-zero outputs) remain part of the measured behavior.

Region processing runs `geocoding_offline.py` followed by `geocoding_online.py` (existing overrides, GeoNames and geocoder-cache matches), then the workflow's geometry selection and `assign_nuts2`. The original extracted spelling and granularity are preserved for these calls, including case-sensitive cache lookup. Defaults match the Dutch workflow: `data/shapes.parquet`, `data/geonames/NL.zip`, `data/vocab/dutch/location_geocoding_overrides.csv`, and `cache/geocode_unmatched_online.jsonl`, with cache country codes `nl,be,bq,aw,cw,sx`. Resource paths and cache scope can be set with `--shapes-parquet`, `--geonames-dir`, `--geocoding-overrides`, `--geocoder-cache`, and `--geocoder-country-codes`. Match these to the production configuration if it changes. No public geocoder requests or gazetteer downloads run during evaluation. Missing gazetteers fail explicitly; uncached/unmatched names remain unresolved. The workflow cache is read only. To add missing online lookups, use the existing cache-filling workflow with the generated `geocoding/unmatched.csv`, preferably targeting an evaluation-specific cache, then pass it via `--geocoder-cache` and rerun. Freeze resources across model comparisons and do not tune overrides from test labels.

The four newer specialist checkpoints explicitly load safetensors weights;
`safetensors` is a direct Pixi dependency. In particular, oxygeneDev's pinned
safetensors checkpoint has a five-label classifier matching its configuration,
while its older `pytorch_model.bin` has a three-label classifier. We never ignore
classifier-size mismatches. MultiSent uses `use_fast: false` to read its
SentencePiece vocabulary: the project's pinned Tokenizers 0.13 cannot deserialize
its newer fast-tokenizer JSON. This follows the supported
[slow-tokenizer option](https://huggingface.co/docs/transformers/v4.27.1/en/migration).
The seven previously successful specialists retain their fast tokenizers and
PyTorch `.bin` checkpoints, preserving the evaluated inference behavior.

A model-loading failure is reported once per model. Individual failures remain
in the prediction log and are retried on the next run, even after loading settings
change. A model that never loaded is reported as `model setup failed` with blank
accuracy/F1, rather than a measured zero score. Inference failures after successful
loading remain sample-level errors.

Five-class sentiment probabilities are summed into negative (1–2 stars / very negative + negative), neutral (3 stars), and positive (4–5 stars / positive + very positive), before choosing the winning label and confidence. MultiSent's `question` is retained as an unsupported class: if it wins, it is counted incorrect and abstains. This mapping is explicit because its `id2label` uses generic labels while `label2id` provides the class meanings. Configurations were checked against the publishers' [Hebban model configuration](https://huggingface.co/BramVanroy/bert-base-dutch-cased-hebban-reviews5/blob/main/config.json), [Tabularis configuration](https://huggingface.co/tabularisai/multilingual-sentiment-analysis/blob/main/config.json), and [MultiSent configuration](https://huggingface.co/ZombitX64/MultiSent-E5-Pro/blob/main/config.json). All eleven models' mappings are stored individually.

`evaluation/metrics.csv` has one row per model: six accuracy/macro-F1 columns,
three selective-threshold columns, three selective-coverage columns, and
`average_runtime_seconds`, alongside the model name. The six main scores always
use every eligible example, before confidence filtering. Errors and unresolved
model outputs remain in their denominator. Location uses NUTS2 region matching when available (exact-name matching for
an exact-only report). Scores are on the 0–1 scale. Missing tasks and failed
models have blank scores.

For each model/task/subset, select the confidence threshold with the highest
accuracy among retained predictions using all eligible annotated examples.
Ties prefer higher coverage, then the lower threshold. Candidates are observed
confidences plus 0 and 1. By default there is no minimum coverage constraint;
`--min-coverage` can impose one explicitly. These thresholds are separate
selective-classification diagnostics; they do not filter the main accuracy/F1
columns. Their retained fraction appears as selective coverage (0–1). A high
threshold can retain very few examples. Filtered scores are only reported as
`accepted_accuracy` and `accepted_macro_f1` in the detailed table. Thresholds
are optimized on the reported data, so their filtered scores are not independent
test performance.

`metrics_detailed.csv` records `n_samples`, `n_accepted`, `coverage`, thresholds,
full-sample and explicitly labelled accepted-only scores, errors and timings
for each task/subset. `accuracy`/`macro_f1` and their `unfiltered_*` aliases use
the entire eligible sample. Sentiment excludes
sentences labelled unrelated to geothermal energy; region scoring excludes
unresolved manual region judgments, so its eligible count can be below 500.
Confidence filtering abstains below threshold; it does not change the predicted
label. Historical split counts are no longer reported.

Average runtime is seconds per example, weighted across the available location,
geothermal and sentiment tasks. It uses recorded inference timings for all
examples, before threshold filtering, includes document fallback calls once,
and excludes separate model setup/download time. Region scoring is not counted
as an additional inference task. Entirely failed tasks do not contribute to the
average. Cached timings are reused; no model rerun is necessary to regenerate
these reports:

```sh
pixi run python scripts/model_evaluation/evaluate.py --rescore-only
```

This also reuses saved region matches and makes no model or geocoder calls.

Macro F1 uses the fixed two/three-class set for classification. Location uses the union of normalized gold and valid predicted place names across the eligible sample, fixed across thresholds. Missing classes contribute zero F1. Location exact-match accuracy is the easiest location metric to interpret when many places occur only once. Location rows also report sample-wide `n_predicted_none`, `n_gold_none`, and `n_none_misses` to distinguish missing predictions from other errors. Confidence is self-reported for Ollama and a softmax probability for specialist models; these scales are not assumed calibrated or comparable.

`metrics_detailed.csv` additionally contains `location_region`, for both the all-paragraph and human-geothermal subsets. Region equality uses NUTS2 codes rather than names: different extracted names in the same province count as a region match. Whole-country labels match only the same country; `NONE` matches only `NONE`. Failed model calls and failed geocoding count as incorrect against resolved gold judgments, and unresolved predictions cannot pass the region confidence threshold. Explicitly unresolved manual judgments are excluded only from region scoring and counted in `n_excluded_gold_regions` across all samples; `n_unresolved_regions` counts unresolved model regions on evaluable rows. Thus exact and region denominators can differ. Region thresholds are optimized separately over all eligible region rows using the final extraction/fallback confidence. Region macro F1 uses the sample's union of gold and valid predicted region keys, fixed across thresholds.

`location_comparison.csv` lists both match flags, manual and predicted regions, split, geocoding source, coordinates and error for every model/paragraph, plus `paragraph_prediction`, `location_source` and `document_seconds`. `predictions.jsonl` stores the final location payload and its original extraction in `raw.paragraph_raw`. An unresolved manual region leaves `region_match` blank. `geocoding/` contains workflow inputs, intermediate CSVs/GPKGs, unmatched names, suggestions, final assigned regions, and `run.json` with resource/code hashes and commands. Geocoding is rerun against the current resources when evaluation resumes, while inference resumes from `predictions.jsonl`. Runtime columns include paragraph inference and any full-document inference. Each document call is counted once, distributed equally across the sampled rows that used it; the repeated `location_region` timing is the same inference cost, not additional inference or geocoding time.

`thresholds.csv` contains the complete accuracy/coverage search over all eligible samples. `predictions.jsonl` contains per-row outputs, errors and timings; `run.json` records code hashes, model settings and machine details. Successful calls resume; failed calls retry on rerun. Location paragraph and document calls have separate checkpoints (`location_paragraphs.jsonl` and `location_documents.jsonl`), so a failed document call does not repeat successful paragraph inference. Cache signatures are scoped to each model/task and check inference code, settings and inputs, including full article text for location. The one-time location-cache reset preserves prior geothermal/sentiment records byte-for-byte and records their audited signature compatibility in `cache_compatibility.json`; later incompatible inference changes still fail validation. Partial task/model runs preserve unrelated prediction and report rows. Configuration/code/input changes affecting a task require a fresh compatible cache for that task. Preserve Ollama model versions and hardware across a comparison; Ollama tags are mutable. Runtime is summed wall time for attempted inference, including tokenization, network round trips, retries, and any Ollama loading within a request. Specialist loading/download time is recorded separately as `setup_seconds`. Batch size is one; no warm-up is excluded. GPU calls synchronize. Timings are end-to-end measurements on the selected hardware, not hardware-normalized benchmarks. Resumed runs retain original successful-call timings; previous failed attempts are not included in the final successful-call total.

These optimized scores describe the annotated sample. Nothing here automatically changes production model selection or thresholds.

To recompute location extraction and both location scores after the location-only cache reset:

```sh
pixi run python scripts/model_evaluation/evaluate.py --tasks location
```

This runs all nine Ollama models and preserves existing geothermal and sentiment
predictions and report rows. The old paragraph-only location results, derived
geocoding files and original logs are archived under the results directory's
`backups/before_location_fallback_…/` directory. They are not read as active caches.
