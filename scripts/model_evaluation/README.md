# Manual Dutch model evaluation

Open `annotation/final/prepare_annotation.ipynb` and run it once. It samples 500 distinct Dutch paragraphs uniformly from the cleaned paragraph corpus, before keyword/model filtering, and 500 distinct sentences uniformly from current Dutch sentiment results. The corpus itself may reflect the original newspaper retrieval criteria. No model predictions, confidence values, keyword hits, geographic matches, or rationales enter the annotation CSVs. Sampling uses seed 42 and refuses to overwrite existing files. This is independent of Snakemake: it only reads existing source files and never changes workflow results or caches.

Launch the annotation interface with `pixi run streamlit run annotation/final/app.py`. It saves labels directly to the CSVs below, tracks progress, and supports revisiting completed items. See [app instructions](../../annotation/final/README.md).

Edit `annotation/final/paragraphs.csv`:

- `gold_location`: ONE primary geographic location, using the most specific clearly supported place name. Write `NONE` if there is no clear primary location. Annotate every paragraph, including those unrelated to geothermal energy. Use consistent Dutch place names; scoring normalizes case, Unicode, whitespace, and curly apostrophes, but does not geocode or infer aliases.
- `gold_geothermal`: `YES` if geothermal energy is the main or substantial subject; `NO` if absent, incidental, or another subject dominates. Resolve uncertain cases manually and record decisions in `notes`.

Edit `annotation/final/sentences.csv`:

- `gold_sentiment`: `negative`, `neutral`, or `positive`. Negative emphasizes risks, costs, criticism, uncertainty, conflict, harm, or failure; positive emphasizes benefits, support, progress, feasibility, opportunity, or success; neutral is factual, procedural, descriptive, or mixed without clear polarity. Judge the sentence itself, matching the workflow input.
- If a sampled sentence is actually unrelated to geothermal energy, use `not_geothermal`. Such rows are counted in the annotation file but excluded from sentiment evaluation; do not silently replace them. Membership in a geothermal paragraph does not guarantee sentence-level relevance.

Only edit `gold_*` and `notes`. The evaluator checks all original IDs, texts, and metadata against the manifest and requires complete labels. The manifest records source file hashes and a deterministic split shared by both samples: approximately 30% of documents for calibration and 70% for testing. All samples from one document stay together. Do not use test labels to change prompts, mappings, or thresholds.

From the repository root:

```sh
pixi run python scripts/model_evaluation/evaluate.py --validate-only
pixi run python scripts/model_evaluation/evaluate.py
# Optional smaller run / Apple GPU for specialist models:
pixi run python scripts/model_evaluation/evaluate.py --models qwen2.5:7b --results-dir annotation/final/qwen7
pixi run python scripts/model_evaluation/evaluate.py --device mps --results-dir annotation/final/mps
```

Start Ollama and pull `qwen2.5:14b`, `qwen2.5:7b`, `llama3.1:8b`, `mistral`, and `phi3` before evaluation. The duplicate Ollama entries in the requested list are run once per task. All five models run all three tasks; the eleven specialist models run sentiment only. Hugging Face weights download on first use. Existing Pixi dependencies provide pandas, numpy, torch, transformers, requests, tenacity, and the workflow helper imports. Model IDs, pinned revisions and explicit label mappings are in `models.json`.

Ollama calls directly reuse the workflow functions, prompts, system messages, options, response parsers and retry behavior. Sentiment always selects `zero_shot`; no examples are supplied. Paragraph location inference runs independently for each sampled paragraph, without workflow relevance gating, document fallback, or geocoding. The unchanged location prompt assumes geothermal relevance, so results include both all paragraphs and the subset manually labeled geothermal. Geothermal `MAYBE` counts as an incorrect binary prediction and cannot pass a confidence threshold. Workflow parser fallbacks (including confidence-zero outputs) remain part of the measured behavior.

Five-class sentiment probabilities are summed into negative (1–2 stars / very negative + negative), neutral (3 stars), and positive (4–5 stars / positive + very positive), before choosing the winning label and confidence. MultiSent's `question` is retained as an unsupported class: if it wins, it is counted incorrect and abstains. This mapping is explicit because its `id2label` uses generic labels while `label2id` provides the class meanings. Configurations were checked against the publishers' [Hebban model configuration](https://huggingface.co/BramVanroy/bert-base-dutch-cased-hebban-reviews5/blob/main/config.json), [Tabularis configuration](https://huggingface.co/tabularisai/multilingual-sentiment-analysis/blob/main/config.json), and [MultiSent configuration](https://huggingface.co/ZombitX64/MultiSent-E5-Pro/blob/main/config.json). All eleven models' mappings are stored individually.

`evaluation/metrics.csv` reports held-out accuracy and macro F1 on every test row, plus accuracy, macro F1 and coverage on accepted test rows. Confidence thresholds mean **abstain below threshold**, not relabel as neutral/NO/NONE. For each model/task/subset, select the threshold that maximizes calibration macro F1 while retaining at least 80% of calibration rows (`--min-coverage` changes this). Ties prefer higher coverage, then a lower threshold. Candidate thresholds are all observed calibration confidences plus 0 and 1. A minimum-coverage constraint avoids a misleading optimum that retains only a handful of easy samples. If no threshold satisfies it, the report says so. Test coverage can fall below the calibration constraint.

Macro F1 uses the fixed two/three-class set for classification. Location uses the union of normalized gold and valid predicted place names within each partition, fixed across thresholds. Missing classes contribute zero F1. Location exact-match accuracy is the easiest location metric to interpret when many places occur only once. Confidence is self-reported for Ollama and a softmax probability for specialist models; these scales are not assumed calibrated or comparable.

`thresholds_calibration.csv` contains the complete calibration search. `predictions.jsonl` contains per-row outputs, errors and timings; `run.json` records code hashes, model settings and machine details. Successful calls resume from this separate log; failed calls retry on rerun. Configuration/code/input changes require a new results directory. Preserve Ollama model versions and hardware across a comparison; Ollama tags are mutable. Runtime is summed wall time for attempted inference, including tokenization, network round trips, retries, and any Ollama loading within a request. Specialist loading/download time is recorded separately as `setup_seconds`. Batch size is one; no warm-up is excluded. GPU calls synchronize. Timings are end-to-end measurements on the selected hardware, not hardware-normalized benchmarks. Resumed runs retain original successful-call timings; previous failed attempts are not included in the final successful-call total.

Compare model rankings on calibration first. Held-out scores assess the chosen settings; choosing a winner from test scores and claiming an unbiased final score would require another independent holdout. Nothing here automatically changes production model selection.
