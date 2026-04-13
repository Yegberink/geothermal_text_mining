# Sentiment Annotation Study

This workflow samples a fixed set of Dutch newspaper sentences, runs five Hugging Face sentiment models on them, assigns the sentences across two annotators, and evaluates model performance afterward in a Jupyter notebook.

## Assignment design

- `150` annotations for `Dekker`
- `150` annotations for `Egberink`
- configurable overlap, default `30`
- default unique sampled sentences: `150 + 150 - 30 = 270`

Assignment groups in `sentences_for_annotation.csv`:

- `overlap`
- `dekker_only`
- `egberink_only`

## Generate the study data

From the repository root:

```bash
python scripts/assess_sentiment_models.py
```

This writes:

- `annotation/sentiment_model_assessment/sentences_for_annotation.csv`
- `annotation/sentiment_model_assessment/model_predictions_long.csv`
- `annotation/sentiment_model_assessment/model_predictions_wide.csv`
- `annotation/sentiment_model_assessment/models.json`

If some Hugging Face checkpoints are not cached locally and the environment is offline, their rows will still be written but with `error` status in the prediction files.

## Run the annotation app

```bash
streamlit run annotation/sentiment_model_assessment/app.py
```

The annotation database is stored at:

- `annotation/sentiment_model_assessment/databases/annotations.db`

## Analyze performance

Open:

- `annotation/sentiment_model_assessment/analysis.ipynb`

The notebook loads the model predictions and annotation database, merges them, and calculates model performance on the annotated sentences.
