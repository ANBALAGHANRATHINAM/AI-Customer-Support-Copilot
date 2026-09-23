# Segment 2 — NLP Intelligence

This segment is isolated from Segment 1 and consumes its existing train, validation, and test CSV files without changing them.

- `code/preprocessing.py` conservatively normalizes customer queries.
- `scripts/train_intent.py` fine-tunes `distilbert-base-uncased` on the existing training split, selects by validation weighted F1, and evaluates on the test split only after training.
- `code/sentiment.py` uses the pretrained `j-hartmann/emotion-english-distilroberta-base` model. Its direct `joy`, `neutral`, and `anger` outputs are aggregated into the required Happy, Neutral, and Angry UI labels.
- `code/priority.py` provides deterministic High/Medium/Low triage.
- `code/pipeline.py` provides the combined inference output.

Run training from the repository root with the project Python environment:

```powershell
python 2_NLP_Intelligence/scripts/train_intent.py
python 2_NLP_Intelligence/scripts/verify_sentiment_model.py
```

Training writes the model under `models/intent_distilbert/` and reproducible evaluation artifacts under `evaluation/`.
