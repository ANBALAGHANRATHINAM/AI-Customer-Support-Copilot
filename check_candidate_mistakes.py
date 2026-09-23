import pandas as pd
from pathlib import Path
import sys

ROOT = Path(".").resolve()
SEG = ROOT / "2_NLP_Intelligence"

sys.path.insert(0, str(SEG / "code"))

from intent import IntentClassifier
from preprocessing import clean_text

data = pd.read_csv(SEG / "evaluation" / "realistic_generalization.csv")

classifier = IntentClassifier(
    SEG / "models" / "intent_distilbert_candidate"
)

mistakes = []

for row in data.itertuples():
    result = classifier.predict(clean_text(row.text))
    predicted = result["intent"]

    if predicted != row.intent:
        mistakes.append(
            {
                "Expected": row.intent,
                "Predicted": predicted,
                "Text": row.text,
            }
        )

print(f"\nTotal mistakes: {len(mistakes)}\n")

for item in mistakes:
    print(f"Expected:  {item['Expected']}")
    print(f"Predicted: {item['Predicted']}")
    print(f"Text:      {item['Text']}")
    print("-" * 80)
