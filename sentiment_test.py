from transformers import pipeline

model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

texts = [
    "Thank you, the internet service is excellent!",
    "I am very happy with the support I received.",
    "My issue was solved quickly. Great service!",
    "I love the new ZENDS plan.",
    "What are your monthly mobile plans?",
    "I want to know the price of your internet plan.",
    "What is the billing date?",
    "Can you tell me about your cloud services?",
    "My internet is not working and I am extremely angry.",
    "This is terrible. I have been waiting for hours.",
    "I am very disappointed with ZENDS service.",
    "Your service is horrible and I want a refund."
]

for text in texts:
    result = model(text)[0]

    print()
    print("Customer:", text)
    print("Model:", result["label"])
    print("Confidence:", round(result["score"], 4))