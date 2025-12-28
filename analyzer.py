import re
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax


# preprocessing
def clean_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# model loading
print("🔄 Loading text models...")

emotion_model_name = "j-hartmann/emotion-english-distilroberta-base"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)
emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_labels = ['negative', 'neutral', 'positive']

print("✅ Models ready!\n")


# core function
def predict_emotion(text: str):
    text = clean_text(text)
    inputs = emotion_tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = emotion_model(**inputs)
    scores = softmax(outputs.logits[0].numpy())
    return {emotion_labels[i]: float(scores[i]) for i in range(len(emotion_labels))}


def predict_sentiment(text: str):
    text = clean_text(text)
    inputs = sentiment_tokenizer(text, return_tensors='pt', truncation=True)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    scores = softmax(outputs.logits[0].numpy())
    idx = int(np.argmax(scores))
    return {
        "label": sentiment_labels[idx],
        "confidence": float(scores[idx]),
        "all": {sentiment_labels[i]: float(scores[i]) for i in range(len(sentiment_labels))}
    }


# Analyser
def analyze_text(text: str):
    if not text or not text.strip():
        return {"error": "Empty text input."}

    cleaned = clean_text(text)
    sent = predict_sentiment(cleaned)
    emos = predict_emotion(cleaned)

    dominant_emo = max(emos, key=emos.get)
    result = {
        "input_text": text,
        "cleaned_text": cleaned,
        "sentiment": {
            "label": sent["label"].capitalize(),
            "confidence": round(sent["confidence"] * 100, 2),
            "all_scores": sent["all"]
        },
        "emotion": {
            "label": dominant_emo.capitalize(),
            "confidence": round(emos[dominant_emo] * 100, 2),
            "all_scores": emos
        }
    }
    return result


# main function

if __name__ == "__main__":
    while True:
        text = input("\nEnter text (or 'exit'): ")
        if text.lower() == "exit":
            break
        out = analyze_text(text)
        print("\n🧹 Cleaned:", out["cleaned_text"])
        print(f"💬 Sentiment: {out['sentiment']['label']} ({out['sentiment']['confidence']}%)")
        print(f"🎭 Emotion: {out['emotion']['label']} ({out['emotion']['confidence']}%)")

        df = pd.DataFrame(out["emotion"]["all_scores"].items(), columns=["Emotion", "Probability"])
        df = df.sort_values("Probability", ascending=False)
        print(df.to_string(index=False))
