# ==========================================================
# Imports
# ==========================================================
import re
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax
from tabulate import tabulate


# ==========================================================
# 1️⃣ Text Preprocessing
# ==========================================================
def clean_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove URLs
    text = re.sub(r"\@\w+", "", text)                    # remove @mentions
    text = re.sub(r"\#", "", text)                       # remove hashtags symbol
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", " ", text)     # remove special chars
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ==========================================================
# 2️⃣ Load Emotion & Sentiment Models
# ==========================================================
print("Loading models... (first run may take a minute)")

# ---- Emotion Model (7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise)
emotion_model_name = "j-hartmann/emotion-english-distilroberta-base"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)
emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

# ---- Sentiment Model (3-way sentiment: negative, neutral, positive)
sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_labels = ['negative', 'neutral', 'positive']


# ==========================================================
# 3️⃣ Emotion Prediction
# ==========================================================
def predict_emotion(text: str):
    text = clean_text(text)
    encoded = emotion_tokenizer(text, return_tensors='pt',max_length=512, truncation=True)
    with torch.no_grad():
        output = emotion_model(**encoded)
    scores = output.logits[0].detach().numpy()
    probs = softmax(scores)
    results = {emotion_labels[i]: float(probs[i]) for i in range(len(emotion_labels))}
    return results


# ==========================================================
# 4️⃣ Sentiment Prediction
# ==========================================================
def predict_sentiment(text: str):
    text = clean_text(text)
    encoded = sentiment_tokenizer(text, return_tensors='pt', truncation=True)
    with torch.no_grad():
        output = sentiment_model(**encoded)
    scores = output.logits[0].detach().numpy()
    probs = softmax(scores)
    label = sentiment_labels[np.argmax(probs)]
    confidence = float(np.max(probs))
    return label.upper(), confidence


# ==========================================================
# 5️⃣ Main Analyzer
# ==========================================================
def analyze_text(text: str):
    text = clean_text(text)
    sentiment_label, sentiment_conf = predict_sentiment(text)
    emotions = predict_emotion(text)

    print("\n================= ANALYSIS RESULT =================")
    print(f"📝 Text: {text}\n")
    print(f"💬 Sentiment: {sentiment_label} (Confidence: {sentiment_conf:.3f})\n")

    df = pd.DataFrame(list(emotions.items()), columns=['Emotion', 'Probability'])
    df = df.sort_values(by='Probability', ascending=False)
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))
    print("===================================================")


# ==========================================================
# 6️⃣ Run Interactively
# ==========================================================
if __name__ == "__main__":
    while True:
        user_input = input("\nEnter text (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Goodbye 👋")
            break
        analyze_text(user_input)