import cv2
from fer import FER
import os
import numpy as np

def analyze_face(image_path: str):
    """
    Detect emotion from a given image using FER (Facial Emotion Recognition).
    Works with color or grayscale images and returns all 7 emotions.
    """
    if not os.path.exists(image_path):
        print(f"❌ Image file not found at: {image_path}")
        return

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Could not load image: {image_path}")
        return

    # Initialize FER detector (uses MTCNN for better face detection)
    detector = FER(mtcnn=True)

    # Detect emotions
    results = detector.detect_emotions(img)

    if not results:
        print("😕 No face detected in image.")
        return

    # FER can detect multiple faces — we’ll take the most confident one
    best_face = max(results, key=lambda r: max(r['emotions'].values()))
    emotions = best_face['emotions']

    # Sort emotions by confidence
    sorted_emotions = dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True))

    dominant_emotion = max(emotions, key=emotions.get)
    confidence = emotions[dominant_emotion] * 100

    print("\n================ FACE EMOTION ANALYSIS ================")
    print(f"🖼️  Image: {os.path.basename(image_path)}")
    print(f"🎭 Dominant Emotion: {dominant_emotion.capitalize()} ({confidence:.2f}%)")
    print("📊 Emotion Probabilities:")
    for emotion, score in sorted_emotions.items():
        print(f"   {emotion.capitalize():<10}: {score*100:.2f}%")
    print("=======================================================")

    return dominant_emotion, emotions


if __name__ == "__main__":
    print("=== EmoSenAI Advanced Facial Emotion Analyzer ===")
    print("Type 'exit' to quit anytime.\n")

    while True:
        path = input("Enter image path (e.g. C:/Users/You/Desktop/happy.jpg): ").strip()

        if path.lower() in ["exit", "quit"]:
            print("👋 Exiting EmoSenAI Facial Analyzer.")
            break

        if not path:
            print("⚠️ Please provide a valid path.")
            continue

        analyze_face(path)
        print("\n✅ You can test another image below ⬇️\n")
