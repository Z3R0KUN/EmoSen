import cv2
from keras.models import model_from_json
import numpy as np

# -------- Load model --------
with open("emotiondetector.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# -------- Face detector --------
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# -------- Labels --------
labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

# -------- Feature extractor --------
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


# -------- Main loop (ONLY runs if file is executed directly) --------
if __name__ == "__main__":
    webcam = cv2.VideoCapture(0)

    if not webcam.isOpened():
        print("Camera not available")
        exit()

    while True:
        ret, im = webcam.read()
        if not ret:
            print("Failed to read frame")
            break

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (p, q, r, s) in faces:
            face_img = gray[q:q+s, p:p+r]
            face_img = cv2.resize(face_img, (48, 48))
            img = extract_features(face_img)

            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]

            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
            cv2.putText(
                im,
                prediction_label,
                (p, q-10),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                2,
                (0, 0, 255),
                2
            )

        cv2.imshow("Output", im)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()
