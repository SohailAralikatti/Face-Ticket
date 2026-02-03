import cv2
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load the trained liveness detection model
liveness_model = load_model("model.h5")

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def is_real_face(face_roi):
    try:
        face_crop = cv2.resize(face_roi, (150, 150))
        cnn_face = face_crop.astype("float32") / 255.0
        cnn_face = img_to_array(cnn_face)
        cnn_face = np.expand_dims(cnn_face, axis=0)

        pred = liveness_model.predict(cnn_face)[0][0]
        print(f"[INFO] Liveness score: {pred:.4f}")
        return pred <= 0.3
    except Exception as e:
        print(f"[ERROR] Failed to process face: {e}")
        return False

cap = cv2.VideoCapture(0)

print("[INFO] Starting webcam... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror view
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        label = "Real" if is_real_face(face_roi) else "Fake"

        color = (0, 255, 0) if label == "Real" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Liveness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

