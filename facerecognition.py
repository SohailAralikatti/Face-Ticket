import cv2
import pickle
import numpy as np
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
from lib.face import detect_single, crop  # Ensure updated face.py is used

# Load FaceNet model
embedder = FaceNet()

# Load stored embeddings
with open("embeddings/encodings.pickle", "rb") as f:
    db = pickle.load(f)

def recognize_face_from_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_box = detect_single(gray)
    if face_box is None:
        return "Unknown", 0.0, frame, None

    x, y, w, h = face_box
    cropped_face = crop(rgb, x, y, w, h)

    # Embed using FaceNet
    results = embedder.extract(cropped_face, threshold=0.9)
    if not results:
        return "Unknown", 0.0, frame, cropped_face

    face_embedding = results[0]['embedding']
    similarities = cosine_similarity([face_embedding], db["embeddings"])[0]
    max_index = np.argmax(similarities)
    max_score = similarities[max_index]

    label = "Unknown"
    if max_score > 0.65:
        label = db["names"][max_index]

    # Draw result on frame
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0) if label != "Unknown" else (0, 0, 255), 2)
    cv2.putText(frame, f"{label} ({max_score:.2f})", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return label, max_score, frame
