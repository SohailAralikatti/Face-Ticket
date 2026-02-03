import cv2
import dlib
import time
import pickle
import numpy as np
from scipy.spatial import distance as dist
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import os
import imutils

# Paths (Update if your filenames are different)
MODEL_PATH = "liveness.h5"
LABEL_ENCODER_PATH = "le.pickle"
DETECTOR_DIR = "."
SHAPE_PREDICTOR_PATH = "shapepredictor/shape_predictor_68_face_landmarks.dat"
CONFIDENCE_THRESHOLD = 0.5

# Load models
print("[INFO] Loading models...")
protoPath = os.path.join(DETECTOR_DIR, "deploy.prototxt")
modelPath = os.path.join(DETECTOR_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
face_net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

model = load_model(MODEL_PATH)
le = pickle.loads(open(LABEL_ENCODER_PATH, "rb").read())

# Dlib facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

# Eye indices for blink detection
LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))

EYE_AR_THRESH = 0.30
EYE_AR_CONSEC_FRAMES = 2

COUNTER_LEFT = 0
COUNTER_RIGHT = 0
TOTAL_LEFT = 0
TOTAL_RIGHT = 0

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Start webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    x = 0
    for rect in rects:
        landmarks = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])
        left_eye = landmarks[LEFT_EYE_POINTS]
        right_eye = landmarks[RIGHT_EYE_POINTS]

        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)

        if ear_left < EYE_AR_THRESH:
            COUNTER_LEFT += 1
        else:
            if COUNTER_LEFT >= EYE_AR_CONSEC_FRAMES:
                TOTAL_LEFT += 1
                print("👁️ Left eye blinked")
                COUNTER_LEFT = 0

        if ear_right < EYE_AR_THRESH:
            COUNTER_RIGHT += 1
        else:
            if COUNTER_RIGHT >= EYE_AR_CONSEC_FRAMES:
                TOTAL_RIGHT += 1
                print("👁️ Right eye blinked")
                COUNTER_RIGHT = 0

        x = TOTAL_LEFT + TOTAL_RIGHT

    # Face detection
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > CONFIDENCE_THRESHOLD and x > 2:  # Only run prediction if enough blinks
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            face = frame[startY:endY, startX:endX]
            try:
                face = cv2.resize(face, (32, 32))
            except:
                continue
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            preds = model.predict(face)[0]
            j = np.argmax(preds)
            label = le.classes_[j]

            label_text = "{}: {:.4f}".format(label, preds[j])
            cv2.putText(frame, label_text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)

    cv2.putText(frame, f"Blinks: {TOTAL_LEFT + TOTAL_RIGHT}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow("Face Liveness Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()

