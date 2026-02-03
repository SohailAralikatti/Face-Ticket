import cv2
import mediapipe as mp
import numpy as np
from math import sqrt
import time

# Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
EYE_AR_THRESHOLD = 0.25

# Liveness parameters
TOTAL_DURATION = 6  # seconds
CLOSE_REQUIRED = 3  # seconds

# EAR calculation
def calculate_EAR(landmarks, eye_indices, image_width, image_height):
    coords = [(int(landmarks[i].x * image_width), int(landmarks[i].y * image_height)) for i in eye_indices]
    v1 = sqrt((coords[1][0] - coords[5][0])**2 + (coords[1][1] - coords[5][1])**2)
    v2 = sqrt((coords[2][0] - coords[4][0])**2 + (coords[2][1] - coords[4][1])**2)
    h = sqrt((coords[0][0] - coords[3][0])**2 + (coords[0][1] - coords[3][1])**2)
    return (v1 + v2) / (2.0 * h)

# Start webcam
cap = cv2.VideoCapture(0)

start_time = None
eyes_closed_start = None
eyes_closed_total = 0
liveness_result = None
result_displayed = False
FONT = cv2.FONT_HERSHEY_SIMPLEX

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    ih, iw = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    current_time = time.time()

    if results.multi_face_landmarks:
        if start_time is None:
            start_time = current_time
            eyes_closed_total = 0
            eyes_closed_start = None
            liveness_result = None
            print("[INFO] Face detected. Timer started.")

        for face_landmarks in results.multi_face_landmarks:
            # EAR computation
            left_ear = calculate_EAR(face_landmarks.landmark, LEFT_EYE, iw, ih)
            right_ear = calculate_EAR(face_landmarks.landmark, RIGHT_EYE, iw, ih)
            ear = (left_ear + right_ear) / 2.0
            print(ear)

            # Check for closed eyes
            if ear < EYE_AR_THRESHOLD:
                if eyes_closed_start is None:
                    eyes_closed_start = current_time
            else:
                if eyes_closed_start:
                    eyes_closed_total += current_time - eyes_closed_start
                    eyes_closed_start = None

            # Show prompt
            cv2.putText(frame, f"Close eyes for at least 3s in 6s", (20, 40), FONT, 0.7, (0, 255, 255), 2)

            # Show timer
            elapsed = int(current_time - start_time)
            cv2.putText(frame, f"Timer: {elapsed}s", (20, 80), FONT, 0.8, (255, 255, 255), 2)

            # End test after 6s
            if elapsed >= TOTAL_DURATION and not result_displayed:
                if eyes_closed_start:
                    eyes_closed_total += current_time - eyes_closed_start
                    eyes_closed_start = None

                if eyes_closed_total >= CLOSE_REQUIRED:
                    liveness_result = "✅ LIVE FACE"
                    print("[RESULT] ✅ LIVE FACE")
                else:
                    liveness_result = "❌ FAKE FACE"
                    print("[RESULT] ❌ FAKE FACE")

                result_displayed = True
                result_display_time = current_time

    else:
        start_time = None
        eyes_closed_start = None
        eyes_closed_total = 0
        result_displayed = False
        cv2.putText(frame, "No face detected", (20, 40), FONT, 1, (0, 0, 255), 2)

    # Display final result
    if liveness_result:
        cv2.putText(frame, liveness_result, (20, 130), FONT, 1.2,
                    (0, 255, 0) if "LIVE" in liveness_result else (0, 0, 255), 3)

    # Close 3s after result
    if result_displayed and time.time() - result_display_time >= 3:
        break

    cv2.imshow("Liveness Detection via Eye Closure", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

