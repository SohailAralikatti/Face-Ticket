import cv2
import mediapipe as mp
import numpy as np
from math import sqrt

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define eye landmarks
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Threshold for EAR to determine eye state
EYE_AR_THRESHOLD = 0.25

def calculate_EAR(landmarks, eye_indices, image_width, image_height):
    # Extract the coordinates of the eye landmarks
    coords = [(int(landmarks[i].x * image_width), int(landmarks[i].y * image_height)) for i in eye_indices]
    
    # Compute the distances
    vertical_1 = sqrt((coords[1][0] - coords[5][0])**2 + (coords[1][1] - coords[5][1])**2)
    vertical_2 = sqrt((coords[2][0] - coords[4][0])**2 + (coords[2][1] - coords[4][1])**2)
    horizontal = sqrt((coords[0][0] - coords[3][0])**2 + (coords[0][1] - coords[3][1])**2)
    
    # Compute EAR
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    image_height, image_width = frame.shape[:2]

    # Convert the BGR image to RGB before processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Calculate EAR for both eyes
            left_EAR = calculate_EAR(face_landmarks.landmark, LEFT_EYE, image_width, image_height)
            right_EAR = calculate_EAR(face_landmarks.landmark, RIGHT_EYE, image_width, image_height)
            avg_EAR = (left_EAR + right_EAR) / 2.0

            # Determine eye state
            if avg_EAR < EYE_AR_THRESHOLD:
                print("Eyes Closed")
            else:
                print("Eyes Open")

    # Display the resulting frame
    cv2.imshow('Eye State Detection', frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

