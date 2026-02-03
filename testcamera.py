import cv2
from facerecognition import recognize_face_from_frame

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    label, score, annotated, _ = recognize_face_from_frame(frame)  # <- note the extra `_` to discard 4th value
    cv2.imshow("Face Recognition", annotated)

    if cv2.waitKey(1) == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()

