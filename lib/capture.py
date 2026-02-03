import cv2
import os
import numpy as np
from lib.face import detect_single, crop  # Ensure this exists and is correct

def get_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    return cap

def capture_images(output_dir, max_images=30):
    os.makedirs(output_dir, exist_ok=True)

    cap = get_camera()
    count = 0

    print("\n📸 Starting camera. Press SPACE to capture face. ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("Capture - Press SPACE", display)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("👋 Exit capture.")
            break

        elif key == 32:  # SPACE
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_box = detect_single(gray)

            if face_box:
                x, y, w, h = [int(v) for v in face_box]
                cropped = crop(frame, x, y, w, h)

                filename = os.path.join(output_dir, f"{count:03}.jpg")
                cv2.imwrite(filename, cropped)
                print(f"✅ Saved {filename}")
                count += 1
            else:
                print("😕 No face detected. Try again.")

            if count >= max_images:
                print("✅ Finished capturing.")
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_images("training_data/Manual_Capture")

