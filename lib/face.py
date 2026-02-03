import cv2

# Load OpenCV's Haar cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_single(gray_image):
    """
    Detect a single face in a grayscale image using Haar cascade.
    Returns (x, y, w, h) if exactly one face is found, else None.
    """
    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 1:
        return faces[0]
    return None

def crop(image, x, y, w, h):
    """
    Crop a region from the image using bounding box.
    """
    return image[y:y+h, x:x+w]

def resize(face_img, size=(160, 160)):
    """
    Resize face image to a target size (optional, FaceNet handles this internally).
    """
    return cv2.resize(face_img, size)

