#!/usr/bin/env python
# coding: utf8
"""MMM-Facial-Recognition - MagicMirror Module
Face Recognition training script config
The MIT License (MIT)

Copyright (c) 2016 Paul-Vincent Roll (MIT License)
Based on work by Tony DiCola (Copyright 2013) (MIT License)
"""
import inspect
import os
import platform
import cv2

CV_MAJOR_VER, CV_MINOR_VER, mv1, mv2 = (cv2.__version__ + '.0.0.0').split(".")[:4]


_platform = platform.system().lower()
path_to_file = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

RECOGNITION_ALGORITHM = 1
POSITIVE_THRESHOLD = 80


def set_recognition_algorithm(algorithm):
    if algorithm < 1 or algorithm > 3:
        print("WARNING: face algorithm must be in the range 1-3")
        RECOGNITION_ALGORITHM = 1
        os._exit(1)
    RECOGNITION_ALGORITHM = algorithm
    # Threshold for the confidence of a recognized face before it's
    # considered a positive match.  Confidence values below this
    # threshold will be considered a positive match because the lower
    # the confidence value, or distance, the more confident the
    # algorithm is that the face was correctly detected.  Start with a
    # value of 3000, but you might need to tweak this value down if
    # you're getting too many false positives (incorrectly recognized
    # faces), or up if too many false negatives (undetected faces).
    # POSITIVE_THRESHOLD = 3500.0
    if RECOGNITION_ALGORITHM == 1:
        POSITIVE_THRESHOLD = 80
    elif RECOGNITION_ALGORITHM == 2:
        POSITIVE_THRESHOLD = 250
    else:
        POSITIVE_THRESHOLD = 3000


users = []  # will be filled in train.py dynamically if needed


# Edit the values below to configure the training and usage of the
# face recognition box.
if ('FACE_ALGORITHM' in os.environ):
    set_recognition_algorithm(int(os.environ['FACE_ALGORITHM']))
    print("Using FACE_ALGORITM: {0}".format(RECOGNITION_ALGORITHM))
else:
    set_recognition_algorithm(1)
    print("Using default FACE_ALGORITM: {0}".format(RECOGNITION_ALGORITHM))


# File to save and load face recognizer model.
TRAINING_FILE = 'training.xml'
TRAINING_DIR = './training_data/'

# Size (in pixels) to resize images for training and prediction.
# Don't change this unless you also change the size of the training images.
FACE_WIDTH = 92
FACE_HEIGHT = 112

# Face detection cascade classifier configuration.
# You don't need to modify this unless you know what you're doing.
# See: http://docs.opencv.org/modules/objdetect/doc/cascade_classification.html
#HAAR_FACES = 'lib/haarcascade_frontalface_alt.xml'
#HAAR_FACES = 'lib/haarcascade_frontalface_alt2.xml'
#HAAR_FACES = 'lib/haarcascade_frontalface_default.xml'
HAAR_FACES = 'lib/haarcascade_frontalface.xml'
HAAR_EYES = 'lib/haarcascade_eye.xml'
HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS_FACE = 6     # More neighbors = more confidence needed
HAAR_MIN_NEIGHBORS_EYES = 2     # Slightly stricter for eyes too
HAAR_MIN_SIZE_FACE = (60, 60)   # Ignore very small regions
HAAR_MIN_SIZE_EYES = (20, 20)


def get_camera(preview=True):
    from . import webcam
    import cv2

    print("🔍 Searching for available webcam...")
    for i in range(5):  # Check camera indexes 0 through 4
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            print(f"✅ Using webcam at index {i}")
            return webcam.OpenCVCapture(device_id=i)

    raise Exception("❌ No available camera found.")



def is_cv2():
    if CV_MAJOR_VER == 2:
        return True
    else:
        return False


def is_cv3():
    if CV_MAJOR_VER == 3:
        return True
    else:
        return False


def model(algorithm, thresh):
    model = None
    if hasattr(cv2, 'face'):  # Ensure cv2.face exists
        if algorithm == 1:
            model = cv2.face.LBPHFaceRecognizer_create(threshold=thresh)
        elif algorithm == 2:
            model = cv2.face.FisherFaceRecognizer_create(threshold=thresh)
        elif algorithm == 3:
            model = cv2.face.EigenFaceRecognizer_create(threshold=thresh)
        else:
            print("WARNING: face algorithm must be in the range 1-3")
            os._exit(1)
    else:
        raise Exception("❌ OpenCV 'face' module not found. Please install opencv-contrib-python.")

    return model



def user_label(i):
    i = i - 1
    if i < 0 or i >= len(users):
        return f"User{i+1}"
    name, usn = users[i]
    return f"{name} ({usn})"
