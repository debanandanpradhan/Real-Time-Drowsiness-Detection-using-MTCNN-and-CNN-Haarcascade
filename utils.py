import cv2
import numpy as np
from tensorflow.keras.models import load_model

def preprocess_eye(frame, eye_position):
    # Extract coordinates of the eye
    x, y = eye_position
    eye_image = frame[y-15:y+15, x-15:x+15]  # Crop a small region around the eye
    eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
    eye_image = cv2.resize(eye_image, (24, 24))
    eye_image = eye_image / 255.0
    eye_image = np.expand_dims(eye_image, axis=(0, -1))
    return eye_image



def load_eye_model():
    return load_model("eye_model.h5")
