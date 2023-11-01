import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras


input_shape = (326, 490, 3)

def load_phone_detector_model(model_path):
    return keras.models.load_model(model_path)

def preprocess_test_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.normalize(image, None, 0, 1.0,cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return image

def detect_phone(model, test_image_path):
    # Load and preprocess the test image
    test_image = preprocess_test_image(test_image_path)  

    # Use the model to predict phone coordinates
    predicted_coords = model.predict(np.array([test_image]))[0]
    return predicted_coords

if __name__ == "__main__":
    model_path = 'phone_detector_model_mse_fullimg.h5'  # Provide the path to the trained model
    test_image_path = sys.argv[1]  # Path to the test image
    
    # Load the trained model
    phone_detector_model = load_phone_detector_model(model_path)

    # Perform phone detection on the test image
    phone_coords = detect_phone(phone_detector_model, test_image_path)

    # Print the normalized coordinates
    print(f"{phone_coords[0]:.4f} {phone_coords[1]:.4f}")
