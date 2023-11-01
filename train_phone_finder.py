import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import cv2


input_shape = (326, 490, 3)

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Normalize the image as CNN do not respond well to large values
    image = cv2.normalize(image, None, 0, 1.0,cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return image

def prepare_data(data_path):
    with open(data_path, 'r') as file:
        lines = file.read().splitlines()

    x_train = []
    y_train = []

    for line in lines:
        parts = line.split()
        image_path = "find_phone/" + parts[0] 
        x, y = float(parts[1]), float(parts[2])

        # Load and preprocess the image
        preprocessed_image = preprocess_image(image_path)

        # Append the preprocessed image to x_train
        x_train.append(preprocessed_image)

        # Append the corresponding coordinates to y_train
        y_train.append((x, y))

    return np.array(x_train), np.array(y_train)


# Create the CNN model with regularization and dropout
def create_object_localization_model(input_shape, l2_reg=0.01, dropout_rate=0.5):
    model = Sequential()

    # Convolutional Layer 1
    model.add(layers.Conv2D(32, (3, 3), input_shape=input_shape, kernel_regularizer=l2(l2_reg)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # Convolutional Layer 2
    model.add(layers.Conv2D(64, (3, 3), kernel_regularizer=l2(l2_reg)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # Convolutional Layer 3
    model.add(layers.Conv2D(64, (3, 3), kernel_regularizer=l2(l2_reg)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # Flatten the feature maps
    model.add(layers.Flatten())

    # Fully Connected Layers
    model.add(layers.Dense(128, use_bias=False, kernel_regularizer=l2(l2_reg)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dropout(dropout_rate))

    # Output Layer for Object Position (2 units for x, y coordinates)
    model.add(layers.Dense(2, activation='linear'))
    # Print model summary
    model.summary()

    return model


def train_phone_detector(x_train, y_train):
    # Load and preprocess the training data
    # Create and compile the model
    phone_detector_model = create_object_localization_model(input_shape, l2_reg=0.01, dropout_rate=0.1)
    modelname = 'phone_detector_model_2layer.h5'
    # Load the previoiusly trained model(Uncomment the line below to load presaved model)
    #phone_detector_model = load_model(modelname)


    # Compile the model with a regression loss
    phone_detector_model.compile(optimizer='adam', loss='mean_absolute_error')

    # Train the model
    phone_detector_model.fit(x_train, y_train, epochs=200) 

    # Save the model
    phone_detector_model.save(modelname)

if __name__ == "__main__":

    train_data_path = sys.argv[1]  # Path to labeled data folder
    
    # Load the labeled data (from labels.txt)
    file_name = "labels.txt"
    for root, dirs, files in os.walk(train_data_path):
        if file_name in files:
            file_path = os.path.join(root, file_name)
            
    x_train, y_train = prepare_data(file_path)

    # Call the function to train the phone detector
    train_phone_detector(x_train, y_train)
