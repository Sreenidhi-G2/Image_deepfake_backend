import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Define model path (should match the one in `views.py`)
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'deepfake_cnn_model.h5')

def preprocess_image(img_path, target_size=(224, 224)):
    """
    Preprocess the image for the model.
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

def predict_deepfake(img_path):
    """
    Predict if the image is a deepfake.
    Loads the model dynamically (since it is downloaded from Google Drive).
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    # Load model dynamically
    model = load_model(MODEL_PATH)

    # Preprocess image
    img_array = preprocess_image(img_path)

    # Make prediction
    prediction = model.predict(img_array)
    
    return prediction[0][0]  # Assuming binary classification
