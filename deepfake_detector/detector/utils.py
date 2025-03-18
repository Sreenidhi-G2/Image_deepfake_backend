import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the model
# Update the model path to point to the new folder
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'deepfake_cnn_model.h5')
model = load_model(model_path)

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
    """
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    return prediction[0][0]  # Assuming binary classification