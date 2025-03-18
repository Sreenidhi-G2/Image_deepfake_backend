from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework import status
from .utils import predict_deepfake
import os
import tempfile
import requests
import tensorflow as tf

# Google Drive model ID (replace with your actual model ID)
MODEL_ID = "1mX6nXgtNNJmK0-jbBZeiMGxvIJSLqp8u"
MODEL_PATH = "deepfake_detector/models/deepfake_cnn_model.h5"

def download_model_from_drive():
    """Download model from Google Drive using proper chunk handling."""
    if os.path.exists(MODEL_PATH):
        print("✅ Model already exists.")
        return

    print("🚀 Downloading model from Google Drive...")
    
    # Step 1: Get confirmation token
    session = requests.Session()
    URL = f"https://drive.google.com/uc?export=download&id={MODEL_ID}"
    response = session.get(URL, stream=True)
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            URL = f"https://drive.google.com/uc?export=download&id={MODEL_ID}&confirm={value}"
            break

    # Step 2: Download model in chunks
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with session.get(URL, stream=True) as response, open(MODEL_PATH, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
            if chunk:
                file.write(chunk)

    print("✅ Model download complete!")

# Ensure model is available
download_model_from_drive()

class DeepfakeDetectionAPI(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return Response({"error": "No image uploaded"}, status=status.HTTP_400_BAD_REQUEST)

        uploaded_image = request.FILES['image']
        temp_image_path = os.path.join(tempfile.gettempdir(), uploaded_image.name)  # Use system temp directory

        # Save the uploaded image temporarily
        with open(temp_image_path, 'wb+') as destination:
            for chunk in uploaded_image.chunks():
                destination.write(chunk)

        # Make prediction
        try:
            prediction = predict_deepfake(temp_image_path)
            result = "Deepfake" if prediction > 0.5 else "Real"
            confidence = float(prediction)
            return Response({
                "result": result,
                "confidence": confidence,
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
