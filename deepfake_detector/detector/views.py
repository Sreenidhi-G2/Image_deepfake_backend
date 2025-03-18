from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework import status
from .utils import predict_deepfake
import os
import tempfile  # Import the tempfile module

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