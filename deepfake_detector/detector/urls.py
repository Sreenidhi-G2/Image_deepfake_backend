from django.urls import path
from .views import DeepfakeDetectionAPI

urlpatterns = [
    path('detect/', DeepfakeDetectionAPI.as_view(), name='deepfake-detect'),
]