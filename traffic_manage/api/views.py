from django.shortcuts import render

# Create your views here.
from rest_framework import viewsets
from .models import VehicleCount, SignalState, Prediction, Log
from .serializers import VehicleCountSerializer, SignalStateSerializer, PredictionSerializer, LogSerializer

class VehicleCountViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = VehicleCount.objects.all().order_by('-timestamp')
    serializer_class = VehicleCountSerializer

class SignalStateViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = SignalState.objects.all()
    serializer_class = SignalStateSerializer

class PredictionViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Prediction.objects.all().order_by('-created_at')
    serializer_class = PredictionSerializer

class LogViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Log.objects.all().order_by('-time')
    serializer_class = LogSerializer