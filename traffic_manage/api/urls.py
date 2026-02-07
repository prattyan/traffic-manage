from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import VehicleCountViewSet, SignalStateViewSet, PredictionViewSet, LogViewSet

router = DefaultRouter()
router.register(r'vehicle-counts', VehicleCountViewSet)
router.register(r'signal-state', SignalStateViewSet)
router.register(r'predictions', PredictionViewSet)
router.register(r'logs', LogViewSet)

urlpatterns = [
    path('', include(router.urls)),
]