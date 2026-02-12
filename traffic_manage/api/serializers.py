from rest_framework import serializers
from .models import VehicleCount, SignalState, Prediction, Log, TrafficSnapshot


class VehicleCountSerializer(serializers.ModelSerializer):
    class Meta:
        model = VehicleCount
        fields = "__all__"


class SignalStateSerializer(serializers.ModelSerializer):
    class Meta:
        model = SignalState
        fields = "__all__"


class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = "__all__"


class LogSerializer(serializers.ModelSerializer):
    class Meta:
        model = Log
        fields = "__all__"


class TrafficSnapshotSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrafficSnapshot
        fields = "__all__"