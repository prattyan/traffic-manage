from django.db.models import Avg, Max, Min, StdDev, Count
from django.utils import timezone
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import VehicleCount, SignalState, Prediction, Log, TrafficSnapshot
from .serializers import (
    VehicleCountSerializer,
    SignalStateSerializer,
    PredictionSerializer,
    LogSerializer,
    TrafficSnapshotSerializer,
)


class VehicleCountViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = VehicleCount.objects.all().order_by("-timestamp")
    serializer_class = VehicleCountSerializer


class SignalStateViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = SignalState.objects.all()
    serializer_class = SignalStateSerializer


class PredictionViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Prediction.objects.all().order_by("-created_at")
    serializer_class = PredictionSerializer


class LogViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Log.objects.all().order_by("-time")
    serializer_class = LogSerializer


class TrafficSnapshotViewSet(viewsets.ReadOnlyModelViewSet):
    """
    Read-only access to traffic snapshots for analytics.
    Snapshots are typically written by internal services or batch jobs.
    """
    queryset = TrafficSnapshot.objects.all().order_by("-timestamp")
    serializer_class = TrafficSnapshotSerializer

    @action(detail=False, methods=["get"], url_path="summary")
    def summary(self, request):
        """
        Analytics summary over TrafficSnapshot: mean, min, max, std, count.
        Optional query: ?hours=24 to limit to last N hours.
        """
        qs = TrafficSnapshot.objects.all()
        hours_raw = request.query_params.get("hours")
        try:
            hours = int(hours_raw) if hours_raw is not None else None
        except (TypeError, ValueError):
            hours = None
        if hours is not None and hours > 0:
            from datetime import timedelta
            since = timezone.now() - timedelta(hours=hours)
            qs = qs.filter(timestamp__gte=since)
        agg = qs.aggregate(
            count=Count("id"),
            mean_vehicle_count=Avg("vehicle_count"),
            std_vehicle_count=StdDev("vehicle_count"),
            min_vehicle_count=Min("vehicle_count"),
            max_vehicle_count=Max("vehicle_count"),
            mean_congestion=Avg("congestion_pct"),
        )
        # StdDev can be None on SQLite with few rows; round for readability
        if agg.get("mean_vehicle_count") is not None:
            agg["mean_vehicle_count"] = round(agg["mean_vehicle_count"], 2)
        if agg.get("std_vehicle_count") is not None:
            agg["std_vehicle_count"] = round(agg["std_vehicle_count"], 2)
        if agg.get("mean_congestion") is not None:
            agg["mean_congestion"] = round(agg["mean_congestion"], 2)
        return Response(agg)