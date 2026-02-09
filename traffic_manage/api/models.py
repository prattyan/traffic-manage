from django.db import models
from django.utils import timezone

# Create your models here.
class VehicleCount(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    cars = models.IntegerField(default=0)
    bikes = models.IntegerField(default=0)
    buses = models.IntegerField(default=0)

class SignalState(models.Model):
    intersection = models.CharField(max_length=50)
    state = models.CharField(max_length=10)  # e.g. GREEN, RED, YELLOW
    updated_at = models.DateTimeField(auto_now=True)

class Prediction(models.Model):
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

class Log(models.Model):
    time = models.DateTimeField()
    cars = models.IntegerField()
    bikes = models.IntegerField()
    buses = models.IntegerField()


class TrafficSnapshot(models.Model):
    """
    Time-series snapshot for analytics: vehicle counts and decision at a point in time.
    Populated by the Dash app when API is available; supports historical analysis.
    """
    timestamp = models.DateTimeField(default=timezone.now, db_index=True)
    vehicle_count = models.IntegerField(default=0)
    cars = models.IntegerField(default=0)
    trucks = models.IntegerField(default=0)
    bikes = models.IntegerField(default=0)
    pedestrians = models.IntegerField(default=0)
    congestion_pct = models.IntegerField(default=0)
    decision = models.CharField(max_length=64, blank=True)

    class Meta:
        ordering = ["-timestamp"]

