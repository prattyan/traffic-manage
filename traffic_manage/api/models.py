from django.db import models

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

