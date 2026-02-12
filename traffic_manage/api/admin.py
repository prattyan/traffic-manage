from django.contrib import admin
from .models import VehicleCount, SignalState, Prediction, Log, TrafficSnapshot


@admin.register(VehicleCount)
class VehicleCountAdmin(admin.ModelAdmin):
    list_display = ("timestamp", "cars", "bikes", "buses")


@admin.register(SignalState)
class SignalStateAdmin(admin.ModelAdmin):
    list_display = ("intersection", "state", "updated_at")


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ("description", "created_at")


@admin.register(Log)
class LogAdmin(admin.ModelAdmin):
    list_display = ("time", "cars", "bikes", "buses")


@admin.register(TrafficSnapshot)
class TrafficSnapshotAdmin(admin.ModelAdmin):
    list_display = ("timestamp", "vehicle_count", "cars", "trucks", "bikes", "pedestrians", "congestion_pct", "decision")
    list_filter = ("decision",)
    date_hierarchy = "timestamp"
