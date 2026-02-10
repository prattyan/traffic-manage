# Generated migration for TrafficSnapshot (analytics support)

from django.db import migrations, models
from django.utils import timezone


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="TrafficSnapshot",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("timestamp", models.DateTimeField(default=timezone.now, db_index=True)),
                ("vehicle_count", models.IntegerField(default=0)),
                ("cars", models.IntegerField(default=0)),
                ("trucks", models.IntegerField(default=0)),
                ("bikes", models.IntegerField(default=0)),
                ("pedestrians", models.IntegerField(default=0)),
                ("congestion_pct", models.IntegerField(default=0)),
                ("decision", models.CharField(blank=True, max_length=64)),
            ],
            options={
                "ordering": ["-timestamp"],
            },
        ),
    ]
