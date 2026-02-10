"""
Traffic analytics module for data scientists and analysts.
Provides summary statistics, time-series metrics, and CSV export.
"""

from collections import deque
from datetime import datetime
import csv
import os
import time
from typing import Any, Dict, List, Optional, Union


def compute_summary_stats(values: Union[deque, List[float]]) -> Dict[str, float]:
    """
    Compute summary statistics for a sequence of numeric values (e.g. vehicle counts).
    Returns dict with mean, std, min, max, median, and percentiles.
    """
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "p25": 0.0,
            "p75": 0.0,
        }
    data = list(values)
    n = len(data)
    sorted_data = sorted(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n if n > 0 else 0
    std = variance ** 0.5

    def percentile(arr: List[float], p: float) -> float:
        if not arr:
            return 0.0
        k = (len(arr) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(arr) else f
        return arr[f] + (k - f) * (arr[c] - arr[f]) if f != c else arr[f]

    return {
        "count": n,
        "mean": round(mean, 2),
        "std": round(std, 2),
        "min": float(min(data)),
        "max": float(max(data)),
        "median": round(percentile(sorted_data, 50), 2),
        "p25": round(percentile(sorted_data, 25), 2),
        "p75": round(percentile(sorted_data, 75), 2),
    }


def compute_congestion_stats(
    traffic_history: Union[deque, List[int]], threshold_high: int = 10
) -> Dict[str, Any]:
    """
    Derive congestion-related metrics from traffic count history.
    threshold_high: count above which we consider "high" density.
    """
    if not traffic_history:
        return {
            "congestion_ratio": 0.0,
            "high_density_ratio": 0.0,
            "current_vs_mean": 0.0,
        }
    data = list(traffic_history)
    n = len(data)
    mean = sum(data) / n
    current = data[-1] if data else 0
    high_count = sum(1 for x in data if x >= threshold_high)
    # Congestion index 0-100 style
    max_seen = max(data) if data else 1
    congestion_ratio = (mean / max(15, max_seen)) * 100
    return {
        "congestion_ratio": round(min(100, congestion_ratio), 2),
        "high_density_ratio": round(100 * high_count / n, 2),
        "current_vs_mean": round(current / mean, 2) if mean else 0.0,
    }


def session_summary(
    traffic_history: Union[deque, List[int]],
    vehicle_types: Dict[str, int],
    start_time: float,
) -> Dict[str, Any]:
    """
    Build a session-level summary for the dashboard and reports.
    """
    stats = compute_summary_stats(traffic_history)
    congestion = compute_congestion_stats(traffic_history)
    elapsed_sec = max(0, time.time() - start_time)
    total_vehicles = sum(vehicle_types.values()) if vehicle_types else 0
    return {
        "traffic_stats": stats,
        "congestion": congestion,
        "elapsed_seconds": round(elapsed_sec, 1),
        "vehicle_breakdown": dict(vehicle_types) if vehicle_types else {},
        "total_vehicles_this_session": total_vehicles,
    }


def export_session_csv(
    rows: List[Dict[str, Any]],
    filepath: Optional[str] = None,
) -> str:
    """
    Export a list of record dicts (e.g. traffic snapshots) to CSV.
    Returns the path of the written file.
    """
    if not rows:
        raise ValueError("No rows to export")
    if filepath is None:
        os.makedirs("data", exist_ok=True)
        filepath = os.path.join(
            "data",
            f"traffic_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
    keys = list(rows[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return filepath
