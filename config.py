import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()


class Config:
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Dashboard
    DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "127.0.0.1")
    DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", 8050))
    DASHBOARD_DEBUG = os.getenv("DASHBOARD_DEBUG", "False").lower() == "true"
    DASHBOARD_UPDATE_INTERVAL = int(os.getenv("DASHBOARD_UPDATE_INTERVAL", 2000))

    # YOLO / Detection
    YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
    DETECTION_CONFIDENCE = float(os.getenv("DETECTION_CONFIDENCE", 0.5))
    NMS_THRESHOLD = float(os.getenv("NMS_THRESHOLD", 0.5))

    # Video
    VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")

    @classmethod
    def print_config(cls):
        print("===== Runtime Configuration =====")
        for key, value in cls.__dict__.items():
            if key.isupper():
                print(f"{key} = {value}")
        print("=================================")