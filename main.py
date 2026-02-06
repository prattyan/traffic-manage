import cv2
import torch
import numpy as np
import time
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import threading
from threading import Lock
from dataclasses import dataclass, field
from typing import Dict, List


# Load pre-trained YOLO model for vehicle detection
yolo_model = YOLO("yolov8n.pt")  # Using YOLOv8 nano for real-time processing

# Load pre-trained LSTM model for traffic prediction
lstm_model = load_model("traffic_lstm.h5")


# Data class for per-intersection state
@dataclass
class IntersectionState:
    """Maintains state for a single traffic intersection."""
    intersection_id: str
    video_source: str
    traffic_data: List = field(default_factory=list)
    vehicle_count: int = 0
    emergency_detected: bool = False
    decision: str = "Waiting for data..."
    last_update: float = 0.0
    lock: Lock = field(default_factory=Lock)


# Multi-intersection manager
class TrafficManager:
    """Manages multiple traffic intersections with concurrent processing."""
    
    def __init__(self):
        self.intersections: Dict[str, IntersectionState] = {}
        self.processing_threads: Dict[str, threading.Thread] = {}
        self.master_lock = Lock()
        self.running = False
    
    def add_intersection(self, intersection_id: str, video_source: str):
        """Add a new intersection to monitor."""
        with self.master_lock:
            self.intersections[intersection_id] = IntersectionState(
                intersection_id=intersection_id,
                video_source=video_source
            )
    
    def get_intersection_data(self, intersection_id: str) -> IntersectionState:
        """Get state for a specific intersection."""
        return self.intersections.get(intersection_id)
    
    def get_all_intersections(self) -> Dict[str, IntersectionState]:
        """Get all intersections."""
        return self.intersections
    
    def start_processing(self):
        """Start processing threads for all intersections."""
        self.running = True
        for intersection_id, state in self.intersections.items():
            thread = threading.Thread(
                target=self._process_intersection,
                args=(intersection_id, state),
                daemon=True
            )
            thread.start()
            self.processing_threads[intersection_id] = thread
    
    def stop_processing(self):
        """Stop all processing threads."""
        self.running = False
        for thread in self.processing_threads.values():
            thread.join(timeout=5)
    
    def _process_intersection(self, intersection_id: str, state: IntersectionState):
        """Worker thread for a single intersection."""
        cap = cv2.VideoCapture(state.video_source)
        
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                continue
            
            # Object detection using YOLO
            results = yolo_model(frame)
            vehicle_count = 0
            emergency_detected = detect_emergency_vehicle(results)
            
            # Process detections
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    label = result.names[int(box.cls[0].item())]
                    
                    if label in ['car', 'bus', 'truck', 'motorbike'] and conf > 0.5:
                        vehicle_count += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    elif label in ['ambulance', 'fire truck', 'police car'] and conf > 0.5:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, "EMERGENCY", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Predict traffic congestion using LSTM model
            traffic_state = np.array([[vehicle_count]])
            predicted_traffic = lstm_model.predict(traffic_state, verbose=0)
            
            # Determine traffic light control decision
            decision = control_traffic(vehicle_count, emergency_detected)
            cv2.putText(frame, f'ID: {intersection_id}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f'Decision: {decision}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f'Vehicles: {vehicle_count}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Update intersection state (thread-safe)
            with state.lock:
                state.vehicle_count = vehicle_count
                state.emergency_detected = emergency_detected
                state.decision = decision
                state.last_update = time.time()
                state.traffic_data.append({
                    "timestamp": state.last_update,
                    "vehicle_count": vehicle_count,
                    "decision": decision
                })
                if len(state.traffic_data) > 30:
                    state.traffic_data.pop(0)
            
            # Display processed video (optional - comment out in production)
            # cv2.imshow(f"Traffic Monitoring - {intersection_id}", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        
        cap.release()
        cv2.destroyAllWindows()


# Define traffic light control logic
def control_traffic(vehicle_count, emergency_detected):
    if emergency_detected:
        return "Give Priority to Emergency Vehicle"
    elif vehicle_count > 50:
        return "Extend Green Light"
    elif vehicle_count < 10:
        return "Shorten Green Light"
    else:
        return "Normal Cycle"


def detect_emergency_vehicle(results):
    """Detect emergency vehicles in detection results."""
    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls[0].item())]
            if label in ['ambulance', 'fire truck', 'police car']:
                return True
    return False


# Initialize traffic manager
traffic_manager = TrafficManager()

# Configuration for multiple intersections
# Replace video sources with actual camera feeds (RTSP, HTTP, file paths, etc.)
INTERSECTIONS_CONFIG = {
    "intersection_1": "traffic_video.mp4",
    "intersection_2": "traffic_video.mp4",  # Replace with actual second feed
    "intersection_3": "traffic_video.mp4",  # Replace with actual third feed
}

# Add all intersections to manager
for intersection_id, video_source in INTERSECTIONS_CONFIG.items():
    traffic_manager.add_intersection(intersection_id, video_source)

# Initialize Dash app for real-time monitoring
dash_app = dash.Dash(__name__)

# Multi-intersection dashboard layout
dash_app.layout = html.Div([
    html.H1("Real-Time Multi-Intersection Traffic Monitoring Dashboard", style={'textAlign': 'center', 'marginBottom': 30}),
    
    # Container for all intersection cards
    html.Div(
        id='intersections-container',
        style={
            'display': 'grid',
            'gridTemplateColumns': 'repeat(auto-fit, minmax(500px, 1fr))',
            'gap': '20px',
            'marginBottom': '30px'
        }
    ),
    
    # Summary stats
    html.Div(
        id='summary-stats',
        style={
            'display': 'grid',
            'gridTemplateColumns': 'repeat(auto-fit, minmax(250px, 1fr))',
            'gap': '15px',
            'marginBottom': '30px'
        }
    ),
    
    # Update interval
    dcc.Interval(id='interval-component', interval=2000, n_intervals=0)
])


# Callback to update all intersection cards and summary
@dash_app.callback(
    [Output('intersections-container', 'children'), 
     Output('summary-stats', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    """Update dashboard with data from all intersections."""
    all_intersections = traffic_manager.get_all_intersections()
    
    if not all_intersections:
        return [html.Div("No intersections configured")], [html.Div("No data available")]
    
    # Create cards for each intersection
    intersection_cards = []
    total_vehicles = 0
    emergency_count = 0
    
    for intersection_id, state in all_intersections.items():
        with state.lock:
            if not state.traffic_data:
                continue
            
            total_vehicles += state.vehicle_count
            if state.emergency_detected:
                emergency_count += 1
            
            # Extract data for graphs
            timestamps = [
                time.strftime('%H:%M:%S', time.localtime(d['timestamp']))
                for d in state.traffic_data
            ]
            vehicle_counts = [d['vehicle_count'] for d in state.traffic_data]
            
            # Create card for this intersection
            card = html.Div(
                [
                    html.H3(f"Intersection: {intersection_id}", style={'color': '#2c3e50'}),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H4("Vehicle Count", style={'margin': '0 0 10px 0'}),
                                    html.H2(state.vehicle_count, style={'color': '#3498db', 'margin': 0})
                                ],
                                style={'flex': '1'}
                            ),
                            html.Div(
                                [
                                    html.H4("Status", style={'margin': '0 0 10px 0'}),
                                    html.P(
                                        state.decision,
                                        style={
                                            'color': '#e74c3c' if state.emergency_detected else '#27ae60',
                                            'fontWeight': 'bold',
                                            'margin': 0
                                        }
                                    )
                                ],
                                style={'flex': '1'}
                            ),
                            html.Div(
                                [
                                    html.H4("Emergency", style={'margin': '0 0 10px 0'}),
                                    html.P(
                                        "🚨 ACTIVE" if state.emergency_detected else "✓ Clear",
                                        style={
                                            'color': '#e74c3c' if state.emergency_detected else '#27ae60',
                                            'fontWeight': 'bold',
                                            'margin': 0
                                        }
                                    )
                                ],
                                style={'flex': '1'}
                            )
                        ],
                        style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}
                    ),
                    dcc.Graph(
                        figure={
                            "data": [
                                {"x": timestamps, "y": vehicle_counts, "type": "line", "name": "Vehicles", "line": {"color": '#3498db'}}
                            ],
                            "layout": {
                                "title": f"Traffic Volume - {intersection_id}",
                                "height": 300,
                                "margin": {"l": 40, "r": 20, "t": 40, "b": 40}
                            }
                        }
                    )
                ],
                style={
                    'border': '2px solid #ecf0f1',
                    'borderRadius': '8px',
                    'padding': '20px',
                    'backgroundColor': '#f8f9fa'
                }
            )
            intersection_cards.append(card)
    
    # Summary statistics
    summary_cards = [
        html.Div(
            [
                html.H4("Total Vehicles", style={'margin': '0 0 10px 0'}),
                html.H2(total_vehicles, style={'color': '#3498db', 'margin': 0})
            ],
            style={
                'border': '1px solid #ecf0f1',
                'borderRadius': '8px',
                'padding': '15px',
                'backgroundColor': '#ecf7ff',
                'textAlign': 'center'
            }
        ),
        html.Div(
            [
                html.H4("Active Intersections", style={'margin': '0 0 10px 0'}),
                html.H2(len(all_intersections), style={'color': '#27ae60', 'margin': 0})
            ],
            style={
                'border': '1px solid #ecf0f1',
                'borderRadius': '8px',
                'padding': '15px',
                'backgroundColor': '#ecfde6',
                'textAlign': 'center'
            }
        ),
        html.Div(
            [
                html.H4("Emergency Alerts", style={'margin': '0 0 10px 0'}),
                html.H2(emergency_count, style={'color': '#e74c3c', 'margin': 0})
            ],
            style={
                'border': '1px solid #ecf0f1',
                'borderRadius': '8px',
                'padding': '15px',
                'backgroundColor': '#ffe8e8',
                'textAlign': 'center'
            }
        )
    ]
    
    return intersection_cards, summary_cards


if __name__ == "__main__":
    # Start processing all intersections
    traffic_manager.start_processing()
    
    try:
        dash_app.run_server(debug=False, host='0.0.0.0', port=8050)
    finally:
        traffic_manager.stop_processing()
