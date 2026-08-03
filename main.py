"""
🚦 TRAFFIC AI COMMAND CENTER v4.0 (Multi-Intersection)
Advanced Real-Time Traffic Management System
Features: YOLOv8 Detection, LSTM Prediction, Glassmorphic UI, Multi-Camera Support
"""

import cv2
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from ultralytics import YOLO
import threading
import time
import os
os.environ['HEADLESS'] = '1'
from datetime import datetime
from collections import deque
import plotly.graph_objs as go
import plotly.express as px
import os
from flask import Response
import base64
import uuid

os.makedirs("cache", exist_ok=True)

# --- GLOBAL SHARED MEMORY (Per Intersection) ---
class IntersectionState:
    def __init__(self, id, name, camera_source):
        self.id = id
        self.name = name
        self.camera_source = camera_source
        self.latest_frame = None
        self.raw_frame = None
        self.latest_boxes = []
        self.traffic_history = deque(maxlen=100)
        self.vehicle_types = {"cars": 0, "trucks": 0, "bikes": 0, "pedestrians": 0}
        self.emergency_log = deque(maxlen=20)
        self.current_decision = "Initializing..."
        self.system_status = {"fps": 0, "uptime": 0, "detections": 0}
        self.start_time = time.time()

# # Define multiple intersections here
# intersections = {
#     "cam1": IntersectionState("cam1", "Broadway & 42nd St", "traffic_video.mp4") 
# }
intersections = {}

# --- LOAD MODELS ---
print("🔄 Loading YOLO Model...")
yolo_model = YOLO("yolov8n.pt")
print("✅ YOLO Loaded Successfully")

# Simulated Brain (Fallback)
class TrafficBrain:
    def predict(self, data):
        # Simple prediction based on trend
        if len(data) > 5:
            trend = np.mean(data[-5:]) - np.mean(data[-10:-5]) if len(data) > 10 else 0
            return [[1]] if trend > 0 else [[0]]
        return [[0]]

print("⚠️ Using Simulated Brain (Fallback)")
lstm_model = TrafficBrain()

# --- HELPER FUNCTIONS ---
def control_traffic(vehicle_count, emergency_detected, congestion_level):
    if emergency_detected:
        return "🚨 EMERGENCY PRIORITY", "danger"
    elif congestion_level > 80:
        return "🔴 CRITICAL CONGESTION", "danger"
    elif vehicle_count > 8:
        return "🟠 EXTEND GREEN", "warning"
    elif vehicle_count > 5:
        return "🟡 MONITOR CLOSELY", "warning"
    elif vehicle_count < 2:
        return "🟢 OPTIMIZE FLOW", "success"
    else:
        return "🔵 NORMAL CYCLE", "info"

def calculate_congestion(history):
    if len(history) < 5:
        return 0
    recent = list(history)[-10:]
    return min(100, int((np.mean(recent) / 15) * 100))

def detect_emergency_vehicle(frame, boxes):
    # Analyze bounding boxes for bright "Emergency Red" to classify fire trucks/ambulances
    for (x1, y1, x2, y2, label, conf, category) in boxes:
        if category in ['trucks', 'cars'] and conf > 0.4:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            lower_red1 = np.array([0, 120, 70])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 120, 70])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = mask1 + mask2
            
            red_ratio = cv2.countNonZero(red_mask) / (crop.shape[0] * crop.shape[1] + 1)
            if red_ratio > 0.25:
                return True, "Fire Truck / Ambulance"
    return False, None

def predict_traffic(history):
    history = list(history)
    if len(history) < 10:
        return history[-5:] if len(history) >= 5 else [0] * 5
        
    try:
        # Use simple linear regression on the last 30 seconds of traffic data
        recent = history[-30:] if len(history) > 30 else history
        x = np.arange(len(recent))
        y = np.array(recent)
        
        # Calculate slope (m) and intercept (c)
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        
        predictions = []
        moving_avg = np.mean(recent[-5:])
        for i in range(1, 6):
            # Project trend forward (scale time steps)
            next_x = len(recent) - 1 + (i * 3) 
            pred = m * next_x + c
            # Blend trend with moving average for stability
            pred = (pred * 0.5) + (moving_avg * 0.5)
            predictions.append(max(0, pred))
            
        return predictions
    except:
        return [np.mean(history[-5:])] * 5

# --- CUSTOM CSS FOR GLASSMORPHISM ---
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=Inter:wght@400;500;600&display=swap');

:root {
    --primary: #3b82f6;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --surface: rgba(30, 41, 59, 0.7);
    --bg-main: #0f172a;
    --border: rgba(255, 255, 255, 0.08);
}

body {
    background-color: var(--bg-main) !important;
    background-image: radial-gradient(circle at top right, rgba(59, 130, 246, 0.1), transparent 40%), radial-gradient(circle at bottom left, rgba(16, 185, 129, 0.05), transparent 40%) !important;
    font-family: 'Inter', sans-serif !important;
    min-height: 100vh;
    color: #f8fafc !important;
}

.glass-card {
    background: var(--surface) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 24px -1px rgba(0, 0, 0, 0.2) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease !important;
    overflow: hidden;
}

.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 32px -4px rgba(0, 0, 0, 0.3) !important;
    border-color: rgba(59, 130, 246, 0.3) !important;
}

.neon-text {
    font-family: 'Outfit', sans-serif !important;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

.status-indicator {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
}

.status-online { background: var(--success); box-shadow: 0 0 12px var(--success); }
.status-warning { background: var(--warning); box-shadow: 0 0 12px var(--warning); }
.status-danger { background: var(--danger); box-shadow: 0 0 12px var(--danger); animation: pulse-danger 1s infinite; }

@keyframes pulse-danger {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.metric-value {
    font-family: 'Outfit', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    color: #f8fafc;
}

.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #94a3b8;
    font-weight: 600;
}

.decision-badge {
    font-family: 'Outfit', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    padding: 12px 24px;
    border-radius: 8px;
    width: 100%;
}

.navbar-glass {
    background: rgba(15, 23, 42, 0.8) !important;
    backdrop-filter: blur(12px) !important;
    border-bottom: 1px solid var(--border) !important;
}

.cmd-log {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #cbd5e1;
    background: rgba(0, 0, 0, 0.3);
    padding: 15px;
    border-radius: 12px;
    max-height: 200px;
    overflow-y: auto;
    border: 1px solid var(--border);
}

.nav-tabs .nav-link {
    color: #94a3b8;
    border: none;
    border-bottom: 2px solid transparent;
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    padding: 10px 20px;
}

.nav-tabs .nav-link.active {
    color: #60a5fa;
    background-color: transparent;
    border-color: transparent;
    border-bottom: 2px solid #60a5fa;
}

/* Custom scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.1); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.2); }
"""

# --- MODERN DASHBOARD SETUP ---
dash_app = dash.Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=Inter:wght@400;500;600&family=JetBrains+Mono&display=swap"
    ],
    suppress_callback_exceptions=True
)

dash_app.index_string = f'''
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>Traffic AI Command Center</title>
        {{%favicon%}}
        {{%css%}}
        <style>{CUSTOM_CSS}</style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
'''

def create_metric_card(title, value_id, icon, color_class="text-primary"):
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Span(icon, style={'fontSize': '1.8rem'}),
                html.Span(className="status-indicator status-online", style={'marginLeft': '12px'})
            ], className="d-flex align-items-center mb-2"),
            html.H2(id=value_id, className=f"metric-value mb-1 {color_class}"),
            html.P(title, className="metric-label mb-0")
        ], className="p-4")
    ], className="glass-card h-100")

def create_status_card(title, status, description_id, default_desc, icon):
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Span(icon, style={'fontSize': '1.2rem', 'marginRight': '10px'}),
                html.Span(title, style={'fontWeight': '600', 'fontSize': '0.95rem'})
            ], className="d-flex align-items-center mb-3 text-light"),
            html.Div([
                html.Span(className="status-indicator status-online"),
                html.Span(status, style={'color': '#10b981', 'fontWeight': '600', 'fontSize': '0.9rem'})
            ], className="mb-2"),
            html.P(default_desc, id=description_id, className="text-muted small mb-0")
        ], className="p-3")
    ], className="glass-card mb-3")

dash_app.layout = dbc.Container([
    # Header
    dbc.Navbar([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("🚦", style={'fontSize': '1.8rem', 'marginRight': '12px'}),
                        html.Span("TRAFFIC AI", className="neon-text", style={'fontSize': '1.6rem', 'marginRight': '8px'}),
                        html.Span("Command Center", style={'fontSize': '1.2rem', 'color': '#94a3b8', 'fontWeight': '400', 'fontFamily': 'Outfit'})
                    ], className="d-flex align-items-center")
                ], width="auto"),
                dbc.Col([
                    html.Div([
                        html.Span("v4.0", className="badge bg-primary me-3", style={'fontWeight': '500'}),
                        html.Span(id="current-time", className="text-muted", style={'fontFamily': 'JetBrains Mono', 'fontSize': '0.9rem'})
                    ])
                ], width="auto", className="ms-auto")
            ], align="center", className="w-100")
        ], fluid=True)
    ], className="navbar-glass mb-4 py-3"),

    # Intersection Tabs and Add Stream Button
    dbc.Row([
        dbc.Col([
            dbc.Tabs(
                id="intersection-tabs",
                active_tab=list(intersections.keys())[0] if intersections else None,
                children=[
                    dbc.Tab(label=state.name, tab_id=iid) for iid, state in intersections.items()
                ],
                className="mb-0"
            )
        ], width="auto"),
        dbc.Col([
            dbc.Button("＋ Add Stream", id="btn-add-stream", color="primary", className="fw-bold text-light", style={'fontFamily': 'Outfit', 'borderRadius': '8px'})
        ], width="auto", className="ms-auto")
    ], className="mb-4 d-flex align-items-center"),

    # Add Stream Modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Add New Camera Stream", style={'fontFamily': 'Outfit', 'fontWeight': '600'})),
        dbc.ModalBody([
            dbc.Label("Street Name / Location", className="text-light"),
            dbc.Input(id="input-street-name", type="text", placeholder="e.g. 5th Ave & 42nd St", className="mb-4 bg-dark text-light border-secondary"),
            
            dbc.Label("Source Type", className="text-light"),
            dbc.RadioItems(
                id="input-source-type",
                options=[
                    {"label": " IP Camera (RTSP/HTTP)", "value": "ip"},
                    {"label": " Video File Upload", "value": "upload"}
                ],
                value="upload",
                inline=True,
                className="mb-4 text-light"
            ),
            
            html.Div(id="ip-fields", children=[
                dbc.Label("IP Stream URL", className="text-light"),
                dbc.Input(id="input-ip-url", type="text", placeholder="rtsp://user:pass@ip:port/stream", className="mb-3 bg-dark text-light border-secondary"),
            ], style={'display': 'none'}),
            
            html.Div(id="upload-fields", children=[
                dbc.Label("Upload Video File", className="text-light"),
                dcc.Upload(
                    id='upload-video',
                    children=html.Div(['Drag and Drop or ', html.A('Select a Video File')]),
                    style={
                        'width': '100%', 'height': '80px', 'lineHeight': '80px',
                        'borderWidth': '2px', 'borderStyle': 'dashed',
                        'borderRadius': '8px', 'textAlign': 'center', 'marginBottom': '10px',
                        'borderColor': '#3b82f6', 'color': '#cbd5e1', 'cursor': 'pointer'
                    },
                    multiple=False
                ),
                html.Div(id='upload-status', className="text-info small mb-3")
            ], style={'display': 'block'}),
            
        ], style={'background': '#0f172a'}),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="btn-cancel-stream", className="ms-auto", color="secondary"),
            dbc.Button("Start Feed", id="btn-submit-stream", color="primary")
        ], style={'background': '#0f172a'})
    ], id="modal-add-stream", is_open=False, backdrop="static", centered=True),

    # Row 1: Key Metrics (4 cols)
    dbc.Row([
        dbc.Col(create_metric_card("Live Vehicles", "vehicle-count", "🚗", "text-info"), lg=3, md=6, className="mb-4"),
        dbc.Col(create_metric_card("Congestion", "congestion-level", "📊", "text-warning"), lg=3, md=6, className="mb-4"),
        dbc.Col(create_metric_card("System FPS", "fps-value", "⚡", "text-success"), lg=3, md=6, className="mb-4"),
        dbc.Col(create_metric_card("Uptime (Min)", "uptime-value", "⏱️", "text-primary"), lg=3, md=6, className="mb-4"),
    ]),

    # Row 2: Split View (Left: Video, Right: Status & Logs)
    dbc.Row([
        # Left Side - Video Feed
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.Span("🎥", style={'marginRight': '10px'}),
                        html.Span("LIVE CAMERA FEED", style={'fontWeight': '600', 'fontFamily': 'Outfit', 'letterSpacing': '1px', 'fontSize': '0.85rem', 'color': '#94a3b8'})
                    ], className="d-flex align-items-center")
                ], style={'background': 'transparent', 'borderBottom': '1px solid var(--border)', 'padding': '15px 20px'}),
                dbc.CardBody([
                    html.Img(id='video-stream', style={'width': '100%', 'height': 'auto', 'maxHeight': '550px', 'objectFit': 'contain', 'borderRadius': '8px', 'background': '#000'})
                ], className="p-3 text-center")
            ], className="glass-card h-100")
        ], lg=8, className="mb-4"),
        
        # Right Side - Status Cards & Logs
        dbc.Col([
            html.Div([
                create_status_card("Camera Feed", "ACTIVE", "camera-status-desc", "ID: CAM-01 | 1080p @ 30fps", "📍"),
                create_status_card("AI Engine", "ONLINE", "ai-status-desc", "YOLOv8n Inference", "🧠"),
                
                # Live Decision
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Span("⚡", style={'fontSize': '1.2rem', 'marginRight': '10px'}),
                            html.Span("AI DECISION", style={'fontWeight': '600', 'fontSize': '0.95rem'})
                        ], className="d-flex align-items-center mb-3 text-light"),
                        html.Div(id="decision-badge", className="text-center")
                    ], className="p-3")
                ], className="glass-card mb-3"),
                
                # Congestion Bar
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Span("NETWORK CONGESTION", className="metric-label")
                        ], className="mb-2"),
                        dbc.Progress(id="congestion-bar", value=0, className="mb-2", style={'height': '8px', 'background': 'rgba(255,255,255,0.05)'}),
                        html.Div([
                            html.Span("0%", className="text-success small"),
                            html.Span(id="congestion-status", className="small text-center"),
                            html.Span("100%", className="text-danger small")
                        ], className="d-flex justify-content-between align-items-center")
                    ], className="p-3")
                ], className="glass-card mb-3"),
                
                # Logs
                dbc.Card([
                    dbc.CardHeader("SYSTEM LOGS", style={'background': 'transparent', 'borderBottom': '1px solid var(--border)', 'fontFamily': 'Outfit', 'fontSize': '0.75rem', 'letterSpacing': '1px', 'color': '#94a3b8', 'padding': '10px 15px'}),
                    dbc.CardBody([
                        html.Div(id="system-logs", className="cmd-log", style={'maxHeight': '140px'})
                    ], className="p-2")
                ], className="glass-card")
            ], className="d-flex flex-column h-100 justify-content-between")
        ], lg=4, className="mb-4")
    ]),

    # Row 3: Main Charts
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.Span("📈", style={'marginRight': '10px'}),
                        html.Span("TRAFFIC DENSITY", style={'fontWeight': '600', 'fontFamily': 'Outfit', 'letterSpacing': '1px', 'fontSize': '0.85rem', 'color': '#94a3b8'})
                    ], className="d-flex align-items-center")
                ], style={'background': 'transparent', 'borderBottom': '1px solid var(--border)', 'padding': '15px 20px'}),
                dbc.CardBody([
                    dcc.Graph(id='traffic-graph', style={'height': '300px'}, config={'displayModeBar': False})
                ], className="p-3")
            ], className="glass-card")
        ], lg=5, className="mb-4"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.Span("🔮", style={'marginRight': '10px'}),
                        html.Span("AI PREDICTION (5 MIN)", style={'fontWeight': '600', 'fontFamily': 'Outfit', 'letterSpacing': '1px', 'fontSize': '0.85rem', 'color': '#94a3b8'})
                    ], className="d-flex align-items-center")
                ], style={'background': 'transparent', 'borderBottom': '1px solid var(--border)', 'padding': '15px 20px'}),
                dbc.CardBody([
                    dcc.Graph(id='prediction-graph', style={'height': '300px'}, config={'displayModeBar': False})
                ], className="p-3")
            ], className="glass-card")
        ], lg=4, className="mb-4"),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.Span("🚙", style={'marginRight': '10px'}),
                        html.Span("CLASSIFICATION", style={'fontWeight': '600', 'fontFamily': 'Outfit', 'letterSpacing': '1px', 'fontSize': '0.85rem', 'color': '#94a3b8'})
                    ], className="d-flex align-items-center")
                ], style={'background': 'transparent', 'borderBottom': '1px solid var(--border)', 'padding': '15px 20px'}),
                dbc.CardBody([
                    dcc.Graph(id='vehicle-pie', style={'height': '300px'}, config={'displayModeBar': False})
                ], className="p-3")
            ], className="glass-card")
        ], lg=3, className="mb-4"),
    ]),

    # Footer
    dbc.Row([
        dbc.Col([
            html.Div([
                html.P([
                    "Traffic AI Command Center | Premium Edition"
                ], className="text-center mb-0", style={'color': '#64748b', 'fontSize': '0.85rem'})
            ], className="py-4")
        ])
    ]),

    # Internal Clock
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0),
    dcc.Interval(id='fast-interval', interval=500, n_intervals=0)

], fluid=True, style={'padding': '30px', 'maxWidth': '1600px'})

def generate_frames(intersection_id):
    last_frame_id = None
    while True:
        state = intersections.get(intersection_id)
        if state and state.latest_frame is not None:
            current_frame_id = id(state.latest_frame)
            if current_frame_id != last_frame_id:
                # Compress JPEG to 70% quality to reduce network bottleneck
                ret, buffer = cv2.imencode('.jpg', state.latest_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                last_frame_id = current_frame_id
        time.sleep(0.033) # Throttle to ~30 FPS explicitly

@dash_app.server.route('/video_feed/<intersection_id>')
def video_feed(intersection_id):
    return Response(generate_frames(intersection_id), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- AI WORKER THREAD ---
def ai_worker(intersection_id):
    state = intersections[intersection_id]
    class_mapping = {2: 'cars', 3: 'bikes', 5: 'trucks', 7: 'trucks', 0: 'pedestrians', 1: 'bikes'}
    while True:
        if state.raw_frame is None:
            time.sleep(0.05)
            continue
        frame_copy = state.raw_frame.copy()
        try:
            # High Accuracy Inference (imgsz=640)
            results = list(yolo_model(frame_copy, verbose=False, stream=True, imgsz=640, conf=0.25))
            cars, trucks, bikes, pedestrians, vehicle_count = 0, 0, 0, 0, 0
            new_boxes = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    if class_id in class_mapping:
                        vehicle_count += 1
                        category = class_mapping[class_id]
                        if category == 'cars': cars += 1
                        elif category == 'trucks': trucks += 1
                        elif category == 'bikes': bikes += 1
                        elif category == 'pedestrians': pedestrians += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = result.names[class_id]
                        new_boxes.append((x1, y1, x2, y2, label, float(box.conf[0]), category))
            state.latest_boxes = new_boxes
            state.vehicle_types = {"cars": cars, "trucks": trucks, "bikes": bikes, "pedestrians": pedestrians}
            emergency, vehicle_type = detect_emergency_vehicle(frame_copy, new_boxes)
            if emergency:
                state.emergency_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 EMERGENCY: {vehicle_type} detected!")
            state.traffic_history.append(vehicle_count)
            state.system_status["detections"] += vehicle_count
            state.system_status["uptime"] = int((time.time() - state.start_time) / 60)
            congestion = calculate_congestion(state.traffic_history)
            state.current_decision, _ = control_traffic(vehicle_count, emergency, congestion)
        except Exception as e:
            print(f"❌ Error in AI loop for {intersection_id}: {e}")
        
        # THROTTLE AI WORKER to max 5 FPS to free up CPU for video playback
        time.sleep(0.2)

# --- VIDEO THREAD (MULTI-CAMERA SUPPORT) ---
def process_traffic(intersection_id):
    state = intersections[intersection_id]
    print(f"📸 STARTING CAMERA {state.camera_source} for {state.name}...")
    if isinstance(state.camera_source, int) and os.name == 'nt':
        cap = cv2.VideoCapture(state.camera_source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(state.camera_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame_count, fps_start = 0, time.time()
    
    target_fps = 60
    frame_time = 1.0 / target_fps
    
    while True:
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        state.raw_frame = frame.copy()
        for (x1, y1, x2, y2, label, conf, category) in state.latest_boxes:
            color = (0, 255, 136) if category != 'pedestrians' else (255, 107, 53)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        frame_count += 1
        if time.time() - fps_start >= 1:
            state.system_status["fps"] = frame_count
            frame_count, fps_start = 0, time.time()
        cv2.putText(frame, f"FPS: {state.system_status.get('fps', 0)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        state.latest_frame = frame
        
        # Regulate FPS explicitly
        elapsed = time.time() - loop_start
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)

# --- CALLBACKS ---
@dash_app.callback(
    [
        Output('video-stream', 'src'),
        Output('traffic-graph', 'figure'),
        Output('vehicle-pie', 'figure'),
        Output('prediction-graph', 'figure'),
        Output('vehicle-count', 'children'),
        Output('congestion-level', 'children'),
        Output('fps-value', 'children'),
        Output('uptime-value', 'children'),
        Output('decision-badge', 'children'),
        Output('congestion-bar', 'value'),
        Output('congestion-bar', 'color'),
        Output('congestion-status', 'children'),
        Output('current-time', 'children'),
        Output('system-logs', 'children'),
        Output('camera-status-desc', 'children')
    ],
    [
        Input('interval-component', 'n_intervals'),
        Input('intersection-tabs', 'active_tab')
    ]
)
def update_dashboard(n, active_tab):
    if not intersections:
        empty_fig = go.Figure()
        empty_fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        return ("", empty_fig, empty_fig, empty_fig, "0", "0%", "0", "0", 
                html.Div("NO FEED ACTIVE", className="decision-badge", style={'color': '#cbd5e1', 'border': '1px solid #cbd5e1'}), 
                0, "success", html.Span("NO FEED", style={'color': '#cbd5e1'}), 
                datetime.now().strftime("%b %d, %Y | %H:%M:%S"), 
                html.Div("Please add a camera feed to start."), "ID: N/A | Source: N/A")

    if not active_tab or active_tab not in intersections:
        active_tab = list(intersections.keys())[0]
    state = intersections[active_tab]
    history_list = list(state.traffic_history)
    
    # Modern Colors
    primary_color = '#3b82f6'
    bg_transparent = 'rgba(0,0,0,0)'
    grid_color = 'rgba(255,255,255,0.05)'
    text_color = '#cbd5e1'
    
    traffic_fig = go.Figure()
    traffic_fig.add_trace(go.Scatter(x=list(range(len(history_list))), y=history_list, mode='lines', fill='tozeroy', line=dict(color=primary_color, width=3), fillcolor='rgba(59, 130, 246, 0.15)'))
    traffic_fig.update_layout(plot_bgcolor=bg_transparent, paper_bgcolor=bg_transparent, font=dict(color=text_color, family='Inter'), xaxis=dict(showgrid=False, title='Time (s)'), yaxis=dict(gridcolor=grid_color, title='Count', range=[0, max(15, max(history_list) + 2) if history_list else 15]), margin=dict(l=40, r=20, t=10, b=40), showlegend=False)
    
    pie_colors = ['#3b82f6', '#f59e0b', '#10b981', '#8b5cf6']
    pie_fig = go.Figure()
    pie_fig.add_trace(go.Pie(labels=['Cars', 'Trucks', 'Bikes', 'Peds'], values=[state.vehicle_types['cars'], state.vehicle_types['trucks'], state.vehicle_types['bikes'], state.vehicle_types['pedestrians']], hole=0.65, marker=dict(colors=pie_colors, line=dict(color='#1e293b', width=2)), textinfo='percent+label', textfont=dict(size=11, family='Inter')))
    pie_fig.update_layout(plot_bgcolor=bg_transparent, paper_bgcolor=bg_transparent, font=dict(color=text_color), showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    
    predictions = predict_traffic(history_list) if len(history_list) > 5 else [0]*5
    pred_fig = go.Figure()
    pred_fig.add_trace(go.Bar(x=['1m', '2m', '3m', '4m', '5m'], y=predictions, marker=dict(color=['#60a5fa', '#3b82f6', '#2563eb', '#1d4ed8', '#1e40af'], line=dict(width=0)), text=[f'{p:.1f}' for p in predictions], textposition='outside', textfont=dict(color=text_color, family='Inter')))
    pred_fig.update_layout(plot_bgcolor=bg_transparent, paper_bgcolor=bg_transparent, font=dict(color=text_color), xaxis=dict(showgrid=False), yaxis=dict(gridcolor=grid_color, range=[0, max(15, max(predictions) + 2) if predictions else 15]), margin=dict(l=30, r=20, t=20, b=30), showlegend=False)
    
    vehicle_count = history_list[-1] if history_list else 0
    congestion = calculate_congestion(history_list)
    fps, uptime = state.system_status.get('fps', 0), state.system_status.get('uptime', 0)
    decision_text, decision_type = control_traffic(vehicle_count, False, congestion)
    
    # Modern Badge Colors
    color_map = {'danger': '#ef4444', 'warning': '#f59e0b', 'success': '#10b981', 'info': '#3b82f6'}
    badge_color = color_map.get(decision_type, '#3b82f6')
    decision_badge = html.Div(decision_text, className="decision-badge", style={'background': f'rgba({int(badge_color[1:3], 16)}, {int(badge_color[3:5], 16)}, {int(badge_color[5:7], 16)}, 0.1)', 'border': f'1px solid {badge_color}', 'color': badge_color})
    
    bar_color = "success" if congestion < 30 else "warning" if congestion < 70 else "danger"
    congestion_status = html.Span(f"{'LOW' if congestion < 30 else 'MODERATE' if congestion < 70 else 'HIGH'}", style={'color': '#10b981' if congestion < 30 else '#f59e0b' if congestion < 70 else '#ef4444', 'fontWeight': '700'})
    
    current_time = datetime.now().strftime("%b %d, %Y | %H:%M:%S")
    default_logs = [f"[{datetime.now().strftime('%H:%M:%S')}] ✅ System operational", f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 Stream active @ {fps} FPS", f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Total logged: {state.system_status.get('detections', 0)}"]
    all_logs = state.emergency_log[-10:] + default_logs if state.emergency_log else default_logs
    logs_display = html.Div([html.Div(log, className="mb-1 text-truncate") for log in all_logs[-7:]])
    
    return (f"/video_feed/{active_tab}", traffic_fig, pie_fig, pred_fig, str(vehicle_count), f"{congestion}%", str(fps), str(uptime), decision_badge, congestion, bar_color, congestion_status, current_time, logs_display, f"ID: {state.id.upper()} | Source: {state.camera_source}")

# --- ADD STREAM CALLBACKS ---
@dash_app.callback(
    [Output("modal-add-stream", "is_open"), Output('upload-status', 'children')],
    [Input("btn-add-stream", "n_clicks"), Input("btn-cancel-stream", "n_clicks")],
    [State("modal-add-stream", "is_open")]
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open, ""
    return is_open, ""

@dash_app.callback(
    [Output("ip-fields", "style"), Output("upload-fields", "style")],
    [Input("input-source-type", "value")]
)
def toggle_fields(source_type):
    if source_type == "ip":
        return {'display': 'block'}, {'display': 'none'}
    return {'display': 'none'}, {'display': 'block'}

@dash_app.callback(
    [Output("intersection-tabs", "children"),
     Output("intersection-tabs", "active_tab")],
    [Input("btn-submit-stream", "n_clicks")],
    [State("input-street-name", "value"),
     State("input-source-type", "value"),
     State("input-ip-url", "value"),
     State("upload-video", "contents"),
     State("upload-video", "filename"),
     State("intersection-tabs", "children"),
     State("intersection-tabs", "active_tab")]
)
def handle_submit_stream(n_clicks, street_name, source_type, ip_url, upload_contents, filename, current_tabs, current_active):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
        
    if not street_name:
        street_name = f"Camera {len(intersections) + 1}"
        
    camera_id = f"cam{len(intersections) + 1}_{uuid.uuid4().hex[:6]}"
    
    source = None
    if source_type == "ip" and ip_url:
        source = ip_url
    elif source_type == "upload" and upload_contents:
        try:
            content_type, content_string = upload_contents.split(',')
            decoded = base64.b64decode(content_string)
            file_path = os.path.join("cache", f"{camera_id}_{filename}")
            with open(file_path, "wb") as f:
                f.write(decoded)
            source = file_path
        except Exception as e:
            print("Error uploading file:", e)
            raise dash.exceptions.PreventUpdate
    else:
        raise dash.exceptions.PreventUpdate
        
    new_state = IntersectionState(camera_id, street_name, source)
    intersections[camera_id] = new_state
    
    spawn_camera_threads(camera_id)
    
    new_tab = dbc.Tab(label=street_name, tab_id=camera_id)
    current_tabs.append(new_tab)
    
    return current_tabs, camera_id

def spawn_camera_threads(iid):
    t = threading.Thread(target=process_traffic, args=(iid,))
    t.daemon = True
    t.start()
    
    ai_t = threading.Thread(target=ai_worker, args=(iid,))
    ai_t.daemon = True
    ai_t.start()
    
# --- MAIN ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚦 TRAFFIC AI COMMAND CENTER v4.0 (Multi-Intersection)")
    print("="*60)
    print("🔄 Starting video processing threads...")
    
    for iid in intersections.keys():
        spawn_camera_threads(iid)
    
    print("🌐 Launching dashboard at http://0.0.0.0:8050")
    print("="*60 + "\n")
    
    # Run Dash app using Werkzeug to avoid Waitress stream buffering issues
    dash_thread = threading.Thread(target=lambda: dash_app.run(host='0.0.0.0', port=8050, debug=False, use_reloader=False))
    dash_thread.daemon = True
    dash_thread.start()

    # Main thread handles OpenCV windows to avoid GUI thread issues
    headless = os.environ.get('HEADLESS') == '1'
    try:
        while True:
            if not headless:
                for iid, state in intersections.items():
                    if state.latest_frame is not None:
                        cv2.imshow(f"Traffic AI v4.0 - {state.name}", state.latest_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            time.sleep(0.03) # ~30 FPS refresh rate for display
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        if not headless:
            cv2.destroyAllWindows()