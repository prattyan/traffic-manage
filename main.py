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
from datetime import datetime
from collections import deque
import plotly.graph_objs as go
import plotly.express as px
import os

# --- GLOBAL SHARED MEMORY (Per Intersection) ---
class IntersectionState:
    def __init__(self, intersection_id, name, camera_source):
        self.id = intersection_id
        self.name = name
        self.camera_source = camera_source
        self.traffic_history = deque(maxlen=100)
        self.vehicle_types = {"cars": 0, "trucks": 0, "bikes": 0, "pedestrians": 0}
        self.emergency_log = deque(maxlen=20)
        self.current_decision = "Initializing..."
        self.system_status = {"fps": 0, "uptime": 0, "detections": 0}
        self.start_time = time.time()
        self.latest_frame = None

# Define multiple intersections here
intersections = {
    "cam1": IntersectionState("cam1", "Main St & 1st Ave", 0),
    # Add more cameras/videos here. E.g., "cam2": IntersectionState("cam2", "Broadway & 42nd St", "video2.mp4")
    "cam2": IntersectionState("cam2", "Broadway & 42nd St", "traffic_video.mp4") 
}

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

try:
    from keras.models import load_model
    lstm_model = load_model('traffic_lstm.h5')
    print("✅ Loaded Real LSTM Brain")
except:
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

def detect_emergency_vehicle(results):
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            label = result.names[class_id]
            if label.lower() in ['ambulance', 'fire truck', 'police car']:
                return True, label
    return False, None

def predict_traffic(history):
    if len(history) < 10:
        return list(history)[-5:] if len(history) >= 5 else [0] * 5
    try:
        input_data = np.array(list(history)[-10:]).reshape(1, 10, 1)
        predictions = []
        for _ in range(5):
            pred = lstm_model.predict(input_data, verbose=0)
            predictions.append(float(pred[0][0]) if hasattr(pred[0], '__iter__') else float(pred[0]))
        return predictions
    except:
        # Fallback prediction
        return [np.mean(list(history)[-5:]) + np.random.uniform(-1, 1) for _ in range(5)]

# --- CUSTOM CSS FOR GLASSMORPHISM ---
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600;700&display=swap');

:root {
    --neon-cyan: #00f5ff;
    --neon-purple: #8b5cf6;
    --neon-green: #00ff88;
    --neon-orange: #ff6b35;
    --neon-red: #ff3366;
    --glass-bg: rgba(15, 23, 42, 0.8);
    --glass-border: rgba(255, 255, 255, 0.1);
}

body {
    background: linear-gradient(135deg, #0a0a1a 0%, #1a0a2e 50%, #0d1a2d 100%) !important;
    font-family: 'Inter', sans-serif !important;
    min-height: 100vh;
}

.glass-card {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.7) 100%) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 20px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    transition: all 0.3s ease !important;
}

.glass-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
    border-color: rgba(0, 245, 255, 0.3) !important;
}

.neon-text {
    font-family: 'Orbitron', sans-serif !important;
    background: linear-gradient(90deg, #00f5ff, #8b5cf6, #00ff88);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 0 0 30px rgba(0, 245, 255, 0.5);
}

.status-indicator {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
    animation: pulse 2s infinite;
}

.status-online {
    background: #00ff88;
    box-shadow: 0 0 20px #00ff88;
}

.status-warning {
    background: #ff6b35;
    box-shadow: 0 0 20px #ff6b35;
}

.status-danger {
    background: #ff3366;
    box-shadow: 0 0 20px #ff3366;
    animation: pulse-danger 0.5s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.7; transform: scale(1.1); }
}

@keyframes pulse-danger {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.metric-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.5rem;
    font-weight: 900;
    color: #00f5ff;
    text-shadow: 0 0 20px rgba(0, 245, 255, 0.6);
}

.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: rgba(255, 255, 255, 0.6);
}

.decision-badge {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.2rem;
    padding: 15px 30px;
    border-radius: 50px;
    animation: glow 2s infinite;
}

@keyframes glow {
    0%, 100% { box-shadow: 0 0 20px currentColor; }
    50% { box-shadow: 0 0 40px currentColor; }
}

.progress-bar-custom {
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(90deg, #00ff88, #00f5ff, #8b5cf6);
    transition: width 0.5s ease;
}

.navbar-glass {
    background: linear-gradient(90deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.9)) !important;
    backdrop-filter: blur(20px) !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
}

.cmd-log {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #00ff88;
    background: rgba(0, 0, 0, 0.5);
    padding: 10px;
    border-radius: 8px;
    max-height: 150px;
    overflow-y: auto;
}

.nav-tabs .nav-link {
    color: rgba(255, 255, 255, 0.6);
    border: none;
    border-bottom: 2px solid transparent;
    font-family: 'Orbitron', sans-serif;
    letter-spacing: 1px;
}

.nav-tabs .nav-link.active {
    color: #00f5ff;
    background-color: transparent;
    border-color: transparent;
    border-bottom: 2px solid #00f5ff;
    text-shadow: 0 0 10px rgba(0, 245, 255, 0.5);
}
"""

# --- MODERN DASHBOARD SETUP ---
dash_app = dash.Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600;700&family=JetBrains+Mono&display=swap"
    ],
    suppress_callback_exceptions=True
)

dash_app.index_string = f'''
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>🚦 Traffic AI Command Center v4.0</title>
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

def create_metric_card(title, value_id, icon, color="#00f5ff"):
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Span(icon, style={'fontSize': '2rem'}),
                html.Span(className="status-indicator status-online", style={'marginLeft': '10px'})
            ], className="d-flex align-items-center mb-2"),
            html.H2(id=value_id, className="metric-value mb-1", style={'color': color}),
            html.P(title, className="metric-label mb-0")
        ])
    ], className="glass-card h-100")

def create_status_card(title, status, description_id, default_desc, icon):
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Span(icon, style={'fontSize': '1.5rem', 'marginRight': '10px'}),
                html.Span(title, style={'fontWeight': '600', 'fontSize': '0.9rem'})
            ], className="d-flex align-items-center mb-3"),
            html.Div([
                html.Span(className="status-indicator status-online"),
                html.Span(status, style={'color': '#00ff88', 'fontWeight': '700'})
            ], className="mb-2"),
            html.P(default_desc, id=description_id, className="text-muted small mb-0")
        ])
    ], className="glass-card h-100")

dash_app.layout = dbc.Container([
    # Background Animation Layer
    html.Div(className="background-animation"),
    
    # Header
    dbc.Navbar([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("🚦", style={'fontSize': '2rem', 'marginRight': '15px'}),
                        html.Span("TRAFFIC AI", className="neon-text", style={'fontSize': '1.8rem', 'fontWeight': '900'}),
                        html.Span(" COMMAND CENTER", style={'fontSize': '1.2rem', 'color': 'rgba(255,255,255,0.7)', 'fontWeight': '300'})
                    ], className="d-flex align-items-center")
                ], width="auto"),
                dbc.Col([
                    html.Div([
                        html.Span("v4.0", className="badge bg-primary me-2"),
                        html.Span(id="current-time", className="text-muted")
                    ])
                ], width="auto", className="ms-auto")
            ], align="center", className="w-100")
        ], fluid=True)
    ], className="navbar-glass mb-4 py-3", dark=True),

    # Intersection Tabs
    dbc.Tabs(
        id="intersection-tabs",
        active_tab=list(intersections.keys())[0],
        children=[
            dbc.Tab(label=state.name, tab_id=iid) for iid, state in intersections.items()
        ],
        className="mb-4"
    ),

    # Row 1: Key Metrics
    dbc.Row([
        dbc.Col(create_metric_card("VEHICLES DETECTED", "vehicle-count", "🚗", "#00f5ff"), lg=3, md=6, className="mb-4"),
        dbc.Col(create_metric_card("CONGESTION INDEX", "congestion-level", "📊", "#ff6b35"), lg=3, md=6, className="mb-4"),
        dbc.Col(create_metric_card("SYSTEM FPS", "fps-value", "⚡", "#00ff88"), lg=3, md=6, className="mb-4"),
        dbc.Col(create_metric_card("UPTIME (MIN)", "uptime-value", "⏱️", "#8b5cf6"), lg=3, md=6, className="mb-4"),
    ]),

    # Row 2: Status Cards
    dbc.Row([
        dbc.Col(create_status_card("Camera Feed", "ACTIVE", "camera-status-desc", "ID: CAM-01 | 1080p @ 30fps", "📍"), lg=3, md=6, className="mb-4"),
        dbc.Col(create_status_card("AI Engine", "ONLINE", "ai-status-desc", "YOLOv8n + LSTM Hybrid", "🧠"), lg=3, md=6, className="mb-4"),
        dbc.Col(create_status_card("Data Pipeline", "STREAMING", "pipeline-status-desc", "Real-time Processing", "🔄"), lg=3, md=6, className="mb-4"),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("⚡", style={'fontSize': '1.5rem', 'marginRight': '10px'}),
                        html.Span("LIVE DECISION", style={'fontWeight': '600', 'fontSize': '0.9rem'})
                    ], className="d-flex align-items-center mb-3"),
                    html.Div(id="decision-badge", className="text-center")
                ])
            ], className="glass-card h-100")
        ], lg=3, md=6, className="mb-4"),
    ]),

    # Row 3: Main Charts
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.Span("📈", style={'marginRight': '10px'}),
                        html.Span("REAL-TIME TRAFFIC DENSITY", style={'fontWeight': '600'})
                    ], className="d-flex align-items-center")
                ], style={'background': 'transparent', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                dbc.CardBody([
                    dcc.Graph(id='traffic-graph', style={'height': '400px'}, config={'displayModeBar': False})
                ])
            ], className="glass-card")
        ], lg=8, className="mb-4"),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.Span("🚙", style={'marginRight': '10px'}),
                        html.Span("VEHICLE CLASSIFICATION", style={'fontWeight': '600'})
                    ], className="d-flex align-items-center")
                ], style={'background': 'transparent', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                dbc.CardBody([
                    dcc.Graph(id='vehicle-pie', style={'height': '400px'}, config={'displayModeBar': False})
                ])
            ], className="glass-card")
        ], lg=4, className="mb-4"),
    ]),

    # Row 4: Prediction & Logs
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.Span("🔮", style={'marginRight': '10px'}),
                        html.Span("AI TRAFFIC PREDICTION (NEXT 5 MIN)", style={'fontWeight': '600'})
                    ], className="d-flex align-items-center")
                ], style={'background': 'transparent', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                dbc.CardBody([
                    dcc.Graph(id='prediction-graph', style={'height': '250px'}, config={'displayModeBar': False})
                ])
            ], className="glass-card")
        ], lg=6, className="mb-4"),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.Span("🚨", style={'marginRight': '10px'}),
                        html.Span("SYSTEM LOGS", style={'fontWeight': '600'})
                    ], className="d-flex align-items-center")
                ], style={'background': 'transparent', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                dbc.CardBody([
                    html.Div(id="system-logs", className="cmd-log")
                ])
            ], className="glass-card")
        ], lg=6, className="mb-4"),
    ]),

    # Row 5: Congestion Progress
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.P("NETWORK CONGESTION STATUS", className="metric-label mb-2"),
                            dbc.Progress(id="congestion-bar", value=0, className="mb-2", style={'height': '12px', 'background': 'rgba(255,255,255,0.1)'}),
                            html.Div([
                                html.Span("0%", className="text-success", style={'fontSize': '0.75rem'}),
                                html.Span("50%", className="text-warning mx-auto", style={'fontSize': '0.75rem'}),
                                html.Span("100%", className="text-danger", style={'fontSize': '0.75rem'})
                            ], className="d-flex justify-content-between")
                        ], lg=8),
                        dbc.Col([
                            html.Div(id="congestion-status", className="text-center mt-2")
                        ], lg=4)
                    ])
                ])
            ], className="glass-card")
        ], className="mb-4"),
    ]),

    # Footer
    dbc.Row([
        dbc.Col([
            html.Div([
                html.P([
                    "Developed with 💜 by ",
                    html.Span("Aditya Patra", className="neon-text"),
                    " | Traffic AI Command Center v4.0"
                ], className="text-center text-muted mb-0")
            ], className="py-3")
        ])
    ]),

    # Internal Clock
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0),
    dcc.Interval(id='fast-interval', interval=500, n_intervals=0)

], fluid=True, style={'padding': '20px'})

# --- VIDEO THREAD (MULTI-CAMERA SUPPORT) ---
def process_traffic(intersection_id):
    state = intersections[intersection_id]
    print(f"📸 STARTING CAMERA {state.camera_source} for {state.name}...")
    
    # FIX FOR WINDOWS FREEZE: Use cv2.CAP_DSHOW for webcams
    if isinstance(state.camera_source, int) and os.name == 'nt':
        cap = cv2.VideoCapture(state.camera_source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(state.camera_source)
    
    # Optional: Force lower resolution for speed and stability
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    fps_start = time.time()
    
    while True:
        try:
            ret, frame = cap.read()
            
            # --- CRASH GUARD: If camera hangs, restart it ---
            if not ret:
                if isinstance(state.camera_source, str):
                    # Loop video file
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print(f"⚠️ Camera hang on {intersection_id}. Re-initializing...")
                    cap.release()
                    time.sleep(0.5)
                    if os.name == 'nt':
                        cap = cv2.VideoCapture(state.camera_source, cv2.CAP_DSHOW)
                    else:
                        cap = cv2.VideoCapture(state.camera_source)
                    continue
            # ------------------------------------------------

            frame_count += 1
            if time.time() - fps_start >= 1:
                state.system_status["fps"] = frame_count
                frame_count = 0
                fps_start = time.time()

            results = list(yolo_model(frame, verbose=False, stream=True))
            
            # --- COUNTING LOGIC ---
            cars, trucks, bikes, pedestrians = 0, 0, 0, 0
            vehicle_count = 0
            
            class_mapping = {
                2: 'cars',      # car
                3: 'bikes',     # motorcycle
                5: 'trucks',    # bus
                7: 'trucks',    # truck
                0: 'pedestrians', # person
                1: 'bikes'      # bicycle
            }
            
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
                        
                        # Draw detection
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        label = result.names[class_id]
                        
                        color = (0, 255, 136) if category != 'pedestrians' else (255, 107, 53)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Update Shared Memory
            state.vehicle_types = {"cars": cars, "trucks": trucks, "bikes": bikes, "pedestrians": pedestrians}
            state.traffic_history.append(vehicle_count)
            state.system_status["detections"] += vehicle_count
            state.system_status["uptime"] = int((time.time() - state.start_time) / 60)

            # Emergency detection
            emergency, vehicle_type = detect_emergency_vehicle(results)
            if emergency:
                state.emergency_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 EMERGENCY: {vehicle_type} detected!")

            congestion = calculate_congestion(state.traffic_history)
            decision, _ = control_traffic(vehicle_count, emergency, congestion)
            state.current_decision = decision

            # Display
            cv2.putText(frame, f"FPS: {state.system_status.get('fps', 0)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Store frame for main thread to display
            state.latest_frame = frame
            
            # Free memory explicitly
            del results
        
        except Exception as e:
            print(f"❌ Error in loop for {intersection_id}: {e}")
            time.sleep(0.1) # Brief pause to prevent CPU spike
            
    cap.release()

# --- CALLBACKS ---
@dash_app.callback(
    [
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
    if not active_tab or active_tab not in intersections:
        active_tab = list(intersections.keys())[0]
        
    state = intersections[active_tab]
    history_list = list(state.traffic_history)
    
    # Traffic Graph
    traffic_fig = go.Figure()
    traffic_fig.add_trace(go.Scatter(
        x=list(range(len(history_list))),
        y=history_list,
        mode='lines',
        fill='tozeroy',
        line=dict(color='#00f5ff', width=3),
        fillcolor='rgba(0, 245, 255, 0.2)'
    ))
    traffic_fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(showgrid=False, title='Time (s)', color='#888'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Count', color='#888', range=[0, max(15, max(history_list) + 2) if history_list else 15]),
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False
    )
    
    # Pie Chart
    pie_fig = go.Figure()
    pie_fig.add_trace(go.Pie(
        labels=['Cars', 'Trucks', 'Bikes', 'Pedestrians'],
        values=[state.vehicle_types['cars'], state.vehicle_types['trucks'], state.vehicle_types['bikes'], state.vehicle_types['pedestrians']],
        hole=0.6,
        marker=dict(colors=['#00f5ff', '#ff6b35', '#00ff88', '#8b5cf6']),
        textinfo='percent+label',
        textfont=dict(size=10)
    ))
    pie_fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    # Prediction Graph
    predictions = predict_traffic(history_list) if len(history_list) > 5 else [0]*5
    pred_fig = go.Figure()
    pred_fig.add_trace(go.Bar(
        x=['1 min', '2 min', '3 min', '4 min', '5 min'],
        y=predictions,
        marker=dict(color=['#8b5cf6', '#7c3aed', '#6d28d9', '#5b21b6', '#4c1d95']),
        text=[f'{p:.1f}' for p in predictions],
        textposition='outside',
        textfont=dict(color='white')
    ))
    pred_fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(showgrid=False, color='#888'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='#888', range=[0, max(15, max(predictions) + 2) if predictions else 15]),
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False
    )
    
    # Metrics
    vehicle_count = history_list[-1] if history_list else 0
    congestion = calculate_congestion(history_list)
    fps = state.system_status.get('fps', 0)
    uptime = state.system_status.get('uptime', 0)
    
    # Decision Badge
    decision_text, decision_type = control_traffic(vehicle_count, False, congestion)
    color_map = {'danger': '#ff3366', 'warning': '#ff6b35', 'success': '#00ff88', 'info': '#00f5ff'}
    badge_color = color_map.get(decision_type, '#00f5ff')
    
    decision_badge = html.Span(
        decision_text,
        className="decision-badge",
        style={
            'background': f'linear-gradient(135deg, {badge_color}33, {badge_color}11)',
            'border': f'2px solid {badge_color}',
            'color': badge_color,
            'display': 'inline-block'
        }
    )
    
    # Congestion bar
    bar_color = "success" if congestion < 30 else "warning" if congestion < 70 else "danger"
    congestion_status = html.Span(
        f"{'LOW' if congestion < 30 else 'MODERATE' if congestion < 70 else 'HIGH'} LOAD",
        style={'color': '#00ff88' if congestion < 30 else '#ff6b35' if congestion < 70 else '#ff3366', 'fontWeight': '700'}
    )
    
    # Time
    current_time = datetime.now().strftime("%H:%M:%S | %d %b %Y")
    
    # System logs
    default_logs = [
        f"[{datetime.now().strftime('%H:%M:%S')}] ✅ System operational",
        f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 Processing frames at {fps} FPS",
        f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Total detections: {state.system_status.get('detections', 0)}"
    ]
    all_logs = state.emergency_log[-10:] + default_logs if state.emergency_log else default_logs
    logs_display = html.Div([html.P(log, className="mb-1") for log in all_logs[-8:]])
    
    # Camera Status Description
    camera_status_desc = f"ID: {state.id.upper()} | Source: {state.camera_source}"
    
    return (
        traffic_fig, pie_fig, pred_fig,
        str(vehicle_count), f"{congestion}%", str(fps), str(uptime),
        decision_badge, congestion, bar_color, congestion_status,
        current_time, logs_display, camera_status_desc
    )

# --- MAIN ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚦 TRAFFIC AI COMMAND CENTER v4.0 (Multi-Intersection)")
    print("="*60)
    print("🔄 Starting video processing threads...")
    
    # Start a thread for each intersection
    threads = []
    for iid in intersections.keys():
        t = threading.Thread(target=process_traffic, args=(iid,))
        t.daemon = True
        t.start()
        threads.append(t)
    
    print("🌐 Launching dashboard at http://0.0.0.0:8050")
    print("="*60 + "\n")
    
    # Run Dash app in a separate thread
    dash_thread = threading.Thread(target=lambda: dash_app.run(debug=False, host='0.0.0.0', port=8050, use_reloader=False))
    dash_thread.daemon = True
    dash_thread.start()

    # Main thread handles OpenCV windows to avoid GUI thread issues
    try:
        while True:
            for iid, state in intersections.items():
                if state.latest_frame is not None:
                    cv2.imshow(f"Traffic AI v4.0 - {state.name}", state.latest_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.03) # ~30 FPS refresh rate for display
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        cv2.destroyAllWindows()