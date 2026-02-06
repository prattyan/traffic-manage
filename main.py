"""
🚦 TRAFFIC AI COMMAND CENTER v3.1 (Hotfix)
Advanced Real-Time Traffic Management System
Developed by: Aditya Patra
Features: YOLOv8 Detection, LSTM Prediction, Glassmorphic UI
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

# --- GLOBAL SHARED MEMORY ---
traffic_history = deque(maxlen=100)
vehicle_types = {"cars": 0, "trucks": 0, "bikes": 0, "pedestrians": 0}
emergency_log = []
prediction_data = []
current_decision = "Initializing..."
system_status = {"fps": 0, "uptime": 0, "detections": 0}
start_time = time.time()

# --- LOAD MODELS ---
print("🔄 Loading YOLO Model...")
yolo_model = YOLO("yolov8n.pt")
print("✅ YOLO Loaded Successfully")

# Simulated Brain (Fallback)
class TrafficBrain:
    def predict(self, data):
        if len(data) > 5:
            trend = np.mean(data[-5:]) - np.mean(data[-10:-5]) if len(data) > 10 else 0
            return [[1]] if trend > 0 else [[0]]
        return [[0]]

try:
    from keras.models import load_model
    if os.path.exists('traffic_lstm.h5'):
        lstm_model = load_model('traffic_lstm.h5')
        print("✅ Loaded Real LSTM Brain")
    else:
        raise FileNotFoundError("LSTM file not found")
except Exception as e:
    print(f"⚠️ {e} - Using Simulated Brain (Fallback)")
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
        return [np.mean(list(history)[-5:]) + np.random.uniform(-1, 1) for _ in range(5)]

# --- CUSTOM CSS ---
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600;700&display=swap');
:root { --neon-cyan: #00f5ff; --glass-bg: rgba(15, 23, 42, 0.8); }
body { background: linear-gradient(135deg, #0a0a1a 0%, #1a0a2e 50%, #0d1a2d 100%); font-family: 'Inter', sans-serif; }
.glass-card { background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.7)); backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.1); border-radius: 20px; }
.neon-text { font-family: 'Orbitron', sans-serif; background: linear-gradient(90deg, #00f5ff, #8b5cf6, #00ff88); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.metric-value { font-family: 'Orbitron', sans-serif; font-size: 2.5rem; color: #00f5ff; }
.decision-badge { font-family: 'Orbitron', sans-serif; padding: 15px 30px; border-radius: 50px; }
.navbar-glass { background: rgba(15, 23, 42, 0.95); backdrop-filter: blur(20px); border-bottom: 1px solid rgba(255,255,255,0.1); }
.cmd-log { font-family: 'JetBrains Mono', monospace; color: #00ff88; background: rgba(0,0,0,0.5); padding: 10px; border-radius: 8px; max-height: 150px; overflow-y: auto; }
"""

# --- DASHBOARD SETUP ---
dash_app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
dash_app.index_string = f'''<!DOCTYPE html><html><head>{{%metas%}}<title>Traffic AI v3.1</title>{{%favicon%}}{{%css%}}<style>{CUSTOM_CSS}</style></head><body>{{%app_entry%}}<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer></body></html>'''

def create_metric_card(title, value_id, icon, color="#00f5ff"):
    return dbc.Card([dbc.CardBody([html.H2(id=value_id, className="metric-value mb-1", style={'color': color}), html.P(title, className="metric-label mb-0")])], className="glass-card h-100")

dash_app.layout = dbc.Container([
    dbc.Navbar([dbc.Row([dbc.Col([html.Span("🚦 TRAFFIC AI", className="neon-text", style={'fontSize': '1.8rem'})])])], className="navbar-glass mb-4 py-3"),
    dbc.Row([
        dbc.Col(create_metric_card("VEHICLES", "vehicle-count", "🚗"), width=3),
        dbc.Col(create_metric_card("CONGESTION", "congestion-level", "📊", "#ff6b35"), width=3),
        dbc.Col(create_metric_card("FPS", "fps-value", "⚡", "#00ff88"), width=3),
        dbc.Col(create_metric_card("UPTIME", "uptime-value", "⏱️", "#8b5cf6"), width=3),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([dbc.Card([dbc.CardBody([dcc.Graph(id='traffic-graph', style={'height': '400px'})])], className="glass-card")], width=8),
        dbc.Col([dbc.Card([dbc.CardBody([html.Div(id="decision-badge", className="text-center")])], className="glass-card h-100")], width=4),
    ], className="mb-4"),
    dcc.Interval(id='interval-component', interval=1000)
], fluid=True, style={'padding': '20px'})

# --- VIDEO THREAD (CLEAN WEBCAM ONLY) ---
def process_traffic():
    global traffic_history, vehicle_types, current_decision, system_status, emergency_log
    
    print("📸 STARTING WEBCAM (Source 0)...")
    
    # 1. Platform-Specific Initialization
    if os.name == 'nt': # Windows
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else: # macOS / Linux
        cap = cv2.VideoCapture(0)
    
    # 2. Resolution Safety
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    time.sleep(2.0) # Warmup time for camera
    
    frame_count = 0
    fps_start = time.time()
    
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Camera signal lost. Re-initializing...")
                cap.release()
                time.sleep(1)
                if os.name == 'nt':
                    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(0)
                continue

            frame_count += 1
            if time.time() - fps_start >= 1:
                system_status["fps"] = frame_count
                frame_count = 0
                fps_start = time.time()

            results = yolo_model(frame, verbose=False)
            
            # Count Logic
            cars, trucks, bikes, pedestrians = 0, 0, 0, 0
            vehicle_count = 0
            class_mapping = {2: 'cars', 3: 'bikes', 5: 'trucks', 7: 'trucks', 0: 'pedestrians', 1: 'bikes'}
            
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0].item())
                    if cls in class_mapping:
                        vehicle_count += 1
                        # Visualization
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 136), 2)

            traffic_history.append(vehicle_count)
            system_status["detections"] = vehicle_count
            system_status["uptime"] = int((time.time() - start_time) / 60)
            congestion = calculate_congestion(traffic_history)
            current_decision, _ = control_traffic(vehicle_count, False, congestion)

            cv2.imshow("Traffic AI v3.1", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(0.1)
            
    cap.release()
    cv2.destroyAllWindows()

@dash_app.callback(
    [Output('traffic-graph', 'figure'), Output('vehicle-count', 'children'), Output('congestion-level', 'children'),
     Output('fps-value', 'children'), Output('uptime-value', 'children'), Output('decision-badge', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    fig = go.Figure(go.Scatter(y=list(traffic_history), line=dict(color='#00f5ff', width=3), fill='tozeroy'))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), margin=dict(l=20, r=20, t=20, b=20))
    
    count = list(traffic_history)[-1] if traffic_history else 0
    cong = calculate_congestion(traffic_history)
    badge = html.Span(current_decision, className="decision-badge", style={'border': '2px solid #00f5ff', 'color': '#00f5ff'})
    
    return fig, str(count), f"{cong}%", str(system_status['fps']), str(system_status['uptime']), badge

if __name__ == "__main__":
    t = threading.Thread(target=process_traffic)
    t.daemon = True
    t.start()
    dash_app.run(debug=False, host='0.0.0.0', port=8050)