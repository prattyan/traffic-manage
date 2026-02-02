import cv2
import time
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from flask import Flask, Response
import numpy as np
from ultralytics import YOLO
from keras.models import load_model
import threading

server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.CYBORG])

yolo_model = YOLO("yolov8n.pt")
lstm_model = load_model("traffic_lstm.h5")

traffic_data = []
current_decision = "Normal Cycle"
current_vehicle_count = 0
lock = threading.Lock()

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
    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls[0].item())]
            if label in ['ambulance', 'fire truck', 'police car']:
                return True
    return False

def generate_frames():
    global current_decision, current_vehicle_count, traffic_data
    
    cap = cv2.VideoCapture("traffic_video.mp4")
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            results = yolo_model(frame)
            vehicle_count = 0
            emergency_detected = detect_emergency_vehicle(results)
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    label = result.names[int(box.cls[0].item())]
                    
                    if label in ['car', 'bus', 'truck', 'motorbike'] and conf > 0.5:
                        vehicle_count += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    elif label in ['ambulance', 'fire truck', 'police car'] and conf > 0.5:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, "EMERGENCY", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            traffic_state = np.array([[vehicle_count]])
            # predicted_traffic = lstm_model.predict(traffic_state) # Will use if displaying prediction
            
            decision = control_traffic(vehicle_count, emergency_detected)
            
            with lock:
                current_vehicle_count = vehicle_count
                current_decision = decision
                traffic_data.append({
                    "timestamp": time.time(),
                    "vehicle_count": vehicle_count,
                    "decision": decision
                })
                if len(traffic_data) > 30:
                    traffic_data.pop(0)

            cv2.putText(frame, f'Status: {decision}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()

@server.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("🚦 AI Traffic Control System", className="text-center text-primary mb-4"), width=12)
    ], className="mt-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Live Traffic Footage"),
                dbc.CardBody([
                    html.Img(src="/video_feed", style={"width": "100%", "height": "auto", "borderRadius": "5px"})
                ], className="p-2")
            ], className="mb-4 shadow-sm")
        ], width=12, lg=7),

        dbc.Col([
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Current Decision"),
                    dbc.CardBody(html.H4(id="live-decision", className="text-warning"))
                ], color="dark", inverse=True, className="mb-3"), width=12),
                
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Vehicle Count"),
                    dbc.CardBody(html.H2(id="live-count", className="text-info"))
                ], color="secondary", inverse=True, className="mb-3"), width=12),
            ]),
            
            dbc.Card([
                dbc.CardHeader("Traffic Density Over Time"),
                dbc.CardBody([
                    dcc.Graph(id='traffic-graph', style={"height": "300px"}, config={'displayModeBar': False})
                ])
            ], className="mb-4 shadow-sm")
        ], width=12, lg=5)
    ]),

    dbc.Row([
        dbc.Col(html.P("System Status: Online | Model: YOLOv8n + LSTM", className="text-center text-muted"), width=12)
    ]),

    dcc.Interval(id='interval-component', interval=2000, n_intervals=0)
], fluid=True)

@app.callback(
    [Output('traffic-graph', 'figure'),
     Output('live-decision', 'children'),
     Output('live-count', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    global traffic_data
    
    with lock:
        data_snapshot = list(traffic_data)
        decision_snapshot = current_decision
        count_snapshot = current_vehicle_count
    
    if not data_snapshot:
        return {"data": [], "layout": {}}, "Initializing...", "0"
    
    timestamps = [time.strftime('%H:%M:%S', time.localtime(d['timestamp'])) for d in data_snapshot]
    vehicle_counts = [d['vehicle_count'] for d in data_snapshot]
    
    fig = {
        "data": [
            {"x": timestamps, "y": vehicle_counts, "type": "area", "name": "Vehicles", "line": {"color": "#00bc8c"}}
        ],
        "layout": {
            "plot_bgcolor": "rgba(0,0,0,0)",
            "paper_bgcolor": "rgba(0,0,0,0)",
            "font": {"color": "#ffffff"},
            "margin": {"l": 40, "r": 20, "t": 20, "b": 40},
            "xaxis": {"showgrid": False},
            "yaxis": {"showgrid": True, "gridcolor": "#444"},
        }
    }
    
    return fig, decision_snapshot, str(count_snapshot)

if __name__ == "__main__":
    app.run_server(debug=True)