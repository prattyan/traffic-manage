import cv2
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from ultralytics import YOLO
import threading

# --- GLOBAL SHARED MEMORY ---
traffic_history = []  # Simple list for the graph

# --- LOAD MODELS ---
print("Loading YOLO...")
yolo_model = YOLO("yolov8n.pt")

# --- SIMULATED BRAIN (Prevents Crash if .h5 is missing) ---
class TrafficBrain:
    def predict(self, data):
        # Logic: If cars > 5, Traffic is High (1)
        return [[1]] if data[0][0] > 5 else [[0]]

try:
    from keras.models import load_model
    lstm_model = load_model('traffic_lstm.h5')
    print("✅ Loaded Real Brain (.h5 file)")
except:
    print("⚠️ Using Simulated Brain (File not found)")
    lstm_model = TrafficBrain()

# --- HELPER FUNCTIONS ---
def control_traffic(vehicle_count, emergency_detected):
    if emergency_detected:
        return "PRIORITY: Emergency Vehicle!"
    elif vehicle_count > 5:
        return "Extend Green Light"
    elif vehicle_count < 2:
        return "Shorten Green Light"
    else:
        return "Normal Cycle"

def detect_emergency_vehicle(results):
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            label = result.names[class_id]
            if label in ['ambulance', 'fire truck', 'police car']:
                return True
    return False

# --- DASHBOARD SETUP ---
dash_app = dash.Dash(__name__)

dash_app.layout = html.Div([
    html.H1("Aditya's Traffic Management System", style={'textAlign': 'center'}),
    dcc.Graph(id='traffic-graph'),
    html.H2(id='decision-text', style={'textAlign': 'center', 'color': 'red'}),
    dcc.Interval(id='interval-component', interval=2000, n_intervals=0)
])

# --- VIDEO THREAD (THE EYES) ---
def process_traffic():
    global traffic_history
    
    # *** 1. IP CAMERA SETUP ***
    # Replace this with your phone's IP if using DroidCam
    # source = "http://10.0.0.1:4747/video" 
    
    # *** 2. VIDEO FILE SETUP (Looping) ***
    source = 0 # Change to video file path for testing with a file 
    
    cap = cv2.VideoCapture(source)
    
    while True:
        # Loop video if it ends
        if not cap.isOpened():
            cap.open(source)
        
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Detect
        results = yolo_model(frame)
        
        # Count Vehicles (Car=2, Bike=3, Bus=5, Truck=7, Person=0)
        target_classes = [2, 3, 5, 7, 0]
        vehicle_count = 0
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                if class_id in target_classes:
                    vehicle_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = result.names[class_id]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update Graph Data
        traffic_history.append(vehicle_count)
        if len(traffic_history) > 50:
            traffic_history.pop(0)

        # Decision
        emergency = detect_emergency_vehicle(results)
        decision = control_traffic(vehicle_count, emergency)
        
        cv2.putText(frame, f"Decision: {decision}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Traffic Monitoring (Aditya)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

# --- DASHBOARD UPDATE ---
@dash_app.callback(
    [Output('traffic-graph', 'figure'), Output('decision-text', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    global traffic_history
    
    if not traffic_history:
        return {'data': [], 'layout': {'title': 'Waiting...'}}, "Initializing..."
    
    figure = {
        'data': [{
            'x': list(range(len(traffic_history))),
            'y': traffic_history,
            'type': 'line',
            'name': 'Vehicles',
            'line': {'color': 'blue', 'width': 3}
        }],
        'layout': {
            'title': 'Real-Time Traffic Density',
            'yaxis': {'title': 'Count', 'range': [0, 10]}
        }
    }
    
    text = f"Live Status: {traffic_history[-1]} Vehicles Detected"
    return figure, text

# --- STARTUP ---
if __name__ == "__main__":
    t = threading.Thread(target=process_traffic)
    t.daemon = True 
    t.start()
    
    # FIXED: run() instead of run_server() for compatibility
    dash_app.run(debug=False, host='0.0.0.0', port=8050)