import cv2
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from ultralytics import YOLO
import threading

# --- GLOBAL SHARED MEMORY ---
traffic_history = []  # Stores vehicle counts for the graph

# --- LOAD MODELS ---
# 1. YOLO Model
print("Loading YOLO...")
yolo_model = YOLO("yolov8n.pt")

# 2. LSTM Brain (Simulated Fallback)
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
    dcc.Interval(id='interval-component', interval=2000, n_intervals=0) # Updates every 2 seconds
])

# --- VIDEO THREAD (THE EYES) ---
def process_traffic():
    global traffic_history
    
    # *** IMPORTANT: UPDATE YOUR IP HERE IF IT CHANGES ***
    # Current IP based on your screenshot: 10.29.222.238
    video_url = "http://10.29.222.238:4747/video"
    cap = cv2.VideoCapture(video_url)
    
    print(f"Connecting to Camera: {video_url} ...")
    
    while True:
        # Safety check: Reconnect if stream drops
        if not cap.isOpened():
            print("Camera disconnected. Retrying...")
            cap.open(video_url)
            cv2.waitKey(1000)
            continue

        ret, frame = cap.read()
        if not ret:
            # If frame is empty, skip this loop (don't crash)
            continue

        # 1. Detect Objects
        results = yolo_model(frame)
        
        # 2. Count Vehicles
        # Counting: Car(2), Bike(3), Bus(5), Truck(7), Person(0), Scissors(76)
        target_classes = [2, 3, 5, 7, 0, 76]
        vehicle_count = 0
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                if class_id in target_classes:
                    vehicle_count += 1
                    # Draw Box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = result.names[class_id]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 3. Save to Global History (For the Graph)
        traffic_history.append(vehicle_count)
        if len(traffic_history) > 50:
            traffic_history.pop(0)

        # 4. Make Decision
        emergency = detect_emergency_vehicle(results)
        decision = control_traffic(vehicle_count, emergency)
        
        # 5. Show Video Window
        cv2.putText(frame, f"Decision: {decision}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Traffic Monitoring (Aditya)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

# --- DASHBOARD UPDATE (THE FACE) ---
@dash_app.callback(
    [Output('traffic-graph', 'figure'), Output('decision-text', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    global traffic_history
    
    # Handle empty data
    if not traffic_history:
        return {
            'data': [], 
            'layout': {'title': 'Waiting for Camera Data...'}
        }, "System Initializing..."
    
    # Draw Graph
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
    
    last_count = traffic_history[-1]
    text = f"Live Status: {last_count} Vehicles Detected"
    return figure, text

# --- STARTUP ---
if __name__ == "__main__":
    # Start the Camera Thread
    t = threading.Thread(target=process_traffic)
    t.daemon = True # Kills thread when main program exits
    t.start()
    
    print("✅ System Started. Go to http://127.0.0.1:8050")
    # Start the Web Server
    dash_app.run(debug=False, host='0.0.0.0', port=8050)