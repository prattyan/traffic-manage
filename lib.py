import cv2
import time
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from ultralytics import YOLO
import threading

# Load pre-trained YOLO model for vehicle detection
yolo_model = YOLO("yolov8n.pt")  # Using YOLOv8 nano for real-time processing

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

# Initialize video capture for real-time traffic monitoring
cap = cv2.VideoCapture("traffic_video.mp4")  # Replace with live camera feed

def detect_emergency_vehicle(results):
    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls[0].item())]
            if label in ['ambulance', 'fire truck', 'police car']:
                return True
    return False

# Initialize Dash app for real-time monitoring
dash_app = dash.Dash(__name__)

dash_app.layout = html.Div([
    html.H1("Real-Time Traffic Monitoring Dashboard"),
    dcc.Graph(id='traffic-graph'),
    html.Div(id='decision-text', style={'fontSize': 20, 'marginTop': 20}),
    dcc.Interval(id='interval-component', interval=5000, n_intervals=0)
])

# Add a lock for thread-safe data access
traffic_data_lock = threading.Lock()
traffic_data = []

def process_traffic():
    global traffic_data
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Object detection using YOLO
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
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                elif label in ['ambulance', 'fire truck', 'police car'] and conf > 0.5:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "EMERGENCY", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Determine traffic light control decision
        decision = control_traffic(vehicle_count, emergency_detected)
        
        # Store data for visualization with thread safety
        with traffic_data_lock:
            traffic_data.append({
                "timestamp": time.time(),
                "vehicle_count": vehicle_count,
                "decision": decision
            })
            if len(traffic_data) > 30:  # Changed from 20 to 30
                traffic_data.pop(0)
        
        # Display processed video
        try:
            cv2.imshow("Traffic Monitoring", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            pass
            
        # Add a small delay to prevent high CPU usage
        time.sleep(0.03)
    
    cap.release()
    cv2.destroyAllWindows()

# Dash callback to update graph and decision text
@dash_app.callback(
    [Output('traffic-graph', 'figure'), Output('decision-text', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    with traffic_data_lock:
        if len(traffic_data) == 0:
            return {"data": [], "layout": {"title": "Traffic Volume Over Time"}}, "Waiting for data..."
        
        timestamps = [time.strftime('%H:%M:%S', time.localtime(d['timestamp'])) for d in traffic_data]
        vehicle_counts = [d['vehicle_count'] for d in traffic_data]
        decisions = [d['decision'] for d in traffic_data]
    
    fig = {
        "data": [
            {"x": timestamps, "y": vehicle_counts, "type": "line", "name": "Vehicle Count"}
        ],
        "layout": {
            "title": "Traffic Volume Over Time",
            "xaxis": {"title": "Time"},
            "yaxis": {"title": "Number of Vehicles"},
            "margin": {"l": 60, "r": 40, "t": 40, "b": 60}
        }
    }
    
    latest_decision = f"Latest Decision: {decisions[-1]}"
    return fig, latest_decision

if __name__ == "__main__":
    # Reduce the interval for more frequent updates
    dash_app.layout.children[3].interval = 1000  # Update every 1 second
    
    traffic_thread = threading.Thread(target=process_traffic)
    traffic_thread.daemon = True  # Make thread daemon so it exits when main program exits
    traffic_thread.start()
    
    dash_app.run_server(debug=True, port=8050)  # Enable debug mode for development
