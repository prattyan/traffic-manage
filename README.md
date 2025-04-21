# 🚦 Real-Time Traffic Monitoring and Control System

An AI-powered system that uses YOLOv8 and LSTM to monitor vehicle traffic, detect emergency vehicles, and intelligently control traffic lights in real time. Includes a live dashboard for visualization.

---

## 📸 Demo

![Demo Screenshot](https://ibb.co/Y4vGhzxf)

🔗 **[Click here to view demo video](https://drive.google.com/file/d/1aijR05oew3JxfjD6C62UK2TpRercrF2t/view?usp=sharing)**

---

## 🧠 Features

- 🔍 Real-time vehicle detection using **YOLOv8**
- 🚨 Emergency vehicle priority handling
- 📈 Traffic prediction using **LSTM neural network**
- 🟢 Adaptive traffic light control (extend/shorten/priority logic)
- 📊 Live dashboard using **Dash** (vehicle count + decisions)
- 🎥 Works with video file or live camera feed

---

## 🧰 Tech Stack

| Component        | Tool / Library           |
|------------------|---------------------------|
| Object Detection | [YOLOv8 - Ultralytics](https://github.com/ultralytics/ultralytics) |
| Video Processing | OpenCV                    |
| Prediction Model | Keras / TensorFlow (LSTM) |
| Dashboard UI     | Plotly Dash               |
| Backend Logic    | Python (threading)        |

---

## 🚀 How It Works

1. **Video Input**: Captures real-time video using OpenCV
2. **Vehicle Detection**: YOLOv8 detects cars, buses, bikes, emergency vehicles
3. **Traffic Prediction**: LSTM model predicts traffic congestion based on vehicle count
4. **Decision Making**: System adjusts traffic light rules based on vehicle flow & emergency status
5. **Dashboard**: Dash updates every few seconds to show live vehicle count + decision history

---

## 🖥️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/traffic-monitoring-system.git
   cd traffic-monitoring-system
2. Install dependencies:
   pip install -r requirements.txt
3. Download YOLOv8 model weights:
# Automatically downloads when using Ultralytics
4. Add a sample video named traffic_video.mp4 in the root directory or connect to a camera feed.
5. Run the app:
   python app.py
   
📁 Folder Structure
📂 traffic-monitoring-system/
│
├── traffic_video.mp4          # Sample traffic footage
├── traffic_lstm.h5            # Pre-trained LSTM model
├── yolov8n.pt                 # YOLOv8-nano weights
├── app.py                     # Main application script
├── requirements.txt
└── README.md

🔮 Future Enhancements
Integrate multiple camera feeds

Store historical data in a database

Real-time alerts to city management systems

Cloud deployment for large-scale integration

🧑‍💻 Author
Prattyan Ghosh
📧 prattyanghosh@gmail.com
🔗[ LinkedIn | Portfolio] (https://www.linkedin.com/in/prattyan-ghosh-26217822a/)


