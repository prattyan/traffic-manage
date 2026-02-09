# 🚦 Real-Time Traffic Monitoring & Intelligent Control System

An **AI-powered smart traffic management system** that leverages **YOLOv8** for real-time vehicle detection and **LSTM neural networks** for traffic flow prediction. The system dynamically controls traffic signals, prioritizes emergency vehicles, and provides a live visualization dashboard.

---

## 📸 Demo

![System Demo](https://i.postimg.cc/Zn2ZpLh8/Chat-GPT-Image-Apr-20-2025-02-37-11-PM.png)

🎥 **[Watch the Demo Video](https://drive.google.com/file/d/1aijR05oew3JxfjD6C62UK2TpRercrF2t/view?usp=sharing)**

---

## 🧠 Key Features

- 🔍 Real-time vehicle detection using **YOLOv8**
- 🚨 Emergency vehicle recognition and priority control
- 📈 Traffic congestion prediction using **LSTM neural networks**
- 🟢 Adaptive traffic signal control
  - Dynamic green light extension
  - Idle time reduction
  - Emergency override logic
- 📊 Live dashboard visualization
  - Vehicle counts
  - Traffic decisions
  - Real-time updates
- 🎥 Supports video files and live camera feeds

---

## 🧰 Technology Stack

| Layer               | Tools / Libraries |
|--------------------|------------------|
| Object Detection   | YOLOv8 (Ultralytics) |
| Video Processing  | OpenCV |
| Prediction Model  | TensorFlow / Keras (LSTM) |
| Dashboard UI      | Plotly Dash |
| Backend Logic     | Python (Multithreading) |

---

## 🚀 System Workflow

1. **Video Input**  
   Captures real-time video from a camera or video file using OpenCV

2. **Vehicle Detection**  
   YOLOv8 detects cars, buses, bikes, and emergency vehicles

3. **Traffic Prediction**  
   LSTM model predicts congestion levels based on vehicle density

4. **Decision Engine**  
   Adjusts traffic signal timing dynamically and prioritizes emergency vehicles

5. **Live Dashboard**  
   Displays vehicle count and traffic decisions with periodic updates

---

## 🖥️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/prattyan/traffic-manage
cd traffic-manage
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Prepare Input
- Add a sample video named `traffic_video.mp4`  
**OR**
- Connect a live camera feed

> YOLOv8 model weights are automatically downloaded via Ultralytics.

### 4️⃣ Run the Application
```bash
python main.py
```

> **Optional:** Run the Django API for persistent storage and analytics endpoints:
> ```bash
> cd traffic_manage && python manage.py migrate && python manage.py runserver
> ```
> Then set `TRAFFIC_API_URL=http://127.0.0.1:8000/api/traffic-snapshots/` to send snapshots from the Dash app to the API.

---

## 📁 Project Structure

```
📂 traffic-manage/
│
├── main.py                 # Main application (Dash + YOLO + LSTM)
├── analytics.py            # Analytics helpers (stats, CSV export)
├── traffic_video.mp4       # Sample traffic footage
├── traffic_lstm.h5         # Pre-trained LSTM model
├── yolov8n.pt              # YOLOv8 nano weights
├── requirements.txt        # Dependencies
├── data/                   # Traffic session CSV logs (created at runtime)
├── traffic_manage/         # Django API (optional)
│   ├── api/                # REST API + TrafficSnapshot model
│   └── manage.py
└── README.md
```

---

## 📊 Data & Analytics (for data scientists / analysts)

- **Session statistics** – The dashboard shows live session stats: mean/min/max vehicle count, standard deviation, congestion ratio, and elapsed time.
- **Historical CSV** – Every 5 seconds, a snapshot is appended to `data/traffic_session_YYYYMMDD.csv` (timestamp, vehicle_count, cars, trucks, bikes, pedestrians, congestion_pct, decision). Use these files for offline analysis, reporting, or model retraining.
- **Analytics module** – `analytics.py` provides:
  - `compute_summary_stats(values)` – mean, std, min, max, median, percentiles
  - `compute_congestion_stats(traffic_history)` – congestion and high-density ratios
  - `session_summary(...)` – combined session metrics
  - `export_session_csv(rows)` – export records to CSV
- **Django API (optional)** – When the API is running, the app can POST snapshots to `/api/traffic-snapshots/`. Use:
  - `GET /api/traffic-snapshots/` – list snapshots
  - `GET /api/traffic-snapshots/summary/?hours=24` – aggregated stats (count, mean, std, min, max, mean_congestion) for the last N hours

---

## 🔮 Future Enhancements

- 📡 Integration of multiple camera feeds
- 📢 Real-time alerts to traffic authorities
- ☁️ Cloud deployment for large-scale use
- 🧠 Reinforcement learning for smarter signal optimization

---

## 🧑‍💻 Author

**Prattyan Ghosh**  
📧 Email: prattyanghosh@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/prattyanghosh) | [Portfolio](https://prattyanghosh.xyz)

---

⭐ If you find this project useful, consider giving it a star!
