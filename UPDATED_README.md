🚦 Real-Time Traffic Monitoring & Intelligent Control System

An AI-powered smart traffic management system that leverages YOLOv8 for real-time vehicle detection and LSTM neural networks for traffic flow prediction. The system integrates a standardized evaluation pipeline to ensure reproducible benchmarking on real-world public traffic datasets.

🌍 Project Overview

Modern urban traffic systems require intelligent monitoring and predictive control mechanisms. This project combines:

🚗 Real-time vehicle detection

📊 Traffic density estimation

📈 Time-series traffic forecasting

⚡ Performance benchmarking & reproducible evaluation

The system is designed to validate detection and prediction models using standardized metrics and real public datasets.

📂 Dataset Used
📌 BDD100K (Berkeley DeepDrive 100K)
📖 Description

BDD100K is a large-scale autonomous driving dataset containing:

100,000 real-world road images

Diverse weather conditions (rain, fog, night, daylight)

Multiple traffic object classes

2D bounding box annotations

For this project, we use the Object Detection subset focusing only on vehicle-related classes.

🔗 Official Dataset Download

Download from the official website:

👉 https://bdd-data.berkeley.edu/

Required files:

bdd100k_images_100k.zip

bdd100k_labels_release.zip

⚠ Important: Images and labels must be downloaded from the SAME official source to avoid dataset mismatch issues.

🏗️ Project Architecture
traffic-manage/
│
├── bdd100k/                 # Raw images
│   ├── train/
│   ├── val/
│   ├── test/
│
├── 100k/                    # JSON annotations
│   ├── train/
│   ├── val/
│   ├── test/
│
├── dataset/                 # YOLO formatted dataset
│   ├── images/
│   ├── labels/
│
├── convert_bdd_to_yolo.py
├── evaluation_pipeline.py
├── lstm_model.py
└── data.yaml
🤖 Model Components
🚗 1. Vehicle Detection — YOLOv8

Framework: Ultralytics YOLOv8

Used for:

Detecting vehicles (car, bus, truck, motorcycle)

Counting vehicles per frame

Generating confusion matrix

Measuring inference speed (FPS)

Bounding boxes are converted from BDD100K JSON to YOLO format:

𝑥
𝑐
𝑒
𝑛
𝑡
𝑒
𝑟
=
𝑥
1
+
𝑥
2
2
𝑊
x
center
	​

=
2W
x1+x2
	​

𝑦
𝑐
𝑒
𝑛
𝑡
𝑒
𝑟
=
𝑦
1
+
𝑦
2
2
𝐻
y
center
	​

=
2H
y1+y2
	​

𝑤
𝑖
𝑑
𝑡
ℎ
=
𝑥
2
−
𝑥
1
𝑊
width=
W
x2−x1
	​

ℎ
𝑒
𝑖
𝑔
ℎ
𝑡
=
𝑦
2
−
𝑦
1
𝐻
height=
H
y2−y1
	​

📈 2. Traffic Flow Prediction — LSTM

Input:

Vehicle counts over time

Output:

Predicted future traffic density

Used for:

Traffic congestion forecasting

Correlation analysis (predicted vs actual)

📊 Standardized Evaluation Pipeline

Implemented in:

evaluation_pipeline.py

The evaluation module computes:

🎯 Detection Metrics

Accuracy

Precision

Recall

Specificity

F1-score

	​

📉 Confusion Matrix

Generated using sklearn.metrics.confusion_matrix

Visualized using seaborn heatmap

Saved as PNG file

📈 Correlation Matrix (LSTM)

Pearson correlation between predicted and actual traffic flow

Heatmap visualization

Evaluates prediction reliability
​


Used to validate real-time capability.

🧪 Reproducible Setup
1️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate
2️⃣ Install Dependencies
pip install ultralytics opencv-python matplotlib seaborn scikit-learn numpy pandas
3️⃣ Convert BDD100K to YOLO Format
python convert_bdd_to_yolo.py
4️⃣ Train YOLOv8
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640
5️⃣ Run Evaluation
python evaluation_pipeline.py

This generates:

Detection metrics

Confusion matrix plot

Correlation matrix plot

FPS output

✅ Acceptance Criteria Coverage
Requirement	Status
Public dataset used	✅
Dataset download documented	✅
Accuracy, Precision, Recall	✅
Specificity	✅
F1-score	✅
Confusion matrix generated	✅
Correlation matrix generated	✅
FPS measurement	✅
Reproducible steps provided	✅
🚀 Key Features

Real-world dataset benchmarking

Automated dataset conversion

Standardized evaluation pipeline

Real-time inference validation

Modular architecture for scalability

🔮 Future Enhancements

Add mAP (mean Average Precision) evaluation

Integrate additional datasets (METR-LA / PEMS-BAY)

Deploy web dashboard for live monitoring

Adaptive traffic signal control logic

Output Achieved:
================================================================================
CLASSIFICATION METRICS (Binary: Vehicle Detected vs Not)
================================================================================

📊 Classification Metrics:
   Accuracy:    0.8480
   Precision:   0.8416
   Recall:      0.8551
   Specificity: 0.8410
   F1-Score:    0.8483

✅ Confusion matrix saved: evaluation_results/confusion_matrix_20260223_204828.png

================================================================================
INFERENCE SPEED BENCHMARK
================================================================================

🎬 Benchmarking YOLOv8m...
   Images: 1000
   Image size: (640, 640)

⚡ Performance Metrics:
   FPS:                30.30
   Time per image:     33.00 ms
   Total time:         33.00 seconds

================================================================================
PREDICTION EVALUATION (LSTM Traffic Flow)
================================================================================

📊 Prediction Metrics:
   MSE (Mean Squared Error):  23.6475
   RMSE (Root MSE):           4.8629
   MAE (Mean Absolute Error): 3.9203
   Correlation:               0.8353

✅ Correlation matrix saved: evaluation_results/correlation_matrix_20260223_204828.png
✅ Prediction comparison saved: evaluation_results/prediction_comparison_20260223_204828.png

================================================================================
EVALUATION COMPLETE
================================================================================

📁 Results saved to: evaluation_results/

✅ All metrics computed:
   ✓ Detection: Accuracy, Precision, Recall, Specificity, F1-Score
   ✓ Confusion Matrix plot
   ✓ Prediction: MSE, RMSE, MAE, Correlation
   ✓ Correlation Matrix plot
   ✓ FPS Benchmark

================================================================================

📜 License

Dataset license governed by official BDD100K terms.
Project intended for academic and research benchmarking purposes.