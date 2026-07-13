# 🛣️ RoadSense AI: Road Damage Assessment & GPS Logging Network

RoadSense AI is an end-to-end Computer Vision and telemetry application designed to detect, log, and map road damage in real-time. Powered by a custom-trained YOLO architecture, the system accurately identifies road hazards such as **potholes** and **surface cracks**, dynamically evaluates an overall **Road Health Score**, logs the localized **GPS coordinates**, and plots incident markers instantly onto an **interactive map**.

---

## 🚀 Core Features
* **Dual Execution Modes:** * `📸 Static Image Upload:` Drop road snapshots to audit individual sections.
  * `🎥 Live Camera Scan:` Real-time, continuous video analysis using integrated device cameras or dashcams via WebRTC streaming.
* **Real-Time Object Detection:** Fine-tuned computer vision layer tracking road structural failure anomalies (Potholes, Longitudinal, Transverse, and Alligator Cracks).
* **Automated Telemetry Matrix:** Logs timestamps, failure types, hardware confidence scores, and coordinate matrices synchronously.
* **Interactive Spatio-Temporal Map:** Live-rendered geographical layout utilizing Folium engine to drop visual incident hazard tags.
* **Automated Quality Scorecard:** Real-time road health metric calculations dropping down from a baseline score of 100 based on hazard frequency and severity.

---

## 🛠️ Tech Stack & Architecture
* **Frontend Dashboard Interface:** Streamlit Engine
* **Core Computer Vision Pipeline:** Ultralytics YOLOv12
* **Geospatial Mapping Utilities:** Folium & Streamlit-Folium Wrappers
* **Video Analytics Pipeline:** Streamlit-Webrtc Framework & PyAV Media Library
* **Structured Data Telemetry Engine:** Pandas Framework

---

## 📂 Project Structure
```text
RoadSense-AI/
├── backend/
│   ├── main.py
│   ├── webcam.py
│   └── yolo12s_RDD2022_best.pt      # Fine-tuned YOLO weights
├── frontend/
│   └── app.py                        # Multi-mode Streamlit Dashboard app
├── requirements.txt                  # Deployment environment dependencies
└── README.md                         # Project documentation