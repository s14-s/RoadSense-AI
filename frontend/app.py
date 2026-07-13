import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import av
import queue
import pandas as pd
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from streamlit_folium import st_folium
import folium

# Set page configuration
st.set_page_config(
    page_title="RoadSense AI Dashboard",
    page_icon="🛣️",
    layout="wide"
)

st.title("🛣️ RoadSense AI: Damage Assessment & Live Mapping")
st.markdown("Scan live or upload a road image to log geographic coordinates and map damage zones.")
st.markdown("---")

# Load YOLO Model
@st.cache_resource
def load_model():
    model_path = os.path.join("backend", "yolo12s_RDD2022_best.pt")
    if not os.path.exists(model_path):
        model_path = "yolo12s_RDD2022_best.pt"
    return YOLO(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load YOLO model. Error: {e}")
    st.stop()

# Initialize session database
if "gps_log" not in st.session_state:
    st.session_state.gps_log = pd.DataFrame(columns=["Timestamp", "Damage Type", "Latitude", "Longitude", "Confidence"])

# Helper function to parse detections out of a YOLO result frame
def count_damages(result):
    potholes = 0
    cracks = 0
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id].lower()
            conf = float(box.conf[0])
            
            if "pothole" in class_name or "d40" in class_name:
                potholes += 1
                detections.append(("Pothole", conf))
            elif "crack" in class_name or class_name in ["d00", "d10", "d20"]:
                cracks += 1
                detections.append(("Crack", conf))
    return potholes, cracks, detections

# Sidebar Mode Selector
with st.sidebar:
    st.header("Navigation")
    app_mode = st.radio("Select Processing Mode", ["📸 Upload Image/Photos", "🎥 Live Camera Scan"])
    
    st.header("AI Settings")
    conf_threshold = st.slider("Model Confidence Threshold", 0.10, 1.00, 0.25, 0.05)
    
    st.header("Simulated GPS Coordinates")
    st.info("Change coordinates below to place pins in different map locations:")
    mock_lat = st.number_input("Latitude", value=28.6139, format="%.6f")
    mock_lon = st.number_input("Longitude", value=77.2090, format="%.6f")

# ==================== MODE 1: UPLOAD PHOTOS ====================
if app_mode == "📸 Upload Image/Photos":
    uploaded_file = st.file_uploader("Choose a road image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        with st.spinner("Analyzing uploaded asset..."):
            results = model.predict(source=img_array, conf=conf_threshold)
            
        result = results[0]
        annotated_img = result.plot()
        p_count, c_count, details = count_damages(result)
        
        # Log detected damage to map database
        if len(details) > 0:
            for dmg_type, conf in details:
                new_entry = pd.DataFrame([{
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Damage Type": dmg_type,
                    "Latitude": mock_lat,
                    "Longitude": mock_lon,
                    "Confidence": f"{conf*100:.1f}%"
                }])
                st.session_state.gps_log = pd.concat([st.session_state.gps_log, new_entry], ignore_index=True)
        
        # UI Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Detected Potholes", value=p_count)
        col2.metric(label="Detected Cracks", value=c_count)
        penalty = (p_count * 15) + (c_count * 5)
        health_score = max(0, 100 - penalty)
        status = "Excellent" if health_score > 80 else ("Fair" if health_score > 50 else "Critical")
        col3.metric(label="Road Health Score", value=f"{health_score}/100", delta=status)
        
        st.markdown("---")
        
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        with col_img2:
            st.subheader("AI Analysis Results")
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            st.image(annotated_img_rgb, use_container_width=True)

# ==================== MODE 2: LIVE CAMERA SCAN ====================
elif app_mode == "🎥 Live Camera Scan":
    result_queue = queue.Queue()

    col1, col2, col3 = st.columns(3)
    pothole_metric = col1.empty()
    crack_metric = col2.empty()
    health_metric = col3.empty()

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = model.predict(source=img, conf=conf_threshold, verbose=False)
        result = results[0]
        annotated_frame = result.plot()
        
        p_count, c_count, details = count_damages(result)
        result_queue.put({"potholes": p_count, "cracks": c_count, "details": details})
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    ctx = webrtc_streamer(
        key="road-sense-live-engine",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    while ctx.state.playing:
        try:
            data = result_queue.get(timeout=1.0)
            p = data["potholes"]
            c = data["cracks"]
            details = data["details"]
            
            penalty = (p * 15) + (c * 5)
            health_score = max(0, 100 - penalty)
            status = "Excellent" if health_score > 80 else ("Fair" if health_score > 50 else "Critical")
            
            pothole_metric.metric("Live Potholes", p)
            crack_metric.metric("Live Cracks", c)
            health_metric.metric("Road Health Score", f"{health_score}/100", delta=status)
            
            if len(details) > 0:
                for dmg_type, conf in details:
                    if st.session_state.gps_log.empty or (p > 0 and st.session_state.gps_log.iloc[-1]["Damage Type"] != dmg_type):
                        new_entry = pd.DataFrame([{
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Damage Type": dmg_type,
                            "Latitude": mock_lat,
                            "Longitude": mock_lon,
                            "Confidence": f"{conf*100:.1f}%"
                        }])
                        st.session_state.gps_log = pd.concat([st.session_state.gps_log, new_entry], ignore_index=True)
                        st.rerun() # Forces page map widget to refresh immediately
        except queue.Empty:
            continue

# ==================== VISUAL LIVE MAP & DATABASE DISPLAY ====================
st.markdown("---")
col_map, col_table = st.columns([3, 2])

with col_table:
    st.subheader("📍 Damage Incident Records")
    if not st.session_state.gps_log.empty:
        st.dataframe(st.session_state.gps_log.tail(8), use_container_width=True)
        if st.button("🗑️ Clear Logs"):
            st.session_state.gps_log = pd.DataFrame(columns=["Timestamp", "Damage Type", "Latitude", "Longitude", "Confidence"])
            st.rerun()
    else:
        st.info("No logs captured yet.")

with col_map:
    st.subheader("🗺️ Live Road Damage Mapping Network")
    
    # Default to center point coordinates
    center_lat = mock_lat
    center_lon = mock_lon
    
    if not st.session_state.gps_log.empty:
        center_lat = float(st.session_state.gps_log.iloc[-1]["Latitude"])
        center_lon = float(st.session_state.gps_log.iloc[-1]["Longitude"])
        
    # Generate interactive base map layout
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
    
    # Loop and insert historical marker pins onto map canvas
    for index, row in st.session_state.gps_log.iterrows():
        lat = float(row["Latitude"])
        lon = float(row["Longitude"])
        dmg = row["Damage Type"]
        conf = row["Confidence"]
        time = row["Timestamp"]
        
        # Color code: Red pins for potholes, orange pins for surface cracks
        pin_color = "red" if dmg == "Pothole" else "orange"
        
        popup_html = f"<b>{dmg}</b><br>Confidence: {conf}<br><small>Logged: {time}</small>"
        
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color=pin_color, icon="exclamation-triangle", prefix="fa")
        ).add_to(m)
        
    # Inject active interactive map element back into frontend interface
    st_folium(m, width="100%", height=400, key="road_map")