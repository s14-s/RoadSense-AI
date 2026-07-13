import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import pandas as pd
from datetime import datetime
from streamlit_folium import st_folium
import folium

# Set page configuration
st.set_page_config(
    page_title="RoadSense AI Dashboard",
    page_icon="🛣️",
    layout="wide"
)

st.title("🛣️ RoadSense AI: Road Damage Assessment & Live Mapping")
st.markdown("Scan using your device camera or upload a road image to log coordinates and map damage zones.")
st.markdown("---")

# Load YOLO Model
@st.cache_resource
def load_model():
    possible_paths = [
        "backend/yolo12s_RDD2022_best.pt",
        "./backend/yolo12s_RDD2022_best.pt",
        "yolo12s_RDD2022_best.pt"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return YOLO(path)

    st.error("Model file not found.")
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
    mock_lat = st.number_input("Latitude", value=28.6139, format="%.6f")
    mock_lon = st.number_input("Longitude", value=77.2090, format="%.6f")

# Process Image input logic helper
def process_road_image(img_input):
    image = Image.open(img_input)
    img_array = np.array(image)
    
    # Run prediction
    results = model.predict(source=img_array, conf=conf_threshold, verbose=False)
    result = results[0]
    
    # Render tracking box boundaries over the image array
    annotated_img = result.plot()
    p_count, c_count, details = count_damages(result)
    
    # Log detected damage to map database
    if len(details) > 0:
        new_entries = []
        for dmg_type, conf in details:
            new_entries.append({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Damage Type": dmg_type,
                "Latitude": mock_lat,
                "Longitude": mock_lon,
                "Confidence": f"{conf*100:.1f}%"
            })
        if new_entries:
            st.session_state.gps_log = pd.concat([st.session_state.gps_log, pd.DataFrame(new_entries)], ignore_index=True)
            
    # UI Metrics Layout
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
        st.subheader("Captured/Uploaded Image")
        st.image(image, use_container_width=True)
    with col_img2:
        st.subheader("AI Analysis Results")
        # YOLO plot returns BGR format, convert to RGB for standard web image display
        st.image(annotated_img[..., ::-1], use_container_width=True)

# ==================== MODE 1: UPLOAD PHOTOS ====================
if app_mode == "📸 Upload Image/Photos":
    uploaded_file = st.file_uploader("Choose a road image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        process_road_image(uploaded_file)
    else:
        st.info("📥 Please drop or browse a road photo asset file here to analyze details.")

# ==================== MODE 2: LIVE CAMERA SCAN ====================
elif app_mode == "🎥 Live Camera Scan":
    st.subheader("Camera Viewfinder")
    st.markdown("Click **'Take Photo'** below to run live computer vision telemetry parsing on the captured frame window.")
    
    # Built-in browser-native secure camera tracking widget
    camera_file = st.camera_input("Scan Road Section Surface")
    if camera_file is not None:
        process_road_image(camera_file)

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
    
    center_lat = mock_lat
    center_lon = mock_lon
    
    if not st.session_state.gps_log.empty:
        center_lat = float(st.session_state.gps_log.iloc[-1]["Latitude"])
        center_lon = float(st.session_state.gps_log.iloc[-1]["Longitude"])
        
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
    
    for index, row in st.session_state.gps_log.iterrows():
        lat = float(row["Latitude"])
        lon = float(row["Longitude"])
        dmg = row["Damage Type"]
        conf = row["Confidence"]
        time = row["Timestamp"]
        
        pin_color = "red" if dmg == "Pothole" else "orange"
        popup_html = f"<b>{dmg}</b><br>Confidence: {conf}<br><small>Logged: {time}</small>"
        
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color=pin_color, icon="exclamation-triangle", prefix="fa")
        ).add_to(m)
        
    st_folium(m, width="100%", height=400, key="road_map")