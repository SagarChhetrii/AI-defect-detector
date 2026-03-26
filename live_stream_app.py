"""
Live Streamlit Video Processing
Real-time defect detection with continuous refresh
"""

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import time
from datetime import datetime

# Page config
st.set_page_config(
    page_title="🎥 Live Stream Processor",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card { background-color: #f5f5f5; padding: 20px; border-radius: 8px; border-left: 4px solid #1f77b4; margin: 10px 0; }
    .status-ok { background-color: #e8f5e9; border-left-color: #2ca02c; }
    .status-defect { background-color: #ffebee; border-left-color: #d62728; }
    .detection-box { padding: 15px; border-radius: 8px; margin: 10px 0; }
    .detection-ok { background-color: #e8f5e9; border-left: 4px solid #2ca02c; }
    .detection-defect { background-color: #ffebee; border-left: 4px solid #d62728; }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_detection_model():
    return YOLO("yolov8n.pt")

@st.cache_resource
def load_casting_model():
    model_path = Path("models/casting_defect_model_v2.pt")
    if model_path.exists():
        return YOLO(str(model_path))
    return YOLO("models/casting_defect_model.pt")

def detect_object(image_bgr, detect_model):
    """Detect objects in image"""
    results = detect_model(image_bgr, verbose=False, conf=0.5)
    
    exclude_classes = ['person', 'human']
    min_confidence = 0.5
    
    if len(results) > 0:
        result = results[0]
        if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
            best_detection = None
            best_confidence = 0
            
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                object_name = result.names[class_id]
                
                if object_name.lower() in exclude_classes or confidence < min_confidence:
                    continue
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    xyxy = box.xyxy[0]
                    if hasattr(xyxy, 'cpu'):
                        xyxy = xyxy.cpu().numpy()
                    elif hasattr(xyxy, 'numpy'):
                        xyxy = xyxy.numpy()
                    
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    
                    best_detection = {
                        "name": object_name,
                        "confidence": confidence,
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                    }
            
            return best_detection
    return None

def analyze_defect(image_bgr, bbox, casting_model):
    """Analyze detected object for defects"""
    if image_bgr is None or bbox is None:
        return None, None
    
    try:
        x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
        
        pad = 5
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(image_bgr.shape[1], x2 + pad)
        y2 = min(image_bgr.shape[0], y2 + pad)
        
        cropped = image_bgr[y1:y2, x1:x2]
        
        if cropped.shape[0] < 10 or cropped.shape[1] < 10:
            return None, None
        
        results = casting_model(cropped, verbose=False)
        if len(results) > 0:
            result = results[0]
            class_id = int(result.probs.top1)
            confidence = float(result.probs.top1conf)
            class_name = result.names[class_id]
            return class_name, confidence
    except:
        pass
    
    return None, None

def draw_detection(image_bgr, detection):
    """Draw detection with all info"""
    if detection is None:
        return image_bgr
    
    try:
        img = image_bgr.copy()
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        confidence = detection['confidence']
        name = detection['name']
        width = x2 - x1
        height = y2 - y1
        
        # Green bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Corner circles
        circle_radius = 10
        cv2.circle(img, (x1, y1), circle_radius, (0, 255, 0), -1)
        cv2.circle(img, (x2, y1), circle_radius, (0, 255, 0), -1)
        cv2.circle(img, (x1, y2), circle_radius, (0, 255, 0), -1)
        cv2.circle(img, (x2, y2), circle_radius, (0, 255, 0), -1)
        
        # Label
        label = f"{name.upper()} ({confidence*100:.0f}%)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        
        cv2.rectangle(img, 
                      (x1, y1 - text_size[1] - 10),
                      (x1 + text_size[0] + 8, y1),
                      (0, 255, 0), -1)
        
        cv2.putText(img, label, (x1 + 4, y1 - 5),
                    font, font_scale, (0, 0, 0), thickness)
        
        # Coordinates
        font_small = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"({x1},{y1})", (x1 - 60, y1 - 15),
                    font_small, 0.6, (0, 255, 0), 1)
        cv2.putText(img, f"({x2},{y1})", (x2 + 5, y1 - 15),
                    font_small, 0.6, (0, 255, 0), 1)
        cv2.putText(img, f"({x1},{y2})", (x1 - 60, y2 + 20),
                    font_small, 0.6, (0, 255, 0), 1)
        cv2.putText(img, f"({x2},{y2})", (x2 + 5, y2 + 20),
                    font_small, 0.6, (0, 255, 0), 1)
        
        # Dimensions
        dim_text = f"W:{width} H:{height}"
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        dim_size = cv2.getTextSize(dim_text, font_small, 0.6, 1)[0]
        
        cv2.rectangle(img,
                      (center_x - dim_size[0]//2 - 5, center_y - 12),
                      (center_x + dim_size[0]//2 + 5, center_y + 8),
                      (0, 0, 0), -1)
        
        cv2.putText(img, dim_text, (center_x - dim_size[0]//2, center_y + 5),
                    font_small, 0.6, (0, 255, 255), 1)
        
        return img
    except:
        return image_bgr

# Header
st.markdown("# 🎥 Live Stream Defect Detection")
st.markdown("**Real-time continuous processing with descriptions**")

# Sidebar controls
st.sidebar.markdown("## ⚙️ Settings")
refresh_rate = st.sidebar.slider("Refresh Rate (ms)", 100, 1000, 200, step=50)
show_coordinates = st.sidebar.checkbox("Show Coordinates", value=True)
show_stats = st.sidebar.checkbox("Show Statistics", value=True)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 📹 Live Camera Feed")
    st.info("📌 **Processing:** Real-time object detection + defect analysis on continuous frames")
    
    # Placeholder for video
    video_placeholder = st.empty()
    frame_info_placeholder = st.empty()

with col2:
    st.markdown("## 📊 Live Statistics")
    
    stats_col1, stats_col2 = st.columns(2)
    with stats_col1:
        frame_counter = st.empty()
        detection_counter = st.empty()
    
    with stats_col2:
        ok_counter = st.empty()
        defect_counter = st.empty()
    
    st.markdown("---")
    st.markdown("## 🎯 Current Detection")
    detection_info = st.empty()
    defect_info = st.empty()

# Load models
detect_model = load_detection_model()
casting_model = load_casting_model()

# Initialize counters
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
    st.session_state.detection_count = 0
    st.session_state.ok_count = 0
    st.session_state.defect_count = 0

# Open webcam
cap = cv2.VideoCapture(0)

if cap.isOpened():
    st.session_state.webcam_active = True
    
    with st.spinner("🎥 Starting live stream..."):
        for i in range(10):  # Show 10 frames continuously
            ret, frame = cap.read()
            
            if not ret:
                st.error("❌ Failed to read from webcam")
                break
            
            st.session_state.frame_count += 1
            
            # Resize
            frame_resized = cv2.resize(frame, (640, 480))
            
            # Detect
            detection = detect_object(frame_resized, detect_model)
            
            # Analyze defect
            defect_status = None
            defect_conf = None
            if detection:
                st.session_state.detection_count += 1
                defect_status, defect_conf = analyze_defect(frame_resized, detection['bbox'], casting_model)
                
                if defect_status:
                    is_ok = "ok" in defect_status.lower()
                    if is_ok:
                        st.session_state.ok_count += 1
                    else:
                        st.session_state.defect_count += 1
            
            # Draw detection
            frame_with_detection = draw_detection(frame_resized, detection)
            frame_rgb = cv2.cvtColor(frame_with_detection, cv2.COLOR_BGR2RGB)
            
            # Update displays
            with video_placeholder:
                st.image(frame_rgb, use_column_width='always', caption=f"Frame {st.session_state.frame_count}")
            
            # Update stats
            with frame_counter:
                st.metric("Total Frames", st.session_state.frame_count)
            
            with detection_counter:
                st.metric("Objects Detected", st.session_state.detection_count)
            
            with ok_counter:
                st.metric("✅ OK Items", st.session_state.ok_count)
            
            with defect_counter:
                st.metric("❌ Defects Found", st.session_state.defect_count)
            
            # Update detection info
            if detection:
                with detection_info:
                    st.markdown(f"""
                    <div class='detection-box detection-ok'>
                        <strong>🔍 Detected:</strong> {detection['name'].upper()}<br>
                        <strong>📊 Confidence:</strong> {detection['confidence']*100:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                
                if show_coordinates:
                    bbox = detection['bbox']
                    st.write(f"""
                    **Coordinates:**
                    - X1: {bbox['x1']}
                    - Y1: {bbox['y1']}
                    - X2: {bbox['x2']}
                    - Y2: {bbox['y2']}
                    - Width: {bbox['x2'] - bbox['x1']}px
                    - Height: {bbox['y2'] - bbox['y1']}px
                    """)
                
                if defect_status:
                    is_ok = "ok" in defect_status.lower()
                    if is_ok:
                        with defect_info:
                            st.markdown(f"""
                            <div class='detection-box detection-ok'>
                                <strong>✅ DEFECT STATUS: NO DEFECTS</strong><br>
                                Confidence: {defect_conf*100:.1f}%<br>
                                Status: <strong>READY TO SHIP</strong>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        with defect_info:
                            st.markdown(f"""
                            <div class='detection-box detection-defect'>
                                <strong>❌ DEFECT STATUS: DEFECT FOUND</strong><br>
                                Type: {defect_status}<br>
                                Confidence: {defect_conf*100:.1f}%<br>
                                Status: <strong>REQUIRES REWORK</strong>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                with detection_info:
                    st.markdown("""
                    <div class='detection-box' style='background-color: #fff3cd; border-left: 4px solid #ff9800;'>
                        <strong>⚠️ NO OBJECTS DETECTED</strong><br>
                        Try adjusting camera angle or lighting
                    </div>
                    """, unsafe_allow_html=True)
            
            # Wait before next frame
            time.sleep(refresh_rate / 1000.0)
    
    cap.release()
else:
    st.error("❌ Could not open webcam. Make sure you have camera permissions.")

# Footer
st.markdown("---")
st.markdown("""
**Live Stream Controls:**
- **Refresh Rate:** Adjust how fast frames are processed
- **Show Coordinates:** Toggle coordinate display
- **Show Statistics:** Toggle live statistics

**Processing Features:**
- Real-time object detection (80+ classes)
- Automatic person filtering
- Defect classification on detected objects
- Continuous frame analysis
- Live coordinate tracking
""")
