"""
AI Visual Inspection System - Professional Dashboard
Clean, minimal industrial-style dashboard for real-time defect detection.
"""

import streamlit as st
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from ultralytics import YOLO
import cv2
import numpy as np
import threading
import time

# Page configuration
st.set_page_config(
    page_title="AI Visual Inspection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Minimal, clean CSS styling
st.markdown("""
    <style>
    /* Remove default padding and spacing */
    .main { padding-top: 0px; }
    
    /* Header styling */
    .header-section {
        padding: 24px 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 32px;
    }
    
    .header-title {
        font-size: 28px;
        font-weight: 600;
        color: #1a1a1a;
        margin: 0;
        line-height: 1.2;
    }
    
    .header-subtitle {
        font-size: 14px;
        color: #666666;
        margin-top: 4px;
    }
    
    /* Prediction card */
    .prediction-card {
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 24px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .status-ok {
        border-left: 4px solid #22c55e;
    }
    
    .status-defect {
        border-left: 4px solid #ef4444;
    }
    
    .status-label {
        font-size: 13px;
        color: #666666;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-value {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    
    .status-ok-text { color: #22c55e; }
    .status-defect-text { color: #ef4444; }
    
    .status-confidence {
        font-size: 13px;
        color: #999999;
        margin-top: 12px;
    }
    
    /* Metric cards */
    .metric-card {
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 20px;
        text-align: center;
    }
    
    .metric-label {
        font-size: 13px;
        color: #666666;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #1a1a1a;
    }
    
    /* Table styling */
    .logs-section {
        margin-top: 32px;
    }
    
    .section-title {
        font-size: 14px;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 16px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Footer */
    .footer-section {
        margin-top: 48px;
        padding-top: 16px;
        border-top: 1px solid #e0e0e0;
        color: #999999;
        font-size: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .status-indicator {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #22c55e;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'inspection_data' not in st.session_state:
    st.session_state.inspection_data = {
        'total_items': 0,
        'defective_items': 0,
        'ok_items': 0,
        'last_prediction': None,
        'last_confidence': 0.0,
        'last_timestamp': None,
        'history': []
    }

if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False
if 'latest_frame' not in st.session_state:
    st.session_state.latest_frame = None
if 'latest_prediction' not in st.session_state:
    st.session_state.latest_prediction = None
if 'webcam_thread' not in st.session_state:
    st.session_state.webcam_thread = None

@st.cache_resource
def load_model():
    """Load trained model"""
    model_path = Path("models/casting_defect_model.pt")
    if model_path.exists():
        return YOLO(str(model_path))
    return None

def process_image(image, model):
    """Run inference on image"""
    results = model(image, conf=0.5, verbose=False)
    if results and len(results) > 0:
        result = results[0]
        if result.probs is not None:
            pred_class = int(result.probs.top1)
            confidence = float(result.probs.top1conf)
            class_names = {0: "OK", 1: "DEFECT"}
            return class_names.get(pred_class, "Unknown"), confidence
    return None, None

def save_data():
    """Save inspection data to file"""
    data_file = Path("inspection_stats.json")
    with open(data_file, 'w') as f:
        json.dump(st.session_state.inspection_data, f, indent=4, default=str)

def load_data():
    """Load inspection data from file"""
    data_file = Path("inspection_stats.json")
    if data_file.exists():
        with open(data_file, 'r') as f:
            return json.load(f)
    return None

# Load saved data
saved_data = load_data()
if saved_data:
    st.session_state.inspection_data = saved_data

# ============================================================================
# HEADER SECTION
# ============================================================================
st.markdown("""
    <div class="header-section">
        <h1 class="header-title">AI Visual Inspection System</h1>
        <p class="header-subtitle">Real-time defect detection and quality monitoring</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN SECTION: Webcam Feed + Prediction Card
# ============================================================================
col_feed, col_prediction = st.columns([1.5, 1], gap="large")

# ============================================================================
# MAIN SECTION: Webcam Feed + Prediction Card
# ============================================================================
col_feed, col_prediction = st.columns([1.5, 1], gap="large")

with col_feed:
    st.markdown("#### Live Webcam Feed")
    
    # Create placeholder for webcam display
    webcam_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Webcam controls
    col_ctrl_a, col_ctrl_b = st.columns(2)
    
    with col_ctrl_a:
        camera_enabled = st.checkbox("📹 Enable Webcam", value=False, key="camera_toggle")
    
    with col_ctrl_b:
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    
    # Initialize webcam session state variables
    if "webcam_active" not in st.session_state:
        st.session_state.webcam_active = False
    if "latest_frame" not in st.session_state:
        st.session_state.latest_frame = None
    if "latest_prediction" not in st.session_state:
        st.session_state.latest_prediction = None
    
    def webcam_loop():
        """Continuous webcam capture loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        model = load_model()
        
        while st.session_state.webcam_active and camera_enabled:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Store frame in session state
            st.session_state.latest_frame = frame.copy()
            
            # Run inference
            if model:
                results = model(frame, conf=confidence_threshold, verbose=False)
                if results and len(results) > 0:
                    result = results[0]
                    if result.probs is not None:
                        pred_class = int(result.probs.top1)
                        confidence = float(result.probs.top1conf)
                        class_names = {0: "OK", 1: "DEFECT"}
                        st.session_state.latest_prediction = {
                            'class': class_names.get(pred_class, "Unknown"),
                            'confidence': confidence,
                            'timestamp': datetime.now().isoformat()
                        }
            
            time.sleep(0.033)  # ~30 FPS
        
        cap.release()
    
    # Handle webcam toggle
    if camera_enabled and not st.session_state.webcam_active:
        st.session_state.webcam_active = True
        st.info("🟢 Webcam starting...")
        time.sleep(1)
    elif not camera_enabled and st.session_state.webcam_active:
        st.session_state.webcam_active = False
        st.info("🔴 Webcam stopped")
        time.sleep(1)
    
    # Display live feed
    if camera_enabled:
        # Start webcam thread if not already running
        if "webcam_thread" not in st.session_state:
            st.session_state.webcam_thread = threading.Thread(target=webcam_loop, daemon=True)
            st.session_state.webcam_thread.start()
        
        # Display current frame
        if st.session_state.latest_frame is not None:
            frame_rgb = cv2.cvtColor(st.session_state.latest_frame, cv2.COLOR_BGR2RGB)
            
            # Draw prediction on frame if available
            if st.session_state.latest_prediction:
                pred = st.session_state.latest_prediction
                is_ok = pred['class'] == "OK"
                color = (34, 197, 94) if is_ok else (239, 68, 68)  # Green or Red (BGR)
                
                # Draw status box
                cv2.rectangle(frame_rgb, (10, 30), (300, 100), color, -1)
                cv2.putText(frame_rgb, f"Status: {pred['class']}", (20, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame_rgb, f"Confidence: {pred['confidence']:.1%}", (20, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            webcam_placeholder.image(frame_rgb, use_container_width=True)
        else:
            webcam_placeholder.info("⏳ Loading webcam...")
    else:
        # Placeholder when disabled
        webcam_placeholder.markdown(
            "<div style='width:100%; height:400px; background:#f5f5f5; "
            "border:1px solid #e0e0e0; border-radius:6px; "
            "display:flex; align-items:center; justify-content:center;'>"
            "<p style='color:#999; font-size:14px;'>Enable webcam to see live feed</p></div>",
            unsafe_allow_html=True
        )

with col_prediction:
    st.markdown("#### Current Prediction")
    
    # Get current prediction (from webcam or image upload)
    pred = st.session_state.latest_prediction or st.session_state.inspection_data.get('last_prediction')
    
    if pred:
        is_ok = pred.get('class') == "OK" or pred == "OK"
        if isinstance(pred, dict):
            status_text = pred.get('class', 'UNKNOWN')
            confidence = pred.get('confidence', 0)
        else:
            status_text = pred
            confidence = st.session_state.inspection_data.get('last_confidence', 0)
        
        status_class = "status-ok" if is_ok else "status-defect"
        text_class = "status-ok-text" if is_ok else "status-defect-text"
        
        st.markdown(f"""
            <div class="prediction-card {status_class}">
                <div class="status-label">Status</div>
                <div class="status-value {text_class}">{status_text}</div>
                <div class="status-confidence">Confidence: {confidence:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Add to stats button
        if st.button("✅ Confirm & Log", use_container_width=True):
            st.session_state.inspection_data['total_items'] += 1
            if is_ok:
                st.session_state.inspection_data['ok_items'] += 1
            else:
                st.session_state.inspection_data['defective_items'] += 1
            
            st.session_state.inspection_data['history'].append({
                'timestamp': datetime.now().isoformat(),
                'class': status_text,
                'confidence': float(confidence)
            })
            save_data()
            st.success("✅ Logged!")
            st.rerun()
    else:
        st.markdown("""
            <div class="prediction-card">
                <div class="status-label">Status</div>
                <div class="status-value" style="color:#ccc;">—</div>
                <div class="status-confidence">Enable webcam or upload image</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# METRICS ROW
# ============================================================================
st.markdown("")  # Spacing
st.markdown("#### Key Metrics")

col1, col2, col3, col4 = st.columns(4, gap="large")

total_items = st.session_state.inspection_data['total_items']
ok_count = st.session_state.inspection_data['ok_items']
defect_count = st.session_state.inspection_data['defective_items']
defect_rate = (defect_count / total_items * 100) if total_items > 0 else 0

with col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Items</div>
            <div class="metric-value">{total_items}</div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">OK Items</div>
            <div class="metric-value" style="color:#22c55e;">{ok_count}</div>
        </div>
        """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Defects</div>
            <div class="metric-value" style="color:#ef4444;">{defect_count}</div>
        </div>
        """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Defect Rate</div>
            <div class="metric-value">{defect_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# LOGS SECTION
# ============================================================================
st.markdown("")  # Spacing
st.markdown('<p class="section-title">Inspection History</p>', unsafe_allow_html=True)

# Add image upload section
st.markdown('<p style="font-size: 13px; color: #666; margin-bottom: 12px;">Or upload image for single inspection:</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload image",
    type=['jpg', 'jpeg', 'png', 'bmp'],
    key="image_upload"
)

if uploaded_file is not None:
    # Read and process image
    image = cv2.imdecode(
        np.frombuffer(uploaded_file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run inference
    model = load_model()
    if model:
        pred_class, confidence = process_image(image, model)
        
        col_img_display, col_img_pred = st.columns([1.5, 1])
        
        with col_img_display:
            st.image(image_rgb, use_container_width=True, caption="Uploaded Image")
        
        with col_img_pred:
            if pred_class:
                is_ok = pred_class == "OK"
                status_class = "status-ok" if is_ok else "status-defect"
                text_class = "status-ok-text" if is_ok else "status-defect-text"
                
                st.markdown(f"""
                    <div class="prediction-card {status_class}">
                        <div class="status-label">Prediction</div>
                        <div class="status-value {text_class}">{pred_class}</div>
                        <div class="status-confidence">{confidence:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if st.button("✅ Add to History", use_container_width=True, key="upload_button"):
                    st.session_state.inspection_data['total_items'] += 1
                    if pred_class == "OK":
                        st.session_state.inspection_data['ok_items'] += 1
                    else:
                        st.session_state.inspection_data['defective_items'] += 1
                    
                    st.session_state.inspection_data['history'].append({
                        'timestamp': datetime.now().isoformat(),
                        'class': pred_class,
                        'confidence': float(confidence)
                    })
                    save_data()
                    st.success("Added!")
                    st.rerun()
            else:
                st.warning("No detection found")

st.markdown("---")

if st.session_state.inspection_data['history']:
    # Create dataframe from history
    history_df = pd.DataFrame(st.session_state.inspection_data['history'])
    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
    history_df = history_df.sort_values('timestamp', ascending=False)
    
    # Format columns
    history_df['Time'] = history_df['timestamp'].dt.strftime('%H:%M:%S')
    history_df['Status'] = history_df['class']
    history_df['Confidence'] = history_df['confidence'].apply(lambda x: f"{x:.1%}")
    
    # Display table (only relevant columns)
    display_df = history_df[['Time', 'Status', 'Confidence']].reset_index(drop=True)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=300
    )
else:
    st.info("No inspection history yet. Use webcam or upload images to start.")

# ============================================================================
# CONTROL SECTION
# ============================================================================
st.markdown("")  # Spacing
st.markdown('<p class="section-title">Controls</p>', unsafe_allow_html=True)

col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3, gap="large")

with col_ctrl1:
    if st.button("✅ Add OK Item", use_container_width=True):
        st.session_state.inspection_data['total_items'] += 1
        st.session_state.inspection_data['ok_items'] += 1
        st.session_state.inspection_data['history'].append({
            'timestamp': datetime.now().isoformat(),
            'class': 'OK',
            'confidence': 1.0
        })
        save_data()
        st.rerun()

with col_ctrl2:
    if st.button("❌ Add Defect Item", use_container_width=True):
        st.session_state.inspection_data['total_items'] += 1
        st.session_state.inspection_data['defective_items'] += 1
        st.session_state.inspection_data['history'].append({
            'timestamp': datetime.now().isoformat(),
            'class': 'DEFECT',
            'confidence': 1.0
        })
        save_data()
        st.rerun()

with col_ctrl3:
    if st.button("🔄 Reset Data", use_container_width=True, type="secondary"):
        st.session_state.inspection_data = {
            'total_items': 0,
            'defective_items': 0,
            'ok_items': 0,
            'last_prediction': None,
            'last_confidence': 0.0,
            'last_timestamp': None,
            'history': []
        }
        save_data()
        st.rerun()

# ============================================================================
# FOOTER SECTION
# ============================================================================
st.markdown("")  # Spacing
st.markdown("""
    <div class="footer-section">
        <div class="status-indicator"></div>
        <span>System running</span>
    </div>
    """, unsafe_allow_html=True)
