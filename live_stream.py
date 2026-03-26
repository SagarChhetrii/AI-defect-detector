"""
Streamlit Live Continuous Detection
Real-time detection with webcam streaming in the browser
"""

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time

# Page config
st.set_page_config(page_title="Live Detection", layout="wide")

st.title("🎥 Live Continuous Object Detection")

@st.cache_resource
def load_models():
    """Load detection and defect models"""
    detect_model = YOLO("yolov8n.pt")
    
    casting_model = None
    model_path = Path("models/casting_defect_model.pt")
    if model_path.exists():
        casting_model = YOLO(str(model_path))
    
    return detect_model, casting_model


def detect_and_draw(frame, detect_model):
    """Detect objects and return annotated frame"""
    
    # Run detection
    results = detect_model(frame, verbose=False, conf=0.5)
    
    detection_info = None
    
    if len(results) > 0:
        result = results[0]
        
        exclude_classes = ['person', 'human']
        best_detection = None
        best_confidence = 0
        
        if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
            # Find best non-person detection
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                object_name = result.names[class_id]
                
                if object_name.lower() in exclude_classes or confidence < 0.5:
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
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2
                    }
        
        # Draw detection on frame
        if best_detection:
            x1, y1, x2, y2 = best_detection['x1'], best_detection['y1'], best_detection['x2'], best_detection['y2']
            conf = best_detection['confidence']
            name = best_detection['name']
            
            detection_info = best_detection
            
            # Draw green bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Draw corner circles
            circle_radius = 10
            for point in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                cv2.circle(frame, point, circle_radius, (0, 255, 0), -1)
                cv2.circle(frame, point, circle_radius - 3, (0, 0, 0), 2)
            
            # Center circle
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), circle_radius - 2, (0, 255, 0), -1)
            cv2.circle(frame, (center_x, center_y), circle_radius - 5, (0, 0, 0), 2)
            
            # Label
            label = f"{name.upper()} ({conf*100:.0f}%)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(label, font, 0.8, 2)[0]
            
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 10),
                         (x1 + text_size[0] + 8, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 4),
                       font, 0.8, (0, 0, 0), 2)
            
            # Draw coordinates
            font_small = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"({x1},{y1})", (x1 - 60, y1 - 15),
                       font_small, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"({x2},{y2})", (x2 + 5, y2 + 20),
                       font_small, 0.6, (0, 255, 0), 2)
            
            # Dimensions
            width = x2 - x1
            height = y2 - y1
            dim_text = f"W:{width}px H:{height}px"
            dim_size = cv2.getTextSize(dim_text, font_small, 0.6, 2)[0]
            
            cv2.rectangle(frame, (center_x - dim_size[0]//2 - 6, center_y - 12),
                         (center_x + dim_size[0]//2 + 6, center_y + 8), (0, 0, 0), -1)
            cv2.putText(frame, dim_text, (center_x - dim_size[0]//2, center_y + 3),
                       font_small, 0.6, (0, 255, 255), 2)
    
    return frame, detection_info


# Load models
detect_model, casting_model = load_models()

st.markdown("---")
st.write("📌 **Confidence Threshold: 50% | Excludes people | Shows highest-confidence object**")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Feed")
    
    # Get webcam access
    camera_input = st.camera_input("Capture continuous frames", key="live_camera")
    
    if camera_input is not None:
        # Open image
        from PIL import Image
        pil_image = Image.open(camera_input)
        image_rgb = np.array(pil_image)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Detect
        with st.spinner("🔍 Detecting..."):
            annotated_frame, detection_info = detect_and_draw(image_bgr, detect_model)
        
        # Display
        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="Detection Result", use_column_width='always')

with col2:
    st.subheader("📊 Detection Info")
    
    if camera_input is not None:
        if detection_info:
            st.success(f"✅ Object Detected")
            st.metric("Type", detection_info['name'].upper())
            st.metric("Confidence", f"{detection_info['confidence']*100:.1f}%")
            
            st.markdown("**Coordinates:**")
            x1 = detection_info['x1']
            y1 = detection_info['y1']
            x2 = detection_info['x2']
            y2 = detection_info['y2']
            
            st.write(f"X1: {x1}")
            st.write(f"Y1: {y1}")
            st.write(f"X2: {x2}")
            st.write(f"Y2: {y2}")
            
            width = x2 - x1
            height = y2 - y1
            st.write(f"**Width:** {width}px")
            st.write(f"**Height:** {height}px")
        else:
            st.warning("⚠️ No objects detected")
            st.info("Try capturing with better lighting or clearer objects")
    else:
        st.info("📷 Capture an image to start detection")

st.markdown("---")
st.markdown("""
**Instructions:**
1. Click 'Capture continuous frames' to start
2. Take multiple photos for live detection
3. System shows highest-confidence object
4. Check coordinates and dimensions in real-time
""")
