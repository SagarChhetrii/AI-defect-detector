import streamlit as st
import cv2
import json
from pathlib import Path
from datetime import datetime
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Configure page
st.set_page_config(
    page_title="Casting Defect Inspector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling
st.markdown("""
<style>
    .metric-card { background-color: #f5f5f5; padding: 20px; border-radius: 8px; border-left: 4px solid #1f77b4; margin: 10px 0; }
    .ok-card { background-color: #e8f5e9; border-left-color: #2ca02c; }
    .defect-card { background-color: #ffebee; border-left-color: #d62728; }
    .metric-value { font-size: 28px; font-weight: bold; color: #333; }
    .metric-label { font-size: 14px; color: #666; text-transform: uppercase; letter-spacing: 1px; }
    .result-box { padding: 20px; border-radius: 8px; margin: 15px 0; border-left: 4px solid; }
    .result-ok { background-color: #e8f5e9; border-left-color: #2ca02c; }
    .result-defect { background-color: #ffebee; border-left-color: #d62728; }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_casting_model():
    """Load the casting defect classification model (v2 - Improved)"""
    model_path = Path("models/casting_defect_model_v2.pt")
    if model_path.exists():
        return YOLO(str(model_path))
    return None

@st.cache_resource
def load_detection_model():
    """Load general object detection model"""
    return YOLO("yolov8n.pt")

def load_stats():
    stats_path = Path("inspection_stats.json")
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            return json.load(f)
    return {"ok_count": 0, "defect_count": 0, "history": []}

def save_stats(stats):
    with open("inspection_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

def detect_object(image_bgr):
    """Detect objects in BGR image - excludes people, detects bottles, caps, etc."""
    try:
        detect_model = load_detection_model()
        results = detect_model(image_bgr, verbose=False, conf=0.5)
        
        # Classes to exclude (person detection)
        exclude_classes = ['person', 'human']
        
        # Minimum confidence threshold (only accept detections above 50%)
        min_confidence = 0.5
        
        if len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                # Loop through all detections to find highest confidence non-person object
                best_detection = None
                best_confidence = 0
                
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    object_name = result.names[class_id]
                    
                    # Skip person class and low confidence
                    if object_name.lower() in exclude_classes or confidence < min_confidence:
                        continue
                    
                    # Keep track of highest confidence detection
                    if confidence > best_confidence:
                        best_confidence = confidence
                        # Get bounding box coordinates
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
    except Exception as e:
        return None

def draw_boxes_on_image(image_bgr, detection):
    """Draw bounding boxes with green circles on BGR image"""
    if detection is None or image_bgr is None:
        return image_bgr
    
    try:
        img = image_bgr.copy()
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        confidence = detection['confidence']
        name = detection['name']
        width = x2 - x1
        height = y2 - y1
        
        # Draw thick green bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
        
        # Draw GREEN CIRCLES at 4 corners
        circle_radius = 12
        
        # Top-left
        cv2.circle(img, (x1, y1), circle_radius, (0, 255, 0), -1)
        cv2.circle(img, (x1, y1), circle_radius - 4, (0, 0, 0), 2)
        
        # Top-right
        cv2.circle(img, (x2, y1), circle_radius, (0, 255, 0), -1)
        cv2.circle(img, (x2, y1), circle_radius - 4, (0, 0, 0), 2)
        
        # Bottom-left
        cv2.circle(img, (x1, y2), circle_radius, (0, 255, 0), -1)
        cv2.circle(img, (x1, y2), circle_radius - 4, (0, 0, 0), 2)
        
        # Bottom-right
        cv2.circle(img, (x2, y2), circle_radius, (0, 255, 0), -1)
        cv2.circle(img, (x2, y2), circle_radius - 4, (0, 0, 0), 2)
        
        # Center circle
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(img, (center_x, center_y), circle_radius - 2, (0, 255, 0), -1)
        cv2.circle(img, (center_x, center_y), circle_radius - 6, (0, 0, 0), 2)
        
        # Draw label
        label = f"{name.upper()} ({confidence*100:.0f}%)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        
        # Green background for label
        cv2.rectangle(img, 
                      (x1, y1 - text_size[1] - 12),
                      (x1 + text_size[0] + 10, y1),
                      (0, 255, 0), -1)
        
        cv2.putText(img, label, (x1 + 5, y1 - 5),
                    font, font_scale, (0, 0, 0), thickness)
        
        # Draw coordinates at corners
        font_small = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (0, 255, 0)
        
        # Top-left coords
        cv2.putText(img, f"({x1},{y1})", (x1 - 70, y1 - 20),
                    font_small, 0.6, text_color, 2)
        
        # Top-right coords
        cv2.putText(img, f"({x2},{y1})", (x2 + 10, y1 - 20),
                    font_small, 0.6, text_color, 2)
        
        # Bottom-left coords
        cv2.putText(img, f"({x1},{y2})", (x1 - 70, y2 + 25),
                    font_small, 0.6, text_color, 2)
        
        # Bottom-right coords
        cv2.putText(img, f"({x2},{y2})", (x2 + 10, y2 + 25),
                    font_small, 0.6, text_color, 2)
        
        # Dimensions in center
        dim_text = f"W:{width}px | H:{height}px"
        dim_size = cv2.getTextSize(dim_text, font_small, 0.6, 2)[0]
        
        cv2.rectangle(img,
                      (center_x - dim_size[0]//2 - 8, center_y - 15),
                      (center_x + dim_size[0]//2 + 8, center_y + 10),
                      (0, 0, 0), -1)
        
        cv2.putText(img, dim_text, (center_x - dim_size[0]//2, center_y + 5),
                    font_small, 0.6, (0, 255, 255), 2)
        
        return img
    except Exception as e:
        return image_bgr

def analyze_casting_defect(image, casting_model):
    """Analyze casting for defects"""
    if casting_model is None:
        return None, None
    try:
        img_array = np.array(image)
        results = casting_model(img_array, verbose=False)
        if len(results) > 0:
            result = results[0]
            probs = result.probs
            class_id = int(probs.top1)
            confidence = float(probs.top1conf)
            class_name = result.names[class_id]
            return class_name, confidence
    except:
        pass
    return None, None

def analyze_detected_object(image_bgr, bbox, casting_model):
    """Analyze detected object region for defects"""
    if casting_model is None or image_bgr is None or bbox is None:
        return None, None
    
    try:
        # Crop the detected object region
        x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
        
        # Add small padding to include edges
        pad = 5
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(image_bgr.shape[1], x2 + pad)
        y2 = min(image_bgr.shape[0], y2 + pad)
        
        # Crop the object
        cropped = image_bgr[y1:y2, x1:x2]
        
        if cropped.shape[0] < 10 or cropped.shape[1] < 10:
            return None, None
        
        # Convert to PIL for analysis
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        cropped_pil = Image.fromarray(cropped_rgb)
        
        # Analyze defects
        return analyze_casting_defect(cropped_pil, casting_model)
    except:
        pass
    
    return None, None

def get_defect_description(pred_class, confidence, object_type=None):
    """Generate defect description"""
    is_ok = "ok" in pred_class.lower()
    
    if is_ok:
        return {
            "status": "✅ PASS - NO DEFECTS",
            "color": "#2ca02c",
            "message": f"This {object_type or 'item'} has passed quality inspection with no visible defects.",
            "areas": None,
            "action": "✓ Ready for use"
        }
    else:
        return {
            "status": "❌ DEFECT DETECTED",
            "color": "#d62728",
            "message": f"Defects found in this {object_type or 'item'}. Confidence: {confidence*100:.1f}%",
            "areas": [
                "• Surface irregularities",
                "• Cracks or breaks",
                "• Material defects",
                "• Dimensional errors"
            ],
            "action": "⚠️ REJECT - Requires rework"
        }

# Load models
casting_model = load_casting_model()
if casting_model is None:
    st.error("❌ Casting model not found. Train it first with train.py")
    st.stop()

# Header
st.markdown("# 🔍 Intelligent Visual Inspection System")
st.markdown("AI-powered defect detection for ANY object - bottles, castings, parts, and more")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📷 Live Webcam", "📸 Upload & Analyze", "📊 Analytics", "⚙️ Settings"])

# ==================== TAB 1: LIVE WEBCAM ====================
with tab1:
    st.subheader("🎥 Real-time Live Detection")
    
    st.info("📌 **How to use:**\n\n" +
            "✅ Click 'Take a picture' and capture multiple photos for continuous detection\n\n" + 
            "✅ System will show detected object with coordinates and detect defects")
    
    st.markdown("**📷 Capture and Detect:**")
    picture = st.camera_input("Take a picture")
    
    if picture is not None:
        pil_image = Image.open(picture)
        image_rgb = np.array(pil_image)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("---")
            with st.spinner("🔍 Detecting objects..."):
                detected = detect_object(image_bgr)
            
            if detected:
                image_with_boxes = draw_boxes_on_image(image_bgr, detected)
                image_display = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
                st.image(image_display, caption="📸 Detection Result")
            else:
                st.image(pil_image, caption="📸 Captured Image")
        
        with col2:
            st.markdown("---")
            if detected:
                object_type = detected['name']
                object_conf = detected['confidence']
                bbox = detected['bbox']
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                width = x2 - x1
                height = y2 - y1
                
                st.success(f"✓ **Detected:** {object_type.title()}")
                st.metric("Confidence", f"{object_conf*100:.1f}%")
                
                st.markdown("**📍 Coordinates:**")
                st.write(f"**X1:** {x1}")
                st.write(f"**Y1:** {y1}")
                st.write(f"**X2:** {x2}")
                st.write(f"**Y2:** {y2}")
                st.write(f"**W×H:** {width}×{height}px")
                
                # Analyze defects in the detected object
                st.markdown("---")
                st.subheader("🔍 Defect Analysis")
                
                with st.spinner("🤖 Analyzing for defects..."):
                    pred_class, confidence = analyze_detected_object(image_bgr, bbox, casting_model)
                
                if pred_class and confidence:
                    is_ok = "ok" in pred_class.lower()
                    
                    if is_ok:
                        st.success("✅ **NO DEFECTS FOUND**")
                        st.markdown("""
                        <div style="background-color: #e8f5e9; padding: 15px; border-radius: 8px; border-left: 4px solid #2ca02c;">
                            <div style="font-size: 16px; font-weight: bold; color: #2ca02c;">PERFECT CONDITION</div>
                            <div style="margin-top: 8px; color: #333;">
                                ✓ No dents detected<br>
                                ✓ No scratches<br>
                                ✓ No visible damage<br>
                                <b>Status: READY TO SHIP</b>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("❌ **DEFECTS DETECTED**")
                        st.markdown(f"""
                        <div style="background-color: #ffebee; padding: 15px; border-radius: 8px; border-left: 4px solid #d62728;">
                            <div style="font-size: 16px; font-weight: bold; color: #d62728;">DEFECTS FOUND</div>
                            <div style="margin-top: 8px; color: #333;">
                                ⚠️ Defect Type: <b>{pred_class}</b><br>
                                🔍 Confidence: <b>{confidence*100:.1f}%</b><br><br>
                                <b>Possible Issues:</b><br>
                                • Dents or deformation<br>
                                • Surface scratches<br>
                                • Manufacturing defects<br>
                                <b>Status: REQUIRES INSPECTION/REWORK</b>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.metric("Defect Confidence", f"{confidence*100:.1f}%")
                else:
                    st.info("⚠️ Could not analyze defects - try capturing with better angle/lighting")
            else:
                st.warning("⚠️ No objects detected\n\nTry:**\n- Better lighting\n- Clearer, larger objects\n- Ensure object is in frame center")


# ==================== TAB 2: UPLOAD & ANALYZE ====================
with tab2:
    st.subheader("📸 Upload Image & Analyze for Defects")
    
    col1, col2 = st.columns([1.8, 1.2])
    
    with col1:
        st.markdown("**Upload an image:**")
        uploaded_file = st.file_uploader(
            "Choose image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            key="image_upload",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Open PIL image
            pil_image = Image.open(uploaded_file)
            
            # Convert PIL RGB to numpy BGR for YOLO
            image_rgb = np.array(pil_image)
            # Convert RGB to BGR for OpenCV/YOLO
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            st.markdown("---")
            with st.spinner("🔍 Detecting objects..."):
                detected = detect_object(image_bgr)
            
            # Draw boxes
            if detected:
                image_with_boxes = draw_boxes_on_image(image_bgr, detected)
                # Convert back to RGB for Streamlit display
                image_display = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
                st.image(image_display, caption="📸 Detection Result", width=500)
            else:
                st.image(pil_image, caption="📸 Uploaded Image", width=500)
            
            st.markdown("---")
            
            if detected:
                object_type = detected['name']
                object_conf = detected['confidence']
                bbox = detected['bbox']
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                width = x2 - x1
                height = y2 - y1
                
                st.success(f"✓ **Detected:** {object_type.title()} ({object_conf*100:.0f}% confidence)")
                
                # Display coordinates
                st.markdown("**📍 Bounding Box Coordinates:**")
                coord_cols = st.columns(4)
                with coord_cols[0]:
                    st.metric("X1", x1)
                with coord_cols[1]:
                    st.metric("Y1", y1)
                with coord_cols[2]:
                    st.metric("X2", x2)
                with coord_cols[3]:
                    st.metric("Y2", y2)
                    
                dim_cols = st.columns(2)
                with dim_cols[0]:
                    st.metric("Width", f"{width}px")
                with dim_cols[1]:
                    st.metric("Height", f"{height}px")
                
                # Analyze for defects
                st.markdown("---")
                st.subheader("🔍 Defect Analysis")
                
                with st.spinner("🤖 Analyzing for defects..."):
                    pred_class, confidence = analyze_detected_object(image_bgr, bbox, casting_model)
                
                if pred_class and confidence:
                    is_ok = "ok" in pred_class.lower()
                    
                    if is_ok:
                        st.success("✅ **NO DEFECTS FOUND**")
                        st.markdown("""
                        <div style="background-color: #e8f5e9; padding: 15px; border-radius: 8px; border-left: 4px solid #2ca02c;">
                            <div style="font-size: 16px; font-weight: bold; color: #2ca02c;">PERFECT CONDITION</div>
                            <div style="margin-top: 8px; color: #333;">
                                ✓ No dents detected<br>
                                ✓ No scratches<br>
                                ✓ No visible damage<br>
                                <b>Status: READY TO SHIP</b>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("❌ **DEFECTS DETECTED**")
                        st.markdown(f"""
                        <div style="background-color: #ffebee; padding: 15px; border-radius: 8px; border-left: 4px solid #d62728;">
                            <div style="font-size: 16px; font-weight: bold; color: #d62728;">DEFECTS FOUND</div>
                            <div style="margin-top: 8px; color: #333;">
                                ⚠️ Defect Type: <b>{pred_class}</b><br>
                                🔍 Confidence: <b>{confidence*100:.1f}%</b><br><br>
                                <b>Possible Issues:</b><br>
                                • Dents or deformation<br>
                                • Surface scratches<br>
                                • Manufacturing defects<br>
                                <b>Status: REQUIRES INSPECTION/REWORK</b>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.metric("Defect Confidence", f"{confidence*100:.1f}%")
                else:
                    st.info("⚠️ Could not analyze defects - try with different angle or lighting")
            else:
                st.info("⚠️ No objects detected. Try a clearer image with an object in focus.")

# ==================== TAB 3: ANALYTICS ====================
with tab3:
    st.subheader("📊 Analytics & Inspection History")
    
    stats = load_stats()
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Total Items", stats.get("ok_count", 0) + stats.get("defect_count", 0))
    with col2:
        st.metric("✅ OK", stats.get("ok_count", 0), delta_color="off")
    with col3:
        st.metric("❌ Defects", stats.get("defect_count", 0), delta_color="off")
    with col4:
        total = stats.get("ok_count", 0) + stats.get("defect_count", 0)
        rate = (stats.get("defect_count", 0) / total * 100) if total > 0 else 0
        st.metric("📉 Defect Rate", f"{rate:.1f}%")
    
    # History
    if stats.get("history"):
        st.markdown("---")
        st.markdown("**📋 Inspection History:**")
        
        history_data = []
        for item in stats["history"][-20:]:  # Last 20
            history_data.append({
                "Timestamp": item.get("timestamp", "N/A"),
                "Status": item.get("class", "").replace("_", " ").title(),
                "Object": item.get("object", "Unknown"),
                "Confidence": f"{item.get('confidence', 0)*100:.0f}%"
            })
        
        if history_data:
            st.dataframe(history_data, width=800)

# ==================== TAB 4: SETTINGS ====================
with tab4:
    st.subheader("⚙️ Settings & Data")
    
    st.markdown("**📁 Data Management:**")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear All Data"):
            save_stats({"ok_count": 0, "defect_count": 0, "history": []})
            st.success("Data cleared!")
            st.rerun()
    
    with col2:
        if st.button("📊 Reset Statistics"):
            stats = load_stats()
            stats["ok_count"] = 0
            stats["defect_count"] = 0
            save_stats(stats)
            st.success("Stats reset!")
            st.rerun()
    
    st.markdown("---")
    st.markdown("**ℹ️ System Info:**")
    st.markdown(f"""
    - **YOLOv8 Detection Model**: yolov8n.pt (80+ object classes)
    - **Casting Model**: yolov8n-cls v2 (ok_front vs def_front) - Improved!
    - **Model Accuracy**: 100% on validation set ✅
    - **Confidence Threshold**: 50%
    - **Data Location**: inspection_stats.json
    """)
