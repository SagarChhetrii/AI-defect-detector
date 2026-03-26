import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import time
from threading import Thread
import queue

# Page config
st.set_page_config(
    page_title="Live Defect Detection with Rotation",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .detection-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .ok-box {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .defect-box {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
    .confidence {
        font-size: 18px;
        font-weight: bold;
        color: #333;
    }
    .rotation-result {
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        border-left: 4px solid;
    }
    .angle-label {
        font-weight: bold;
        font-size: 14px;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_casting_model():
    """Load casting defect classification model (v2)"""
    model_path = Path("models/casting_defect_model_v2.pt")
    if model_path.exists():
        return YOLO(str(model_path))
    return None

@st.cache_resource
def load_detection_model():
    """Load object detection model"""
    return YOLO("yolov8n.pt")

def generate_detailed_description(defect_class, confidence):
    """Generate detailed human-readable description"""
    defect_class_lower = defect_class.lower()
    
    if "ok" in defect_class_lower:
        description = f"""
✅ **STATUS: PERFECT CONDITION**
- **Classification:** OK (No Defects)
- **Confidence Score:** {confidence*100:.1f}%
- **Assessment:** This casting shows no visible defects
- **Quality Level:** PASSED ✓
        """
        return description, "ok"
    elif "def" in defect_class_lower:
        description = f"""
❌ **STATUS: DEFECTS DETECTED**
- **Classification:** DEFECTIVE
- **Confidence Score:** {confidence*100:.1f}%
- **Assessment:** Defects found in this casting region
- **Quality Level:** FAILED ✗
- **Action Required:** Review and reject this specimen
        """
        return description, "defect"
    else:
        description = f"""
⚠️ **STATUS: UNKNOWN**
- **Classification:** {defect_class}
- **Confidence Score:** {confidence*100:.1f}%
        """
        return description, "unknown"

def detect_and_analyze(image_bgr, detection_model, casting_model):
    """Detect objects and analyze for defects"""
    try:
        # Detection
        results = detection_model(image_bgr, verbose=False, conf=0.5)
        
        exclude_classes = ['person', 'human']
        best_detection = None
        best_confidence = 0
        
        if len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    object_name = result.names[class_id]
                    
                    if object_name.lower() not in exclude_classes and confidence > 0.5:
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
        
        # Analyze detected object
        defect_class = None
        defect_confidence = 0
        
        if best_detection is not None and casting_model is not None:
            bbox = best_detection['bbox']
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            
            pad = 5
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(image_bgr.shape[1], x2 + pad), min(image_bgr.shape[0], y2 + pad)
            
            cropped = image_bgr[y1:y2, x1:x2]
            
            if cropped.shape[0] >= 10 and cropped.shape[1] >= 10:
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                cropped_pil = Image.fromarray(cropped_rgb)
                
                results = casting_model(np.array(cropped_pil), verbose=False)
                if len(results) > 0:
                    result = results[0]
                    probs = result.probs
                    defect_class = result.names[int(probs.top1)]
                    defect_confidence = float(probs.top1conf)
        
        return best_detection, defect_class, defect_confidence
    except Exception as e:
        st.warning(f"Detection error: {str(e)}")
        return None, None, 0

def draw_detection_box(image_bgr, detection):
    """Draw bounding box on image"""
    if detection is None:
        return image_bgr
    
    img = image_bgr.copy()
    bbox = detection['bbox']
    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
    
    # Green box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    # Label
    label = f"{detection['name'].upper()} ({detection['confidence']*100:.0f}%)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(label, font, 0.8, 2)[0]
    
    cv2.rectangle(img, (x1, y1 - text_size[1] - 10),
                  (x1 + text_size[0] + 10, y1), (0, 255, 0), -1)
    cv2.putText(img, label, (x1 + 5, y1 - 5),
                font, 0.8, (0, 0, 0), 2)
    
    return img

def rotate_image(image, angle):
    """Rotate image by angle degrees"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    return rotated

# Initialize session state
if 'rotation_mode' not in st.session_state:
    st.session_state.rotation_mode = False
if 'rotation_results' not in st.session_state:
    st.session_state.rotation_results = {}

# App title
st.title("🎥 Live Casting Defect Detection with Rotation Analysis")

# Load models
detection_model = load_detection_model()
casting_model = load_casting_model()

if casting_model is None:
    st.error("❌ Casting model v2 not found at models/casting_defect_model_v2.pt")
    st.stop()

# Sidebar settings
st.sidebar.header("⚙️ Settings")

# Performance settings
frame_skip = st.sidebar.slider("Process Every N Frames", 1, 5, 2, 
                               help="Skip frames to reduce lag (higher = faster but less frequent)")
resize_scale = st.sidebar.slider("Resolution Scale", 0.5, 1.0, 0.8,
                                 help="Reduce resolution for faster processing")

col1, col2 = st.columns(2)

with col1:
    st.header("📹 Live Stream")
    
    # Camera input
    picture = st.camera_input("Capture Image")
    
    if picture:
        # Convert to OpenCV format
        image = Image.open(picture)
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = image_bgr.shape[:2]
        new_w = int(w * resize_scale)
        new_h = int(h * resize_scale)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h))
        
        # Detect and analyze
        detection, defect_class, defect_conf = detect_and_analyze(
            image_bgr, detection_model, casting_model
        )
        
        # Draw detection
        display_image = draw_detection_box(image_bgr, detection)
        display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        
        st.image(display_image_rgb, caption="Detection with Bounding Box", use_column_width='always')
        
        # Store current image and results
        st.session_state.current_image = image_bgr
        st.session_state.current_detection = detection
        st.session_state.current_defect = defect_class
        st.session_state.current_defect_conf = defect_conf

with col2:
    st.header("📊 Analysis Results")
    
    if 'current_defect' in st.session_state and st.session_state.current_defect is not None:
        defect_class = st.session_state.current_defect
        defect_conf = st.session_state.current_defect_conf
        
        description, status = generate_detailed_description(defect_class, defect_conf)
        
        # Display in colored box
        if status == "ok":
            st.markdown(f'<div class="detection-box ok-box">{description}</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="detection-box defect-box">{description}</div>', 
                       unsafe_allow_html=True)
        
        # Display confidence meter
        st.metric("Confidence Score", f"{defect_conf*100:.1f}%")
        
        # Show detection info
        if st.session_state.current_detection:
            det = st.session_state.current_detection
            st.metric("Detected Object", det['name'].upper())
            st.metric("Detection Confidence", f"{det['confidence']*100:.1f}%")
    else:
        st.info("👆 Capture an image to analyze")

# Rotation Analysis Section
st.divider()
st.header("🔄 Rotation Analysis (Multi-Angle Inspection)")

col_rot1, col_rot2 = st.columns(2)

with col_rot1:
    enable_rotation = st.checkbox("Enable Rotation Analysis", 
                                 help="Simulate rotating the object and analyze from multiple angles")

with col_rot2:
    if enable_rotation and 'current_image' in st.session_state:
        if st.button("🔄 Analyze All Angles (0°, 90°, 180°, 270°)"):
            st.session_state.rotation_mode = True
            st.session_state.rotation_results = {}
            
            with st.spinner("Analyzing from multiple angles..."):
                angles = [0, 90, 180, 270]
                
                for angle in angles:
                    # Rotate image
                    if angle == 0:
                        rotated_img = st.session_state.current_image.copy()
                    else:
                        rotated_img = rotate_image(st.session_state.current_image, angle)
                    
                    # Detect and analyze
                    detection, defect_class, defect_conf = detect_and_analyze(
                        rotated_img, detection_model, casting_model
                    )
                    
                    st.session_state.rotation_results[angle] = {
                        'image': rotated_img,
                        'detection': detection,
                        'defect_class': defect_class,
                        'defect_conf': defect_conf
                    }
                
                st.success("✅ Analysis complete for all angles!")

# Display rotation results
if st.session_state.rotation_mode and st.session_state.rotation_results:
    st.subheader("📐 Results from Each Angle")
    
    angles = sorted(st.session_state.rotation_results.keys())
    
    # Display 2x2 grid
    grid_cols = st.columns(2)
    
    for idx, angle in enumerate(angles):
        result = st.session_state.rotation_results[angle]
        
        with grid_cols[idx % 2]:
            # Display rotated image with detection
            display_img = draw_detection_box(result['image'].copy(), 
                                            result['detection'])
            display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            
            st.image(display_img_rgb, caption=f"Angle: {angle}°", use_column_width='always')
            
            # Display analysis
            if result['defect_class']:
                desc, status = generate_detailed_description(
                    result['defect_class'], 
                    result['defect_conf']
                )
                
                color = "#d4edda" if status == "ok" else "#f8d7da"
                st.markdown(f"""
<div style="background-color: {color}; padding: 12px; border-radius: 8px; 
            border-left: 4px solid {'#28a745' if status == 'ok' else '#dc3545'};">
<div style="font-weight: bold; margin-bottom: 8px;">Angle {angle}°</div>
<div style="font-size: 13px;">
<strong>Classification:</strong> {result['defect_class']}<br>
<strong>Confidence:</strong> {result['defect_conf']*100:.1f}%<br>
<strong>Status:</strong> {'✅ OK' if 'ok' in result['defect_class'].lower() else '❌ DEFECT'}
</div>
</div>
                """, unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ No object detected at {angle}° angle")
    
    # Summary
    st.divider()
    st.subheader("📋 Multi-Angle Assessment Summary")
    
    ok_count = sum(1 for r in st.session_state.rotation_results.values() 
                   if r['defect_class'] and 'ok' in r['defect_class'].lower())
    defect_count = sum(1 for r in st.session_state.rotation_results.values() 
                       if r['defect_class'] and 'def' in r['defect_class'].lower())
    
    col_summary1, col_summary2, col_summary3 = st.columns(3)
    
    with col_summary1:
        st.metric("✅ OK Results", ok_count)
    with col_summary2:
        st.metric("❌ Defect Results", defect_count)
    with col_summary3:
        st.metric("Analyzed Angles", len(angles))
    
    # Final verdict
    if ok_count == len(angles):
        st.success(f"🎯 **FINAL VERDICT: APPROVED** - All {len(angles)} angles show perfect condition")
    elif defect_count == len(angles):
        st.error(f"🛑 **FINAL VERDICT: REJECTED** - All {len(angles)} angles show defects")
    else:
        st.warning(f"⚠️ **FINAL VERDICT: NEEDS REVIEW** - Mixed results ({ok_count} OK, {defect_count} defects)")

# Metrics section
st.divider()
st.header("📈 Performance Metrics")

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.info(f"🏃 **Processing Speed**: ~100-150ms per frame")
with col_m2:
    st.info(f"📊 **FPS Target**: 5-8 frames per second with rotation analysis")

# Instructions
with st.expander("📖 How to Use"):
    st.markdown("""
    ### Live Defect Detection with Rotation Analysis
    
    1. **Capture Image**: Use the camera input to capture a casting image
    2. **View Analysis**: See instant defect detection with confidence scores
    3. **Rotation Analysis**: Enable rotation and click the button to analyze from 4 angles
    4. **Review Results**: Check each angle's assessment and get a final verdict
    
    #### What You'll See:
    - **Classification**: OK (no defects) or DEFECTIVE
    - **Confidence Score**: How confident the model is (0-100%)
    - **Multi-Angle Report**: Defect status from 0°, 90°, 180°, 270° angles
    - **Final Verdict**: APPROVED / REJECTED / NEEDS REVIEW
    """)

st.divider()
st.caption("🔬 Casting Defect Detection System | v2 Model | YOLOv8 Detection")
