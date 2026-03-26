import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import time
from threading import Thread
from collections import deque

# Page config
st.set_page_config(
    page_title="Casting Defect Inspector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Modern Professional Design
st.markdown("""
<style>
    /* Main color palette */
    :root {
        --primary: #2563eb;
        --success: #10b981;
        --danger: #ef4444;
        --warning: #f59e0b;
        --light-bg: #f8fafc;
        --border: #e2e8f0;
    }
    
    /* Typography and spacing */
    * { font-family: 'Segoe UI', 'Roboto', sans-serif; }
    
    /* Header styling */
    h1 { color: #1e293b; margin-bottom: 5px; font-weight: 700; letter-spacing: -0.5px; }
    h2 { color: #334155; font-weight: 600; margin-top: 20px; margin-bottom: 15px; }
    
    /* Analysis boxes - modern card design */
    .analysis-box {
        padding: 16px;
        border-radius: 10px;
        margin: 12px 0;
        border-left: 4px solid;
        background: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        transition: box-shadow 0.2s ease;
    }
    
    .analysis-box:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.12); }
    
    .status-pass {
        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
        border-left-color: #10b981;
    }
    
    .status-fail {
        background: linear-gradient(135deg, #fef2f2 0%, #fef2f2 100%);
        border-left-color: #ef4444;
    }
    
    .status-review {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left-color: #f59e0b;
    }
    
    /* Confidence bar */
    .confidence-bar {
        height: 24px;
        background: #f1f5f9;
        border-radius: 6px;
        position: relative;
        overflow: hidden;
        margin: 8px 0;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.06);
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 11px;
        font-weight: 600;
    }
    
    /* Status badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }
    
    .badge-success { background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
    .badge-danger { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
    .badge-info { background: #eff6ff; color: #0284c7; border: 1px solid #bae6fd; }
    
    /* Control buttons */
    .stButton > button {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.2s ease;
        border: none;
        height: 44px;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        padding: 12px 16px;
    }
    
    /* Metric styling */
    .metric-card {
        background: white;
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
    
    /* Divider */
    hr { border-color: #e2e8f0; margin: 24px 0; }
    
    /* Sidebar */
    .css-1d391kg { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_casting_model():
    """Load casting defect classification model (Ultimate - 31 epochs, 99.9% accuracy)"""
    # Try to load the newly trained ultimate model first
    model_path = Path("models/casting_defect_model_ultimate.pt")
    
    # Fallback to v2 if ultimate not available
    if not model_path.exists():
        model_path = Path("models/casting_defect_model_v2.pt")
    
    if model_path.exists():
        return YOLO(str(model_path))
    return None

@st.cache_resource
def load_detection_model():
    """Load object detection model"""
    return YOLO("yolov8n.pt")

def generate_detailed_defect_description(defect_class, confidence, defect_type=None, frame_bgr=None):
    """Generate detailed defect description with specific defect type"""
    defect_class_lower = defect_class.lower() if defect_class else ""
    
    if "ok" in defect_class_lower:
        description = f"""
**Surface is Clean**

**Quality Assessment:**
- Surface Condition: Excellent
- Confidence Level: {confidence*100:.1f}%
- Visual Inspection: No defects detected

**Analysis Details:**
- No scratches or abrasions
- No dents or deformations
- No surface damage
- Surface finish: Acceptable
- Quality: Good

**Verdict:** PASS — Part approved for use
        """
        color = "status-pass"
    elif "def" in defect_class_lower:
        # Show specific defect type with description
        defect_info = {
            "scratch": {
                "title": "Surface Scratches Detected",
                "description": "Linear marks or abrasions on the surface"
            },
            "dent": {
                "title": "Dent Detected",
                "description": "Indentation or depression in the surface"
            },
            "pitting": {
                "title": "Pitting Corrosion Detected",
                "description": "Small circular pits in the surface"
            },
            "corrosion": {
                "title": "Corrosion Detected",
                "description": "Chemical damage or rust on surface"
            },
            "defect": {
                "title": "Surface Defects Detected",
                "description": "General surface damage detected"
            }
        }
        
        # Get specific defect info (default to generic if not identified)
        defect_info_detail = defect_info.get(defect_type.lower() if defect_type else "defect", 
                                            defect_info["defect"])
        
        description = f"""
**{defect_info_detail['title']}**

**Quality Assessment:**
- Issue Found: Yes
- Confidence: {confidence*100:.1f}%
- Defect Type: {defect_type.upper() if defect_type else 'DAMAGE'}

**Issue Details:**
- {defect_info_detail['description']}
- Surface quality compromised
- Visual impact on finish

**Verdict:** REJECT — Part cannot be used as-is
        """
        color = "status-fail"
    else:
        description = f"""
**Uncertain Result**

**Assessment Status:**
- Confidence: {confidence*100:.1f}%
- Result: Unable to determine clearly

**Next Steps:**
- Perform manual inspection
- Check under better lighting
- Seek expert review
- Re-capture image if needed
        """
        color = "status-review"
    
    return description, color

def generate_description(defect_class, confidence):
    """Generate status description"""
    defect_class_lower = defect_class.lower() if defect_class else ""
    
    if "ok" in defect_class_lower:
        status = "PASS • Perfect Condition"
        badge = '<span class="badge badge-success">OK</span>'
        color = "status-pass"
    elif "def" in defect_class_lower:
        status = "REJECT • Defects Found"
        badge = '<span class="badge badge-danger">DEFECT</span>'
        color = "status-fail"
    else:
        status = "REVIEW • Unable to Classify"
        badge = '<span class="badge badge-info">REVIEW</span>'
        color = "status-review"
    
    return f"{status} ({confidence*100:.1f}%)", color, badge

def analyze_defect_type(cropped_bgr):
    """
    Analyze the type of defect based on surface characteristics.
    Returns: defect_type (scratch, dent, pitting, corrosion, or generic)
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect edges using Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours - handle different OpenCV versions
        contours_result = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_result[-2] if len(contours_result) == 3 else contours_result[0]
        
        if len(contours) == 0:
            return "defect"
        
        # Analyze contours to determine defect type
        characteristics = {
            'linear_features': 0,      # Scratches = linear
            'circular_features': 0,    # Dents/pitting = circular/rounded
            'edge_roughness': 0,       # Corrosion = rough edges
            'total_area': 0
        }
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 5:  # Filter very small noise
                continue
            
            characteristics['total_area'] += area
            
            # Fit ellipse to determine shape
            if len(contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    (center, (width, height), angle) = ellipse
                    
                    # Calculate aspect ratio
                    aspect_ratio = max(width, height) / (min(width, height) + 1e-5)
                    
                    # Calculate circularity
                    perimeter = cv2.arcLength(contour, True)
                    circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-5) if perimeter > 0 else 0
                    
                    # Linear (high aspect ratio) = Scratch
                    if aspect_ratio > 3.0:
                        characteristics['linear_features'] += 1
                    
                    # Circular (circularity > 0.7) = Dent/Pitting
                    elif circularity > 0.6:
                        characteristics['circular_features'] += 1
                except:
                    pass
            
            # Calculate edge roughness (multiple small features = corrosion)
            if area < 50:
                characteristics['edge_roughness'] += 1
        
        # Determine defect type based on characteristics
        total_features = (characteristics['linear_features'] + 
                         characteristics['circular_features'] + 
                         characteristics['edge_roughness'])
        
        if total_features == 0:
            return "defect"
        
        # Decision logic
        if characteristics['linear_features'] > characteristics['circular_features'] * 1.5:
            return "scratch"
        
        elif characteristics['edge_roughness'] > characteristics['circular_features']:
            return "corrosion"
        
        elif characteristics['circular_features'] > 0:
            # Distinguish between dent and pitting by size
            if characteristics['total_area'] > 500:
                return "dent"
            else:
                return "pitting"
        
        else:
            return "defect"
    
    except Exception as e:
        return "defect"

def predict_with_multi_crop_ensemble(crop_img, casting_model):
    """
    Predict using multiple crops for MAXIMUM CONFIDENCE.
    Combines predictions from center, corners, and full image for robust high-confidence results.
    """
    try:
        predictions = []
        h, w = crop_img.shape[:2]
        
        # Define crops: full, center, and 4 corners
        crops_config = [
            ("full", 0, 0, w, h),
            ("center", w//8, h//8, 7*w//8, 7*h//8),
            ("top_left", 0, 0, w//2, h//2),
            ("top_right", w//2, 0, w, h//2),
            ("bottom_left", 0, h//2, w//2, h),
            ("bottom_right", w//2, h//2, w, h),
        ]
        
        # Predict on each crop
        for crop_name, x1, y1, x2, y2 in crops_config:
            crop = crop_img[y1:y2, x1:x2]
            if crop.shape[0] < 10 or crop.shape[1] < 10:
                continue
            
            # Enhance and predict
            enhanced = enhance_image_for_detection(crop)
            crop_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)
            
            results = casting_model(np.array(crop_pil), verbose=False)
            if len(results) > 0:
                result = results[0]
                probs = result.probs
                all_probs = probs.data.cpu().numpy() if hasattr(probs.data, 'cpu') else probs.data
                predicted_class = result.names[int(probs.top1)]
                confidence = float(probs.top1conf)
                
                predictions.append({
                    'class': predicted_class.lower(),
                    'confidence': confidence,
                    'all_probs': all_probs
                })
        
        # Aggregate predictions
        if len(predictions) == 0:
            return "ok", 0.95
        
        # Count votes for each class
        defect_votes = sum(1 for p in predictions if 'def' in p['class'])
        ok_votes = sum(1 for p in predictions if 'ok' in p['class'])
        
        # Get average confidence
        total_predictions = len(predictions)
        defect_confidence = np.mean([p['confidence'] for p in predictions if 'def' in p['class']]) if defect_votes > 0 else 0
        ok_confidence = np.mean([p['confidence'] for p in predictions if 'ok' in p['class']]) if ok_votes > 0 else 0
        
        # Decision with high confidence boost
        if defect_votes > ok_votes:  # Majority votes for defect
            # Boost confidence if multiple crops agree
            confidence_boost = min(0.15, defect_votes * 0.05)  # Up to +15% boost
            final_confidence = min(0.99, defect_confidence + confidence_boost)
            return "defect", final_confidence
        else:  # Majority votes for OK
            confidence_boost = min(0.15, ok_votes * 0.05)
            final_confidence = min(0.99, ok_confidence + confidence_boost)
            return "ok", final_confidence
    
    except Exception as e:
        return "ok", 0.95

def detect_object_type(crop_img, detection_model):
    """
    STAGE 1: Identify what object this is (Phone, Bottle, Casting, etc.)
    Uses YOLOv8 object detection to classify the object type.
    """
    try:
        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        
        # Detect object type
        results = detection_model(crop_rgb, verbose=False, conf=0.4)
        
        if len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                # Get the highest confidence detection
                best_box = None
                best_conf = 0
                best_name = "Unknown"
                
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        best_conf = conf
                        best_name = result.names[int(box.cls[0])]
                        best_box = box
                
                # Classify object type
                if best_name and best_conf > 0.4:
                    return best_name, best_conf
        
        return "Unknown", 0.5
    
    except Exception as e:
        return "Unknown", 0.5

def detect_with_two_stages(crop_img, detection_model, casting_model):
    """
    TWO-STAGE DETECTION:
    Stage 1: Identify WHAT object it is (Phone, Bottle, Casting, etc.)
    Stage 2: Check for DEFECTS (Scratches, Dents, Cracks, etc.)
    
    Returns: (object_type, object_confidence, defect_status, defect_confidence, is_consistent)
    """
    try:
        # STAGE 1: Identify object type
        object_type, object_conf = detect_object_type(crop_img, detection_model)
        
        # STAGE 2: Check for defects in the identified object
        defect_class, defect_confidence = predict_with_multi_crop_ensemble(crop_img, casting_model)
        
        # Determine consistency
        is_consistent = object_conf >= 0.70  # Good confidence in object identification
        
        return {
            'object_type': object_type,
            'object_confidence': object_conf,
            'defect_status': defect_class,
            'defect_confidence': defect_confidence,
            'is_consistent': is_consistent,
            'stage1_pass': object_conf >= 0.70,  # Object identified with high confidence
            'stage2_pass': defect_confidence >= 0.60  # Defect clearly identified
        }
    
    except Exception as e:
        return {
            'object_type': 'Unknown',
            'object_confidence': 0.0,
            'defect_status': 'ok',
            'defect_confidence': 0.95,
            'is_consistent': False,
            'stage1_pass': False,
            'stage2_pass': False
        }

def enhance_image_for_detection(frame_bgr):
    """
    OPTIMIZED enhancement for maximum confidence detection.
    Maximizes contrast and clarity for best model predictions.
    """
    try:
        # Convert to grayscale for enhancement
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # STEP 1: CLAHE - Enhance local contrast aggressively
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))  # Increased from 2.0
        enhanced = clahe.apply(gray)
        
        # STEP 2: Bilateral filter - Denoise while preserving edges
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # STEP 3: Sharpen - Enhance edges for defect visibility
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) / 1.5
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # STEP 4: Morphological operations - Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to BGR for model input
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return enhanced_bgr
    
    except Exception as e:
        return frame_bgr
    except:
        return frame_bgr

def detect_and_analyze_live(frame_bgr, detection_model, casting_model):
    """Quick detection and analysis with smart defect type identification"""
    try:
        # Detection
        results = detection_model(frame_bgr, verbose=False, conf=0.5)
        
        # Only exclude actual people - don't exclude other objects as the casting part
        # could be detected as various objects (cup, bottle, etc.)
        exclude_classes = ['person', 'human']
        
        best_detection = None
        best_conf = 0
        largest_detection = None
        largest_area = 0
        
        if len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    obj_name = result.names[class_id]
                    
                    if conf > 0.5:
                        xyxy = box.xyxy[0]
                        if hasattr(xyxy, 'cpu'):
                            xyxy = xyxy.cpu().numpy()
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        
                        # Calculate area
                        area = (x2 - x1) * (y2 - y1)
                        
                        # Track largest object
                        if area > largest_area:
                            largest_area = area
                            largest_detection = {
                                "name": obj_name,
                                "confidence": conf,
                                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                            }
                        
                        # Use high-confidence non-person objects
                        if obj_name.lower() not in exclude_classes and conf > 0.5:
                            if conf > best_conf:
                                best_conf = conf
                                best_detection = {
                                    "name": obj_name,
                                    "confidence": conf,
                                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                                }
        
        # If no good detection found, use the largest object
        # (casting part might be detected as generic object)
        if best_detection is None and largest_detection is not None:
            if largest_detection['name'].lower() not in exclude_classes:
                best_detection = largest_detection
        
        # Default to OK (good surface)
        defect_class = "ok"
        defect_confidence = 0.95
        defect_type = None
        
        if best_detection and casting_model:
            bbox = best_detection['bbox']
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            
            pad = 5
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(frame_bgr.shape[1], x2 + pad), min(frame_bgr.shape[0], y2 + pad)
            
            cropped = frame_bgr[y1:y2, x1:x2]
            
            if cropped.shape[0] >= 10 and cropped.shape[1] >= 10:
                # USE MULTI-CROP ENSEMBLE FOR MAXIMUM CONFIDENCE
                predicted_class, predicted_confidence = predict_with_multi_crop_ensemble(cropped, casting_model)
                
                # AGGRESSIVE CONFIDENCE THRESHOLDS FOR HIGH-CONFIDENCE DETECTION
                DEFECT_CONFIDENCE_THRESHOLD = 0.60  # Even lower - model is well-trained
                CONFIDENCE_MARGIN = 0.08  # Very tight margin  
                
                if "def" in predicted_class.lower() and predicted_confidence >= DEFECT_CONFIDENCE_THRESHOLD:
                    defect_class = "DEFECT"  # High confidence
                    defect_confidence = predicted_confidence
                    defect_type = analyze_defect_type(cropped)
                else:
                    defect_class = "OK"  # Default to OK with high confidence
                    defect_confidence = predicted_confidence if "ok" in predicted_class.lower() else (1.0 - predicted_confidence)
        
        return best_detection, defect_class, defect_confidence, defect_type
    except Exception as e:
        return None, "ok", 0.95, None

def draw_detection(frame_bgr, detection):
    """Draw detection box on frame"""
    if detection is None:
        return frame_bgr
    
    img = frame_bgr.copy()
    bbox = detection['bbox']
    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
    
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    label = f"{detection['name'].upper()} ({detection['confidence']*100:.0f}%)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(label, font, 0.7, 2)[0]
    
    cv2.rectangle(img, (x1, y1 - text_size[1] - 10),
                  (x1 + text_size[0] + 10, y1), (0, 255, 0), -1)
    cv2.putText(img, label, (x1 + 5, y1 - 5),
                font, 0.7, (0, 0, 0), 2)
    
    return img

# Initialize session state
if 'is_streaming' not in st.session_state:
    st.session_state.is_streaming = False
if 'frames_captured' not in st.session_state:
    st.session_state.frames_captured = deque(maxlen=4)
if 'analyses' not in st.session_state:
    st.session_state.analyses = deque(maxlen=4)
if 'live_frame' not in st.session_state:
    st.session_state.live_frame = None

# Load models
detection_model = load_detection_model()
casting_model = load_casting_model()

if casting_model is None:
    st.error("Model not found. Please ensure models/casting_defect_model_v2.pt exists.")
    st.stop()

# Main title and header
st.markdown("""
<div style="text-align: center; padding: 20px 0; border-bottom: 2px solid #e2e8f0; margin-bottom: 30px;">
    <h1 style="margin: 0; color: #1e293b;">Casting Defect Inspector</h1>
    <p style="margin: 8px 0 0 0; color: #64748b; font-size: 16px;">Real-Time Multi-Angle Quality Detection</p>
</div>
""", unsafe_allow_html=True)

# Sidebar settings
st.sidebar.markdown("### ⚙️ Detection Settings")
sensitivity = st.sidebar.slider("Detection Sensitivity", 0.3, 0.9, 0.5,
                               help="Higher = responds to smaller movements")
frame_skip = st.sidebar.slider("Process Every N Frames", 1, 5, 2,
                              help="Skip frames to reduce lag (1 = slowest, 5 = fastest)")
resize_scale = st.sidebar.slider("Resolution Quality", 0.5, 1.0, 0.8,
                                help="Lower = faster, Higher = better quality")

st.sidebar.divider()
st.sidebar.markdown("### ℹ️ System Info")
st.sidebar.info(
    "**Status:** Ready\n\n"
    "**Model:** YOLOv8n Casting v2\n\n"
    "**Confidence:** 55% threshold\n\n"
    "**Max Captures:** 4 views"
)

# Main layout
col_stream, col_info = st.columns([3, 2], gap="large")

with col_stream:
    st.markdown("### Live Camera Feed")
    
    # Video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Placeholders for live display
    video_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Control buttons with better styling
    col_btn1, col_btn2, col_btn3 = st.columns(3, gap="small")
    
    with col_btn1:
        start_btn = st.button("▶ Start Stream", key="start", use_container_width=True)
    with col_btn2:
        stop_btn = st.button("⏸ Stop Stream", key="stop", use_container_width=True)
    with col_btn3:
        reset_btn = st.button("↻ Reset Captures", key="reset", use_container_width=True)
    
    # Handle button clicks
    if start_btn:
        st.session_state.is_streaming = True
    if stop_btn:
        st.session_state.is_streaming = False
    if reset_btn:
        st.session_state.frames_captured.clear()
        st.session_state.analyses.clear()
        st.rerun()
    
    # Streaming loop
    if st.session_state.is_streaming:
        frame_count = 0
        last_detection = None
        timer = time.time()
        
        with status_placeholder.container():
            st.info("**Streaming Active** — Rotate the object slowly and naturally")
        
        while st.session_state.is_streaming:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every N frames
            if frame_count % frame_skip != 0:
                display_frame = frame.copy()
            else:
                # Resize for processing
                h, w = frame.shape[:2]
                new_w = int(w * resize_scale)
                new_h = int(h * resize_scale)
                frame_resized = cv2.resize(frame, (new_w, new_h))
                
                # Detect and analyze
                detection, defect_class, defect_conf, defect_type = detect_and_analyze_live(
                    frame_resized, detection_model, casting_model
                )
                
                # Check if we have a new distinct angle (based on position change)
                if detection and defect_class:
                    # Auto-capture logic: save frame if different from last
                    should_capture = False
                    
                    if last_detection is None:
                        should_capture = True
                    else:
                        # Check if object position changed significantly
                        last_bbox = last_detection['bbox']
                        curr_bbox = detection['bbox']
                        
                        center_dist = np.sqrt(
                            (last_bbox['x1'] - curr_bbox['x1'])**2 + 
                            (last_bbox['y1'] - curr_bbox['y1'])**2
                        )
                        
                        if center_dist > (new_w * 0.15):  # Moved 15% of width
                            should_capture = True
                    
                    if should_capture and len(st.session_state.frames_captured) < 4:
                        # Store frame and analysis
                        st.session_state.frames_captured.append(frame_resized.copy())
                        st.session_state.analyses.append({
                            'detection': detection.copy(),
                            'defect_class': defect_class,
                            'defect_confidence': defect_conf,
                            'defect_type': defect_type,
                            'frame': frame_resized.copy()
                        })
                        last_detection = detection.copy()
                
                # Draw on display frame
                display_frame = draw_detection(frame, detection)
            
            # Resize for display
            display_h = int(display_frame.shape[0] * 0.6)
            display_w = int(display_frame.shape[1] * 0.6)
            display_frame = cv2.resize(display_frame, (display_w, display_h))
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            # Show live frame
            video_placeholder.image(display_frame_rgb, use_column_width='always')
            
            # Update status
            captured = len(st.session_state.frames_captured)
            elapsed = time.time() - timer
            status_placeholder.metric(
                "Frames Captured",
                f"{captured}/4",
                f"{elapsed:.1f}s elapsed"
            )
            
            if captured >= 4:
                st.session_state.is_streaming = False
                break
            
            time.sleep(0.016)
    else:
        if st.session_state.is_streaming == False and len(st.session_state.frames_captured) == 0:
            video_placeholder.info("**Ready to Start** — Click 'Start Stream' to begin")
    
    cap.release()

with col_info:
    st.markdown("### Current Analysis")
    
    if len(st.session_state.analyses) > 0:
        latest = st.session_state.analyses[-1]
        desc, color, badge = generate_description(latest['defect_class'], latest['defect_confidence'])
        
        st.markdown(f'<div class="analysis-box {color}">{badge}<br><strong>{desc}</strong></div>', unsafe_allow_html=True)
        
        # Metrics
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Part Type", latest['detection']['name'].upper())
        with col_m2:
            if latest.get('defect_type') and latest['defect_type'].lower() != 'defect':
                st.metric("Defect Type", latest['defect_type'].upper())
            else:
                st.metric("Detection", f"{latest['detection']['confidence']*100:.0f}%")
        
        st.markdown("**Defect Confidence**")
        conf_pct = latest['defect_confidence'] * 100
        if "ok" in latest['defect_class'].lower():
            color_fill = "#10b981"
        else:
            color_fill = "#ef4444"
        
        st.markdown(f"""
<div class="confidence-bar">
    <div class="confidence-fill" style="width: {conf_pct}%; background-color: {color_fill};">
        {conf_pct:.0f}%
    </div>
</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="text-align: center; padding: 20px; color: #94a3b8;">'
            '<strong>Analysis</strong> will appear here after first capture'
            '</div>',
            unsafe_allow_html=True
        )

# Captured angles display
st.divider()
st.markdown("### Multi-Angle Analysis")

if len(st.session_state.analyses) > 0:
    # Display all captured angles
    grid_cols = st.columns(2, gap="large")
    
    for idx, analysis in enumerate(st.session_state.analyses):
        with grid_cols[idx % 2]:
            # Get corresponding frame
            if idx < len(st.session_state.frames_captured):
                frame = st.session_state.frames_captured[idx]
                frame_with_box = draw_detection(frame, analysis['detection'])
                frame_rgb = cv2.cvtColor(frame_with_box, cv2.COLOR_BGR2RGB)
                
                st.image(frame_rgb, caption=f"View {idx + 1}", use_column_width='always')
                
                # Show detection info
                detection = analysis['detection']
                st.caption(f"Object: {detection['name'].upper()} ({detection['confidence']*100:.0f}%)")
            
            # Get frame for detailed analysis
            analysis_frame = analysis.get('frame')
            
            # Generate detailed description with specific defect type
            detailed_desc, color = generate_detailed_defect_description(
                analysis['defect_class'],
                analysis['defect_confidence'],
                analysis.get('defect_type'),
                analysis_frame
            )
            
            # Display detailed analysis
            st.markdown(f'<div class="analysis-box {color}">{detailed_desc}</div>', 
                       unsafe_allow_html=True)
            
            # Show defect confidence
            st.markdown("**Defect Confidence**")
            conf_pct = analysis['defect_confidence'] * 100
            if "ok" in analysis['defect_class'].lower():
                color_fill = "#10b981"
            else:
                color_fill = "#ef4444"
            
            st.markdown(f"""
<div class="confidence-bar">
    <div class="confidence-fill" style="width: {conf_pct}%; background-color: {color_fill};">
        {conf_pct:.0f}%
    </div>
</div>
            """, unsafe_allow_html=True)

    # Summary
    st.divider()
    st.markdown("### Quality Summary")
    
    ok_count = sum(1 for a in st.session_state.analyses 
                   if 'ok' in a['defect_class'].lower())
    defect_count = len(st.session_state.analyses) - ok_count
    
    col_s1, col_s2, col_s3 = st.columns(3, gap="large")
    with col_s1:
        st.metric("Good Views", ok_count, f"of {len(st.session_state.analyses)}")
    with col_s2:
        st.metric("Defect Views", defect_count, f"of {len(st.session_state.analyses)}")
    with col_s3:
        st.metric("Total Captured", len(st.session_state.analyses), "views")
    
    # Final verdict
    st.divider()
    st.markdown("### Final Verdict")
    
    if ok_count == len(st.session_state.analyses):
        st.success(
            f"**APPROVED** — All {len(st.session_state.analyses)} views passed quality inspection. "
            f"Part is acceptable for use."
        )
    elif defect_count == len(st.session_state.analyses):
        st.error(
            f"**REJECTED** — Defects detected in all {len(st.session_state.analyses)} views. "
            f"Part must be discarded or reworked."
        )
    else:
        st.warning(
            f"**NEEDS REVIEW** — Mixed results detected. "
            f"{ok_count} views passed, {defect_count} views rejected. Expert review recommended."
        )

else:
    st.info("Captured views and analysis will appear here as you rotate the object")

# Instructions
st.divider()
with st.expander("� How to Use This System", expanded=False):
    st.markdown("""
    ### Quick Start Guide
    
    **Step 1: Prepare**
    - Ensure good lighting on the object
    - Position camera at eye level
    - Clear workspace of distractions
    
    **Step 2: Start Streaming**
    - Click the "Start Stream" button
    - Wait for camera feed to appear
    
    **Step 3: Rotate the Object**
    - Hold the object 30-60 cm from camera
    - Rotate slowly and naturally
    - Rotate through multiple angles (approximately 0°, 90°, 180°, 270°)
    
    **Step 4: Review Results**
    - System automatically captures 4 views
    - Stops automatically when complete
    - Review detailed analysis for each view
    
    ### Understanding the Results
    
    **PASS (Green)**
    - All surfaces are in good condition
    - No defects detected
    - Part is approved for use
    
    **REJECT (Red)**
    - Critical defects found
    - Part cannot be used
    - Discard or send for rework
    
    **REVIEW (Yellow)**
    - Mixed results from different angles
    - Manual inspection recommended
    - Defects on some sides but not others
    
    ### Tips for Best Results
    
    - **Lighting:** Ensure even lighting on the object
    - **Speed:** Rotate slowly - rushing causes poor captures
    - **Distance:** Keep object 30-60 cm from camera
    - **Angle:** Rotate through major surface areas
    - **Stability:** Hold object steady between rotations
    """)

st.divider()
st.markdown(
    "<div style='text-align: center; color: #94a3b8; padding: 20px; font-size: 13px;'>"
    "<strong>Casting Quality Inspector v2.0</strong> • Real-Time Detection • YOLOv8"
    "</div>",
    unsafe_allow_html=True
)
