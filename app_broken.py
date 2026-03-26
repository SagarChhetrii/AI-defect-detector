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
    """Load the casting defect classification model"""
    model_path = Path("models/casting_defect_model.pt")
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

def detect_object(image_array):
    """Detect objects in image (bottle, can, casting, etc.) with bounding boxes"""
    try:
        # Handle PIL Image or numpy array
        if isinstance(image_array, Image.Image):
            image_array = np.array(image_array)
        
        # Ensure image is in correct format
        if len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        elif image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
        
        detect_model = load_detection_model()
        results = detect_model(image_array, verbose=False)
        
        detected_objects = []
        if len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    if confidence > 0.3:  # Lowered threshold
                        object_name = result.names[class_id]
                        xyxy = box.xyxy[0]
                        if hasattr(xyxy, 'cpu'):
                            xyxy = xyxy.cpu().numpy()
                        else:
                            xyxy = xyxy.numpy() if hasattr(xyxy, 'numpy') else xyxy
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        detected_objects.append({
                            "name": object_name,
                            "confidence": confidence,
                            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                        })
        
        if detected_objects:
            return detected_objects[0]
        return None
    except Exception as e:
        return None

def draw_detections(image, detection):
    """Draw bounding boxes with corner circles like professional tracking"""
    if detection is None:
        return image
    
    try:
        # Handle PIL Image - convert to numpy BGR for OpenCV
        if isinstance(image, Image.Image):
            img_copy = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            # Already numpy array
            img_copy = image.copy()
        
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        confidence = detection['confidence']
        name = detection['name']
        width = x2 - x1
        height = y2 - y1
        
        # Draw thick bounding box (bright green)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 4)
        
        # Draw GREEN CIRCLES at all 4 corners (professional tracking style)
        circle_radius = 12
        circle_color = (0, 255, 0)  # Bright green
        circle_thickness = -1  # Filled circles
        
        # Top-left corner
        cv2.circle(img_copy, (x1, y1), circle_radius, circle_color, circle_thickness)
        cv2.circle(img_copy, (x1, y1), circle_radius - 4, (0, 0, 0), 2)  # Black outline
        
        # Top-right corner
        cv2.circle(img_copy, (x2, y1), circle_radius, circle_color, circle_thickness)
        cv2.circle(img_copy, (x2, y1), circle_radius - 4, (0, 0, 0), 2)
        
        # Bottom-left corner
        cv2.circle(img_copy, (x1, y2), circle_radius, circle_color, circle_thickness)
        cv2.circle(img_copy, (x1, y2), circle_radius - 4, (0, 0, 0), 2)
        
        # Bottom-right corner
        cv2.circle(img_copy, (x2, y2), circle_radius, circle_color, circle_thickness)
        cv2.circle(img_copy, (x2, y2), circle_radius - 4, (0, 0, 0), 2)
        
        # Draw center circle as well
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(img_copy, (center_x, center_y), circle_radius - 2, (0, 255, 0), circle_thickness)
        cv2.circle(img_copy, (center_x, center_y), circle_radius - 6, (0, 0, 0), 2)
        
        # Draw label with background at top
        label = f"{name.upper()} | Conf: {confidence*100:.0f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        
        # Green background for label
        cv2.rectangle(img_copy, 
                      (x1 + 5, y1 - text_size[1] - 15),
                      (x1 + text_size[0] + 15, y1 - 5),
                      (0, 255, 0), -1)
        
        # Black text
        cv2.putText(img_copy, label, (x1 + 10, y1 - 8),
                    font, font_scale, (0, 0, 0), thickness)
        
        # Draw coordinate boxes at corners
        font_small = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_small = 0.6
        text_color = (0, 255, 0)
        
        # Top-left coordinates
        coord_tl = f"({x1},{y1})"
        cv2.putText(img_copy, coord_tl, (x1 - 80, y1 - 15),
                    font_small, font_scale_small, text_color, 2)
        
        # Top-right coordinates
        coord_tr = f"({x2},{y1})"
        cv2.putText(img_copy, coord_tr, (x2 + 15, y1 - 15),
                    font_small, font_scale_small, text_color, 2)
        
        # Bottom-left coordinates
        coord_bl = f"({x1},{y2})"
        cv2.putText(img_copy, coord_bl, (x1 - 80, y2 + 25),
                    font_small, font_scale_small, text_color, 2)
        
        # Bottom-right coordinates
        coord_br = f"({x2},{y2})"
        cv2.putText(img_copy, coord_br, (x2 + 15, y2 + 25),
                    font_small, font_scale_small, text_color, 2)
        
        # Draw dimensions (width x height) with background
        dim_text = f"W:{width}px | H:{height}px"
        dim_size = cv2.getTextSize(dim_text, font_small, font_scale_small, 2)[0]
        
        # Semi-transparent background for center text
        cv2.rectangle(img_copy,
                      (center_x - dim_size[0]//2 - 8, center_y - 15),
                      (center_x + dim_size[0]//2 + 8, center_y + 10),
                      (0, 0, 0), -1)
        
        cv2.putText(img_copy, dim_text, (center_x - dim_size[0]//2, center_y + 5),
                    font_small, font_scale_small, (0, 255, 255), 2)
        
        return img_copy
    except Exception as e:
        # If drawing fails, return original image
        return image

def analyze_casting_defect(image, casting_model):
    """Analyze casting for defects"""
    if casting_model is None:
        return None, None
    img_array = np.array(image)
    results = casting_model(img_array, verbose=False)
    if len(results) > 0:
        result = results[0]
        probs = result.probs
        class_id = int(probs.top1)
        confidence = float(probs.top1conf)
        class_name = result.names[class_id]
        return class_name, confidence
    return None, None

def get_defect_description(pred_class, confidence, object_type=None):
    """Generate defect description based on prediction and object type"""
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
    st.subheader("🎥 Live Webcam with Real-time Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Live Camera Feed**")
        
        # Webcam HTML with proper controls
        camera_html = """
        <html>
        <head>
        <meta charset="UTF-8">
        <style>
            body { margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto; }
            #camera-container { position: relative; background: #000; border: 3px solid #1f77b4; border-radius: 8px; overflow: hidden; }
            #video { width: 100%; height: auto; display: block; }
            #canvas { display: none; }
            .controls { display: flex; gap: 10px; margin: 15px 0; flex-wrap: wrap; }
            button { flex: 1; min-width: 100px; padding: 12px; font-size: 13px; font-weight: bold; border: none; border-radius: 6px; cursor: pointer; background: #1f77b4; color: white; transition: all 0.2s; }
            button:hover:not(:disabled) { background: #165a8f; transform: translateY(-2px); }
            button:disabled { background: #ccc; cursor: not-allowed; opacity: 0.6; }
            #status { padding: 10px; margin: 10px 0; border-radius: 6px; font-weight: bold; background: #f5f5f5; }
            #status.live { background: #e8f5e9; color: #2ca02c; }
            .indicator { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }
            .live { background: #2ca02c; animation: pulse 1s infinite; }
            @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
            #error { color: #d62728; padding: 10px; background: #ffebee; border-radius: 4px; margin: 10px 0; }
        </style>
        </head>
        <body>
        <div id="camera-container">
            <video id="video" autoplay playsinline muted style="width: 100%;"></video>
        </div>
        <canvas id="canvas"></canvas>
        
        <div class="controls">
            <button id="startBtn" onclick="startCamera()">▶️ START CAMERA</button>
            <button id="stopBtn" onclick="stopCamera()" disabled>⏹️ STOP CAMERA</button>
            <button id="captureBtn" onclick="captureFrame()" disabled>📸 CAPTURE & ANALYZE</button>
        </div>
        
        <div id="status">
            <span class="indicator"></span>Camera ready
        </div>
        <div id="error"></div>
        
        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const captureBtn = document.getElementById('captureBtn');
            const status = document.getElementById('status');
            const errorDiv = document.getElementById('error');
            let stream = null;
            let isActive = false;
            
            async function startCamera() {
                try {
                    stream = await navigator.mediaDevices.getUserMedia({
                        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
                    });
                    video.srcObject = stream;
                    video.onloadedmetadata = () => {
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        isActive = true;
                        startBtn.disabled = true;
                        stopBtn.disabled = false;
                        captureBtn.disabled = false;
                        status.className = 'live';
                        status.innerHTML = '<span class="indicator live"></span>🔴 CAMERA ACTIVE';
                        errorDiv.innerHTML = '';
                    };
                } catch (err) {
                    errorDiv.innerHTML = '❌ Camera access denied. Check browser permissions.';
                    status.innerHTML = '<span class="indicator"></span>❌ Camera not available';
                }
            }
            
            function stopCamera() {
                if (stream) {
                    stream.getTracks().forEach(t => t.stop());
                    stream = null;
                    isActive = false;
                }
                startBtn.disabled = false;
                stopBtn.disabled = true;
                captureBtn.disabled = true;
                status.className = '';
                status.innerHTML = '<span class="indicator"></span>Camera stopped';
                video.srcObject = null;
            }
            
            function captureFrame() {
                if (!isActive || !video.srcObject) return;
                
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                canvas.toBlob(blob => {
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        window.parent.postMessage({
                            type: 'webcam_frame_captured',
                            image: e.target.result,
                            timestamp: new Date().toISOString()
                        }, '*');
                        status.innerHTML = '<span class="indicator"></span>⏳ Processing frame...';
                    };
                    reader.readAsDataURL(blob);
                }, 'image/jpeg', 0.85);
            }
        </script>
        </body>
        </html>
        """
        
        st.components.v1.html(camera_html, height=520)
    
    with col2:
        st.markdown("**📊 Live Results**")
        st.markdown("---")
        
        # Result area
        result_placeholder = st.empty()
        with result_placeholder.container():
            st.info("👈 Capture a frame to see analysis")
        
        st.markdown("---")
        st.markdown("**⚡ Actions**")
        
        col_ok, col_def = st.columns(2)
        with col_ok:
            if st.button("✅ Log OK", width="stretch"):
                stats = load_stats()
                stats["history"].append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "class": "ok_front",
                    "object": "Camera Capture",
                    "confidence": 1.0
                })
                stats["ok_count"] += 1
                save_stats(stats)
                st.success("✅ Logged!")
                st.rerun()
        
        with col_def:
            if st.button("❌ Log Defect", width="stretch"):
                stats = load_stats()
                stats["history"].append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "class": "def_front",
                    "object": "Camera Capture",
                    "confidence": 1.0
                })
                stats["defect_count"] += 1
                save_stats(stats)
                st.success("❌ Logged!")
                st.rerun()

# ==================== TAB 2: UPLOAD & ANALYZE ====================
with tab2:
    st.subheader("� Upload Image & Analyze for Defects")
    
    col1, col2 = st.columns([1.8, 1.2])
    
    with col1:
        st.markdown("**Upload or capture an image:**")
        
        uploaded_file = st.file_uploader(
            "Choose image or capture from camera",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            key="image_upload",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            
            # Convert to numpy for processing -  keep as RGB
            image_array = np.array(image)
            
            # Detect object type
            st.markdown("---")
            with st.spinner("🔍 Identifying object type..."):
                detected = detect_object(image_array)
            
            # Draw bounding box on image
            if detected:
                # Draw detections returns BGR, need to convert back to RGB for Streamlit
                image_with_bbox = draw_detections(image_array, detected)
                # Convert BGR back to RGB for proper display
                if image_with_bbox is not None:
                    image_with_bbox_rgb = cv2.cvtColor(image_with_bbox, cv2.COLOR_BGR2RGB) if image_with_bbox.shape[2] == 3 else image_with_bbox
                    st.image(image_with_bbox_rgb, caption="📸 Detected Object with Bounding Box", use_column_width=True)
                else:
                    st.image(image, caption="📸 Captured Image - Detection failed", use_column_width=True)
            else:
                st.image(image, caption="📸 Uploaded Image", use_column_width=True)
                st.info("⚠️ No objects detected. Try uploading an image with clearer objects.")
            
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
            else:
                object_type = "Unknown Object"
                st.info("Object type: Generic Item")
            
            # Analyze for defects
            st.markdown("---")
            with st.spinner("🤖 Analyzing for defects..."):
                pred_class, confidence = analyze_casting_defect(image, casting_model)
            
            if pred_class and confidence:
                desc = get_defect_description(pred_class, confidence, object_type)
                
                # Display result
                st.markdown(f"""
                <div class="result-box result-{'ok' if 'PASS' in desc['status'] else 'defect'}">
                    <div style="font-size: 20px; font-weight: bold; color: {desc['color']};">
                        {desc['status']}
                    </div>
                    <div style="margin-top: 12px; font-size: 14px;">
                        {desc['message']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show affected areas if defect
                if desc['areas']:
                    with st.expander("🔴 Affected Areas & Issues"):
                        for area in desc['areas']:
                            st.write(area)
    
    with col2:
        st.markdown("**📋 Result Summary**")
        st.markdown("---")
        
        if uploaded_file is not None and 'desc' in locals():
            st.markdown(f"""
            <div style="background-color: #f5f5f5; padding: 15px; border-radius: 8px;">
            <strong>Object Type:</strong><br/>
            {object_type.title()}<br/><br/>
            <strong>Status:</strong><br/>
            {desc['status']}<br/><br/>
            <strong>Action:</strong><br/>
            <span style="color: {desc['color']}; font-weight: bold;">
            {desc['action']}
            </span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("**⚡ Quick Actions:**")
            
            col_ok, col_def = st.columns(2)
            with col_ok:
                if st.button("✅ Log as OK", width="stretch"):
                    stats = load_stats()
                    stats["history"].append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "class": "ok_front",
                        "object": object_type,
                        "confidence": 1.0
                    })
                    stats["ok_count"] += 1
                    save_stats(stats)
                    st.success("✅ Logged!")
            
            with col_def:
                if st.button("❌ Log as Defect", width="stretch"):
                    stats = load_stats()
                    stats["history"].append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "class": "def_front",
                        "object": object_type,
                        "confidence": confidence if confidence else 1.0
                    })
                    stats["defect_count"] += 1
                    save_stats(stats)
                    st.success("❌ Logged!")
        else:
            st.info("👈 Upload an image to see analysis results here")

# ==================== TAB 3: ANALYTICS ====================
with tab3:
    stats = load_stats()
    total = stats["ok_count"] + stats["defect_count"]
    defect_rate = (stats["defect_count"] / total * 100) if total > 0 else 0
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Items</div>
            <div class="metric-value">{total}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card ok-card">
            <div class="metric-label">✅ OK</div>
            <div class="metric-value" style="color: #2ca02c;">{stats['ok_count']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card defect-card">
            <div class="metric-label">❌ Defects</div>
            <div class="metric-value" style="color: #d62728;">{stats['defect_count']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Defect Rate</div>
            <div class="metric-value" style="color: #ff7f0e;">{defect_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("📋 Inspection History")
    
    if stats["history"]:
        history_data = []
        for record in reversed(stats["history"]):
            history_data.append({
                "Time": record["timestamp"],
                "Status": "✅ OK" if "ok" in record["class"].lower() else "❌ DEFECT",
                "Object": record.get("object", "Unknown"),
                "Confidence": f"{record.get('confidence', 1.0)*100:.1f}%"
            })
        st.dataframe(history_data, use_container_width=True, hide_index=True)
    else:
        st.info("No inspection history yet. Start analyzing images!")

# ==================== TAB 4: SETTINGS ====================
with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Data Management")
        stats = load_stats()
        st.write(f"**Total Inspections:** {len(stats['history'])}")
        st.write(f"**OK Count:** {stats['ok_count']}")
        st.write(f"**Defect Count:** {stats['defect_count']}")
        
        if st.button("🔄 Reset All Data"):
            save_stats({"ok_count": 0, "defect_count": 0, "history": []})
            st.success("✅ Data has been reset")
            st.rerun()
    
    with col2:
        st.subheader("ℹ️ System Info")
        st.write("**Detection Model:** YOLOv8 Nano (General)")
        st.write("**Classification Model:** YOLOv8 Nano (Casting)")
        st.write("**Classes Detected:** 80+ object types")
        st.write("**Defect Classes:** OK / DEFECT")
        
        model_path = Path("models/casting_defect_model.pt")
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024*1024)
            st.write(f"**Model Size:** {size_mb:.1f} MB ✅")
    
    st.markdown("---")
    st.subheader("📖 How It Works")
    st.markdown("""
    ### **Step 1: Object Detection**
    When you upload an image, the system first identifies **what object** is in the image:
    - Bottles, cans, castings, parts, products, etc.
    - Uses general YOLO model trained on 80+ object classes
    
    ### **Step 2: Defect Analysis**  
    Then it analyzes the detected object for defects:
    - Checks for surface irregularities
    - Detects cracks, breaks, and damage
    - Identifies dimensional errors
    - Reports material defects
    
    ### **Step 3: Report Generation**
    You get a detailed report showing:
    - ✅ **PASS** - Item ready for use
    - ❌ **DEFECT** - Item needs rework or disposal
    - Specific problem areas identified
    - Confidence percentage
    
    ### **Objects Detectable**
    Bottles • Cans • Castings • Screws • Bolts • Cups • Plates • Glass • Metal parts • And 70+ more!
    
    ### **Quick Tips**
    - Use good lighting for best results
    - Position object clearly in center of frame
    - Ensure object takes up ~50-80% of image
    - Log results immediately for accurate history
    """)
