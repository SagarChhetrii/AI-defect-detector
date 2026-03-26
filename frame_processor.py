"""
Simple Flask server for real-time frame processing
Runs alongside Streamlit to handle live defect detection
"""
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from pathlib import Path
import json
from PIL import Image
import io
import threading

app = Flask(__name__)

# Load model
MODEL_PATH = Path("models/casting_defect_model.pt")
model = None

if MODEL_PATH.exists():
    model = YOLO(str(MODEL_PATH))

def get_defect_description(pred_class, confidence):
    """Generate detailed description of detected defect"""
    is_ok = "ok" in pred_class.lower()
    
    if is_ok:
        return {
            "status": "✅ PASS",
            "color": "#2ca02c",
            "description": "No defects detected. Casting passes quality inspection.",
            "areas": None,
            "icon": "✅"
        }
    else:
        return {
            "status": "❌ DEFECT",
            "color": "#d62728",
            "description": "Defects detected in casting. Surface irregularities and material inconsistencies found.",
            "areas": ["Surface texture", "Material consistency", "Dimensional accuracy"],
            "icon": "⚠️"
        }

def annotate_frame(image_array, pred_class, confidence):
    """Add visual annotations to the frame"""
    h, w = image_array.shape[:2]
    
    # Color based on prediction
    is_ok = "ok" in pred_class.lower()
    color = (46, 160, 67) if is_ok else (214, 39, 40)  # BGR: green or red
    thickness = 3
    
    # Draw border
    cv2.rectangle(image_array, (0, 0), (w-1, h-1), color, thickness)
    
    # Add status text at top
    status = "✓ PASS" if is_ok else "✗ DEFECT"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image_array, status, (20, 40), font, 1.2, color, 2)
    
    # Add confidence at bottom
    conf_text = f"Confidence: {confidence*100:.1f}%"
    text_size = cv2.getTextSize(conf_text, font, 0.8, 2)[0]
    cv2.putText(image_array, conf_text, (w - text_size[0] - 20, h - 20), 
                font, 0.8, color, 2)
    
    # If defect, add highlighting zones
    if not is_ok:
        # Add semi-transparent red overlay for defect areas
        overlay = image_array.copy()
        cv2.rectangle(overlay, (40, 60), (w-40, h-80), (214, 39, 40), -1)
        cv2.addWeighted(overlay, 0.15, image_array, 0.85, 0, image_array)
    
    return image_array

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process a single frame from webcam"""
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Read frame from request
        frame_file = request.files['frame']
        frame_data = frame_file.read()
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(frame_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run inference
        results = model(image)
        
        pred_class = None
        confidence = 0
        
        if len(results) > 0:
            result = results[0]
            probs = result.probs
            class_id = int(probs.top1)
            confidence = float(probs.top1conf)
            pred_class = result.names[class_id]
        
        # Get description
        desc = get_defect_description(pred_class, confidence)
        
        # Annotate frame
        annotated = annotate_frame(image.copy(), pred_class, confidence)
        
        # Convert back to base64
        _, buffer = cv2.imencode('.jpg', annotated)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'frame': f'data:image/jpeg;base64,{frame_b64}',
            'prediction': pred_class,
            'confidence': round(float(confidence), 4),
            'description': desc['description'],
            'areas': desc['areas'],
            'color': desc['color'],
            'icon': desc['icon']
        })
    
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

if __name__ == '__main__':
    print("🔧 Frame processor server starting on http://localhost:5000")
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
