"""
Live Video Stream Processor
Real-time defect detection with continuous webcam feed
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import time

def load_models():
    """Load detection and classification models"""
    print("📦 Loading models...")
    
    # Detection model
    detect_model = YOLO("yolov8n.pt")
    
    # Defect classification model (v2)
    casting_model_path = Path("models/casting_defect_model_v2.pt")
    if not casting_model_path.exists():
        print("⚠️ v2 model not found, using v1")
        casting_model_path = Path("models/casting_defect_model.pt")
    
    casting_model = YOLO(str(casting_model_path))
    print("✅ Models loaded!")
    
    return detect_model, casting_model

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
        
        # Add padding
        pad = 5
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(image_bgr.shape[1], x2 + pad)
        y2 = min(image_bgr.shape[0], y2 + pad)
        
        cropped = image_bgr[y1:y2, x1:x2]
        
        if cropped.shape[0] < 10 or cropped.shape[1] < 10:
            return None, None
        
        # Analyze
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

def main():
    """Main live processing loop"""
    print("\n" + "="*70)
    print("🎥 LIVE VIDEO STREAM PROCESSING")
    print("="*70)
    
    # Load models
    detect_model, casting_model = load_models()
    
    # Open webcam
    print("\n📷 Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    print("✅ Webcam opened!")
    print("\n🎯 Controls:")
    print("  Press 'q' to quit")
    print("  Press 's' to save frame")
    print("  Press 'p' to pause/resume")
    print("\n" + "="*70 + "\n")
    
    frame_count = 0
    paused = False
    fps_counter = 0
    fps_time = time.time()
    current_fps = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error reading frame")
                break
            
            frame_count += 1
            fps_counter += 1
            
            # Resize for faster processing
            frame_resized = cv2.resize(frame, (640, 480))
            
            # Detect
            detection = detect_object(frame_resized, detect_model)
            
            # Analyze defect if detected
            defect_status = None
            defect_conf = None
            if detection:
                defect_status, defect_conf = analyze_defect(frame_resized, detection['bbox'], casting_model)
            
            # Draw detection
            frame_with_detection = draw_detection(frame_resized, detection)
            
            # Calculate FPS
            if time.time() - fps_time > 1:
                current_fps = fps_counter
                fps_counter = 0
                fps_time = time.time()
            
            # Add info overlay
            info_y = 30
            cv2.putText(frame_with_detection, f"Frame: {frame_count}", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_with_detection, f"FPS: {current_fps}", (10, info_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show detection status
            if detection:
                status_text = f"✓ Detected: {detection['name'].upper()}"
                cv2.putText(frame_with_detection, status_text, (10, info_y + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show defect status
                if defect_status:
                    is_ok = "ok" in defect_status.lower()
                    if is_ok:
                        defect_text = f"✅ OK - No Defects ({defect_conf*100:.0f}%)"
                        color = (0, 255, 0)  # Green
                    else:
                        defect_text = f"❌ DEFECT - {defect_status} ({defect_conf*100:.0f}%)"
                        color = (0, 0, 255)  # Red
                    
                    cv2.putText(frame_with_detection, defect_text, (10, info_y + 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                cv2.putText(frame_with_detection, "❌ No Objects Detected", (10, info_y + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display
            cv2.imshow("🎥 Live Defect Detection Stream", frame_with_detection)
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n🛑 Stopping stream...")
            break
        elif key == ord('s'):
            filename = f"frame_{frame_count}.jpg"
            cv2.imwrite(filename, frame_resized)
            print(f"💾 Saved: {filename}")
        elif key == ord('p'):
            paused = not paused
            status = "⏸️ PAUSED" if paused else "▶️ RUNNING"
            print(f"{status}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print(f"✅ Stream closed. Processed {frame_count} frames")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
