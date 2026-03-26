"""
Live Continuous Object Detection with Webcam
Real-time detection with bounding boxes and coordinates
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
from datetime import datetime


def load_models():
    """Load detection and defect models"""
    detect_model = YOLO("yolov8n.pt")
    
    casting_model = None
    model_path = Path("models/casting_defect_model.pt")
    if model_path.exists():
        casting_model = YOLO(str(model_path))
    
    return detect_model, casting_model


def detect_and_draw(frame, detect_model, casting_model):
    """Detect objects and draw boxes on frame"""
    
    # Run detection with confidence threshold
    results = detect_model(frame, verbose=False, conf=0.5)
    
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
    
    return frame


def main():
    """Main live detection loop"""
    print("=" * 60)
    print("🎥 LIVE CONTINUOUS OBJECT DETECTION")
    print("=" * 60)
    print("\nLoading models...")
    
    detect_model, casting_model = load_models()
    
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n✅ Webcam opened successfully!")
    print("\nControls:")
    print("  SPACE - Capture screenshot")
    print("  Q     - Quit")
    print("=" * 60)
    print("\n🔴 Live detection running...\n")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect and draw
            frame_with_detection = detect_and_draw(frame, detect_model, casting_model)
            
            # Add FPS counter
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processing frame: {frame_count}", end='\r')
            
            # Add instructions on frame
            cv2.putText(frame_with_detection, "SPACE=Screenshot | Q=Quit",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display
            cv2.imshow("Live Object Detection | Confidence >= 50%", frame_with_detection)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n\n✅ Stopping detection...")
                break
            elif key == ord(' '):
                # Save screenshot
                filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame_with_detection)
                print(f"✅ Screenshot saved: {filename}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Detection interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Webcam closed")


if __name__ == "__main__":
    main()
