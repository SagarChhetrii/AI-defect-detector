"""
Real-time Casting Defect Detection using Webcam
This script runs inference on webcam feed using trained YOLOv8 model.
"""

import cv2
import os
from pathlib import Path
from ultralytics import YOLO
import numpy as np

def main():
    """Main webcam inference function"""
    
    # Define model path
    model_path = Path("models/casting_defect_model.pt")
    
    # Check if model exists
    if not model_path.exists():
        print("❌ Error: Trained model not found!")
        print(f"   Expected path: {model_path}")
        print("\n📝 Please train the model first by running:")
        print("   python train.py")
        return
    
    print("=" * 60)
    print("🎥 Real-time Casting Defect Detection")
    print("=" * 60)
    
    # Load trained model
    print(f"\n📦 Loading trained model from: {model_path}")
    model = YOLO(str(model_path))
    
    # Define class names (must match your training dataset folder names)
    class_names = {
        0: "OK",       # ok_front folder
        1: "DEFECT"    # def_front folder
    }
    
    # Note: The exact class names depend on folder order during training
    # If your results are inverted, swap '0' and '1' above
    
    print("✅ Model loaded successfully!")
    print("\n📸 Opening webcam...")
    
    # Open webcam (0 = default webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam!")
        return
    
    print("✅ Webcam opened!")
    print("\n🎮 Controls:")
    print("   - Press 'q' to quit")
    print("   - Press 's' to save screenshot")
    print("=" * 60)
    
    # Webcam settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    detection_count = 0
    defect_count = 0
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Error: Failed to read frame from webcam!")
            break
        
        frame_count += 1
        
        # Run inference on the frame
        # conf=0.5 means only predictions with >50% confidence are shown
        results = model(frame, conf=0.5, verbose=False)
        
        # Process results
        if results and len(results) > 0:
            result = results[0]
            
            # Get predictions
            if result.probs is not None:
                # Get class with highest probability
                pred_class = int(result.probs.top1)
                confidence = float(result.probs.top1conf)
                class_name = class_names.get(pred_class, "Unknown")
                
                detection_count += 1
                if class_name == "DEFECT":
                    defect_count += 1
                
                # Draw prediction on frame
                # Determine color based on class (Red for DEFECT, Green for OK)
                color = (0, 0, 255) if class_name == "DEFECT" else (0, 255, 0)
                
                # Draw text on frame
                text = f"{class_name}: {confidence:.2f}"
                
                # Draw semi-transparent background for text readability
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                font_thickness = 2
                
                # Get text size to draw background
                (text_width, text_height) = cv2.getTextSize(
                    text, font, font_scale, font_thickness
                )[0]
                
                # Draw background rectangle
                cv2.rectangle(
                    frame,
                    (10, 30),
                    (10 + text_width + 10, 30 + text_height + 10),
                    color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    frame,
                    text,
                    (20, 50),
                    font,
                    font_scale,
                    (255, 255, 255),
                    font_thickness
                )
                
                # Draw border
                cv2.rectangle(
                    frame,
                    (10, 30),
                    (10 + text_width + 10, 30 + text_height + 10),
                    color,
                    2
                )
        
        # Display statistics on frame
        stats_text = f"Frames: {frame_count} | Detected: {detection_count} | Defects: {defect_count}"
        cv2.putText(
            frame,
            stats_text,
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Display frame
        cv2.imshow("Casting Defect Detection - Press 'q' to quit", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n❌ Exiting...")
            break
        elif key == ord('s'):
            # Save screenshot
            screenshot_path = f"screenshot_{frame_count}.jpg"
            cv2.imwrite(screenshot_path, frame)
            print(f"📸 Screenshot saved: {screenshot_path}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    print("\n" + "=" * 60)
    print("📊 Final Statistics:")
    print(f"   - Total frames: {frame_count}")
    print(f"   - Total detections: {detection_count}")
    print(f"   - Defective items: {defect_count}")
    if detection_count > 0:
        defect_rate = (defect_count / detection_count) * 100
        print(f"   - Defect rate: {defect_rate:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
