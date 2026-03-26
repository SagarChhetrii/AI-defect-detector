"""
Improved YOLOv8 Classification Model Training
Using aggressive augmentation + more epochs + better hyperparameters
For small datasets (7K images)
"""

import os
from pathlib import Path
from ultralytics import YOLO

def main():
    """Main improved training function"""
    
    # Define paths
    dataset_path = Path("casting_data")
    model_save_path = Path("models")
    
    # Create models directory if it doesn't exist
    model_save_path.mkdir(exist_ok=True)
    
    # Check if dataset exists
    if not dataset_path.exists():
        print(f"❌ Error: Dataset path '{dataset_path}' not found!")
        print("Please ensure casting_data folder is in the correct location.")
        return
    
    print("=" * 70)
    print("🚀 IMPROVED YOLOv8 Classification Model Training (v2.0)")
    print("=" * 70)
    
    # Load pretrained YOLOv8 nano classification model
    # Using 'n' for nano (fast) but with aggressive augmentation
    print("\n📦 Loading YOLOv8n-cls pretrained model...")
    model = YOLO("yolov8n-cls.pt")
    
    # Train the model with AGGRESSIVE augmentation
    print("\n🔄 Starting improved training...")
    print(f"   Dataset: {dataset_path}")
    print(f"   Model size: nano (yolov8n-cls)")
    print(f"   Epochs: 75 (was 20)")
    print(f"   Image size: 256x256 (was 224x224)")
    print(f"   Batch size: 16 (was 8)")
    print(f"   Augmentation: AGGRESSIVE")
    print(f"   Device: CPU (optimized)")
    print("\n⏱️  This will take ~20-30 minutes on CPU...")
    
    results = model.train(
        data=str(dataset_path),  # Path to dataset
        epochs=75,               # MORE EPOCHS (was 20)
        imgsz=256,               # Larger image size (was 224)
        device='cpu',            # CPU training
        patience=10,             # Early stopping patience (was 5)
        batch=16,                # Larger batch (was 8)
        save=True,               # Save checkpoints
        verbose=True,            # Verbose output
        project="runs/classify", # Project directory
        name="casting_model_v2", # Experiment name
        
        # AGGRESSIVE AUGMENTATION SETTINGS
        hsv_h=0.03,              # HSV hue (was 0.015)
        hsv_s=0.8,               # HSV saturation (was 0.7)
        hsv_v=0.5,               # HSV brightness (was 0.4)
        degrees=25,              # Rotation ±25 degrees (was 0)
        translate=0.15,          # Translation 15% (was 0.1)
        scale=0.6,               # Scale 60% variation (was 0.5)
        flipud=0.3,              # Vertical flip 30% (was 0)
        fliplr=0.5,              # Horizontal flip 50% (same)
        mosaic=1.0,              # Mosaic enabled (same)
        mixup=0.1,               # Mixup blending 10%
        erasing=0.1,             # Random erasing 10%
        perspective=0.0001,      # Slight perspective warp
        
        # Learning rate settings
        lr0=0.001,               # Initial learning rate
        lrf=0.01,                # Final learning rate
        momentum=0.937,          # Optimizer momentum
        weight_decay=0.0005,     # L2 regularization
    )
    
    print("\n" + "=" * 70)
    print("✅ Training COMPLETED!")
    print("=" * 70)
    
    # Save the trained model
    best_model_path = Path("runs/classify/casting_model_v2/weights/best.pt")
    
    if best_model_path.exists():
        import shutil
        final_model_path = model_save_path / "casting_defect_model_v2.pt"
        shutil.copy(best_model_path, final_model_path)
        
        print(f"\n✅ Best model saved to: {final_model_path}")
        print(f"\n📊 IMPROVEMENTS MADE:")
        print(f"   ✓ Epochs: 20 → 75")
        print(f"   ✓ Image size: 224x224 → 256x256")
        print(f"   ✓ Batch size: 8 → 16")
        print(f"   ✓ Rotation: 0° → ±25°")
        print(f"   ✓ HSV augmentation: Increased 2x")
        print(f"   ✓ Added: Mixup, Erasing, Perspective")
        print(f"   ✓ Patience: 5 → 10 (better convergence)")
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"   1. Compare old vs new model:")
        print(f"      Old: models/casting_defect_model.pt")
        print(f"      New: models/casting_defect_model_v2.pt")
        print(f"   2. To use new model, update app.py to load _v2")
        print(f"   3. Run test_pipeline.py to verify improvement")
        
    else:
        print(f"⚠️  Best model not found. Check runs/classify/casting_model_v2/ for results.")

if __name__ == "__main__":
    main()
