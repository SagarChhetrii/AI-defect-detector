"""
Train YOLOv8 Classification Model for Casting Defect Detection
This script trains a YOLOv8 nano classification model on the casting dataset.
"""

import os
from pathlib import Path
from ultralytics import YOLO

def main():
    """Main training function"""
    
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
    
    print("=" * 60)
    print("🚀 YOLOv8 Classification Model Training")
    print("=" * 60)
    
    # Load pretrained YOLOv8 nano classification model
    # 'n' = nano (smallest, fastest)
    # 's' = small, 'm' = medium, 'l' = large, 'x' = xlarge
    print("\n📦 Loading YOLOv8n-cls pretrained model...")
    model = YOLO("yolov8n-cls.pt")
    
    # Train the model
    print("\n🔄 Starting training...")
    print(f"   Dataset: {dataset_path}")
    print(f"   Model size: nano (yolov8n-cls)")
    print(f"   Epochs: 20")
    print(f"   Image size: 224x224")
    
    results = model.train(
        data=str(dataset_path),  # Path to dataset
        epochs=20,               # Number of epochs (increase for better results)
        imgsz=224,               # Image size for classification
        device='cpu',            # Use CPU (change to 0 if GPU available)
        patience=5,              # Early stopping patience
        batch=8,                 # Batch size (reduced for CPU)
        save=True,               # Save checkpoints
        verbose=True,            # Verbose output
        project="runs/classify",  # Project directory
        name="casting_model",    # Experiment name
    )
    
    print("\n✅ Training completed!")
    
    # Save the trained model
    # The best model is automatically saved in runs/classify/casting_model/weights/best.pt
    best_model_path = Path("runs/classify/casting_model/weights/best.pt")
    
    if best_model_path.exists():
        # Copy to models folder for easier access
        import shutil
        final_model_path = model_save_path / "casting_defect_model.pt"
        shutil.copy(best_model_path, final_model_path)
        print(f"✅ Model saved to: {final_model_path}")
        print(f"\n📊 Training Results:")
        print(f"   - Best model: {final_model_path}")
        print(f"   - Training completed successfully!")
    else:
        print(f"⚠️  Best model not found. Check runs/classify/casting_model/ for results.")

if __name__ == "__main__":
    main()
