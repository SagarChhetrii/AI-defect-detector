"""
Resume YOLOv8 Training from Epoch 32
Continues from the last checkpoint and trains remaining epochs (32-150)
"""

import os
from pathlib import Path
from ultralytics import YOLO

def main():
    """Resume training from epoch 32"""
    
    # Define paths
    dataset_path = Path("casting_data")
    checkpoint_path = Path("runs/classify/runs/classify/casting_ultimate_model_384px_ensemble1/weights/last.pt")
    
    print("=" * 80)
    print("  🚀 RESUMING YOLOv8 Training from Epoch 32")
    print("=" * 80)
    
    # Validate checkpoint exists
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return
    
    if not dataset_path.exists():
        print(f"❌ Error: Dataset not found at {dataset_path}")
        return
    
    print(f"\n✅ Found checkpoint: {checkpoint_path}")
    print(f"✅ Dataset ready: {dataset_path}")
    
    # Load model from last checkpoint (already trained to epoch 31)
    print("\n📦 Loading pre-trained checkpoint...")
    model = YOLO(str(checkpoint_path))
    
    print(f"\n🔄 Resuming training from epoch 32...")
    print(f"   Starting from: Epoch 32")
    print(f"   Total epochs: 150")
    print(f"   Remaining: 118 epochs")
    print(f"   Batch size: 32")
    print(f"   Image size: 384×384")
    print(f"   Device: CPU")
    
    # Resume training with resume=True
    # This tells YOLO to continue from the checkpoint with the same settings
    results = model.train(
        data=str(dataset_path),
        epochs=150,  # Total epochs (YOLO will resume from where it left off)
        imgsz=384,
        device='cpu',
        patience=20,
        batch=32,
        save=True,
        verbose=True,
        project="runs/classify",
        name="casting_ultimate_model_384px_ensemble1",
        resume=True,  # CRITICAL: Resume from checkpoint
        exist_ok=True,  # Allow overwriting the same experiment
        
        # Keep augmentation settings from original training
        hsv_h=0.05,
        hsv_s=0.9,
        hsv_v=0.6,
        degrees=30,
        translate=0.2,
        scale=0.7,
        shear=0.1,
        flipud=0.4,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        erasing=0.15,
        perspective=0.0002,
        
        # Learning rate
        lr0=0.001,
        lrf=0.0001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        cos_lr=True,
        close_mosaic=20,
        
        save_period=-1,
        plots=True,
    )
    
    print("\n" + "=" * 80)
    print("✅ Training Resumed Successfully!")
    print("=" * 80)
    print("""
    Next Steps:
    1. Monitor the training progress in the terminal
    2. Check training curves at: runs/classify/casting_ultimate_model_384px_ensemble1/
    3. The best model will be saved automatically
    4. When complete, the model will be ready for deployment
    """)


if __name__ == "__main__":
    main()
