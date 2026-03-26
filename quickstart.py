#!/usr/bin/env python3
"""
Quick Start Guide - Run this to verify setup and get started!
"""

import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed"""
    print("=" * 60)
    print("🔍 Checking Dependencies...")
    print("=" * 60)
    
    required_packages = {
        'ultralytics': 'YOLOv8',
        'cv2': 'OpenCV',
        'streamlit': 'Streamlit',
        'plotly': 'Plotly',
        'pandas': 'Pandas'
    }
    
    missing = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name:15} - OK")
        except ImportError:
            print(f"❌ {name:15} - MISSING")
            missing.append(package)
    
    if missing:
        print("\n⚠️  Missing packages. Install them with:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    print("\n✅ All dependencies installed!")
    return True

def check_dataset():
    """Check if dataset structure is correct"""
    print("\n" + "=" * 60)
    print("📁 Checking Dataset Structure...")
    print("=" * 60)
    
    required_paths = [
        'casting_data/train/ok_front',
        'casting_data/train/def_front',
        'casting_data/test/ok_front',
        'casting_data/test/def_front',
    ]
    
    all_exist = True
    
    for path in required_paths:
        p = Path(path)
        if p.exists():
            # Count files
            files = list(p.glob('*'))
            print(f"✅ {path:40} ({len(files)} files)")
        else:
            print(f"❌ {path:40} - MISSING")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  Some dataset folders are missing!")
        print("   Create the following structure:")
        print("   casting_data/")
        print("   ├── train/")
        print("   │   ├── ok_front/      (add OK sample images here)")
        print("   │   └── def_front/     (add DEFECT sample images here)")
        print("   └── test/")
        print("       ├── ok_front/")
        print("       └── def_front/")
        return False
    
    print("\n✅ Dataset structure looks good!")
    return True

def check_scripts():
    """Check if all required scripts exist"""
    print("\n" + "=" * 60)
    print("📜 Checking Script Files...")
    print("=" * 60)
    
    required_scripts = {
        'train.py': 'Training script',
        'webcam.py': 'Webcam detection script',
        'app.py': 'Streamlit dashboard',
        'README.md': 'Documentation'
    }
    
    all_exist = True
    
    for script, description in required_scripts.items():
        if Path(script).exists():
            print(f"✅ {script:15} - {description}")
        else:
            print(f"❌ {script:15} - MISSING")
            all_exist = False
    
    if all_exist:
        print("\n✅ All scripts are ready!")
    
    return all_exist

def main():
    """Main setup check"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + "  🎯 Casting Defect Visual Inspection System".center(58) + "║")
    print("║" + "         Quick Start Guide".center(58) + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check dataset
    dataset_ok = check_dataset()
    
    # Check scripts
    scripts_ok = check_scripts()
    
    # Final summary
    print("\n" + "=" * 60)
    print("📋 Summary")
    print("=" * 60)
    
    print("\n✅ Ready?" if all([deps_ok, dataset_ok, scripts_ok]) else "\n⚠️  Not Ready")
    
    print("\n🚀 Getting Started:")
    print("\n   Step 1: Install dependencies (if needed)")
    print("   $ pip install -r requirements.txt")
    print("\n   Step 2: Train the model")
    print("   $ python train.py")
    print("\n   Step 3: Test with webcam (in another terminal)")
    print("   $ python webcam.py")
    print("\n   Step 4: View dashboard (in another terminal)")
    print("   $ streamlit run app.py")
    
    print("\n📚 For detailed information, read:")
    print("   $ cat README.md")
    
    print("\n" + "=" * 60)
    print("Good luck! 🎯")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
