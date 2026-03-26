# 🔍 Casting Defect Visual Inspection System

A beginner-friendly AI-powered visual inspection system for detecting casting defects using YOLOv8 classification, OpenCV, and Streamlit.

## 📋 Project Structure

```
hackathon/
├── casting_data/              # Your training dataset
│   ├── train/
│   │   ├── ok_front/         # OK samples
│   │   └── def_front/        # Defect samples
│   └── test/
│       ├── ok_front/
│       └── def_front/
├── models/                    # Saved trained models
├── train.py                   # Training script
├── webcam.py                  # Real-time detection
├── app.py                     # Streamlit dashboard
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `ultralytics` (YOLOv8)
- `opencv-python` (OpenCV)
- `streamlit` (Dashboard)
- `plotly` (Charts)
- `pandas` (Data handling)

### Step 2: Train the Model

```bash
python train.py
```

**What happens:**
- Loads YOLOv8 nano pretrained model
- Trains for 20 epochs on your dataset
- Saves best model to `models/casting_defect_model.pt`
- Shows training metrics and results

**Expected output:**
```
============================================================
🚀 YOLOv8 Classification Model Training
============================================================

📦 Loading YOLOv8n-cls pretrained model...

🔄 Starting training...
   Dataset: casting_data
   Model size: nano (yolov8n-cls)
   Epochs: 20
   Image size: 224x224

✅ Training completed!
✅ Model saved to: models/casting_defect_model.pt
```

**Training Tips:**
- ⏱️ Nano model trains fast (10-30 min on CPU, 2-5 min on GPU)
- 📈 Increase `epochs` from 20 to 30-50 for better accuracy
- 📸 More training images = better results (aim for 100+ per class)
- 🔧 Use GPU if available for 5-10x faster training

### Step 3: Run Real-time Webcam Detection

```bash
python webcam.py
```

**Controls:**
- `q` - Quit application
- `s` - Save screenshot

**What you'll see:**
- Live video feed from webcam
- Real-time predictions (OK or DEFECT)
- Confidence score
- Frame count and defect statistics

**Features:**
- 🎥 Processes each frame in real-time
- 📊 Shows cumulative statistics
- 💾 Save screenshots with predictions
- 🎯 Color-coded results (Green=OK, Red=DEFECT)

### Step 4: View Dashboard

```bash
streamlit run app.py
```

Then open your browser to: `http://localhost:8501`

**Dashboard Pages:**

1. **📊 Dashboard** - Main metrics and overview
   - Total items inspected
   - OK items count
   - Defective items count
   - Defect rate percentage
   - Pie chart visualization
   - Quick action buttons

2. **🎥 Live Detection** - Image inference
   - Upload images for testing
   - Get instant predictions
   - Add results to statistics

3. **📈 Statistics** - Detailed analysis
   - Full inspection history
   - Detailed metrics
   - Export data as CSV

4. **ℹ️ About** - System information
   - Technology stack details
   - Usage instructions
   - Troubleshooting guide

## 📁 File Descriptions

### `train.py`
Trains a YOLOv8 nano classification model on your casting dataset.

**Key Features:**
- ✅ Automatic dataset validation
- ✅ GPU/CPU support
- ✅ Early stopping (patience=5)
- ✅ Model visualization and checkpoints
- ✅ Detailed logging

**Configuration Options:**
```python
epochs=20          # Number of training iterations
batch=16           # Batch size (increase for more GPU memory)
imgsz=224          # Image size (224 is standard for classification)
patience=5         # Early stopping patience
```

**Output:**
- Best model: `models/casting_defect_model.pt`
- Training logs: `runs/classify/casting_model/`
- Results: Training/validation plots and metrics

### `webcam.py`
Real-time defect detection using trained model and webcam.

**Key Features:**
- 🎥 Real-time inference
- 📊 Live statistics display
- 💾 Screenshot capability
- 🎨 Color-coded predictions
- ⚡ Confidence thresholding

**Configuration Options:**
```python
conf=0.5           # Confidence threshold (0-1)
device=0           # GPU device or 'cpu'
```

**Output:**
- Live video with predictions
- Console statistics
- Screenshot files (screenshot_*.jpg)

### `app.py`
Interactive Streamlit dashboard for tracking inspection metrics.

**Key Features:**
- 📊 Real-time statistics
- 📈 Data visualization (pie charts)
- 📤 CSV export
- 💾 Auto-save inspection history
- 🎯 Both automated and manual logging

**Data Persistence:**
- Saves to `inspection_stats.json`
- Loads automatically on dashboard start

## 🔧 Configuration & Customization

### Model Size Options
In `train.py`, change `yolov8n-cls.pt` to:
- `yolov8n-cls.pt` - Nano (fastest, lowest accuracy)
- `yolov8s-cls.pt` - Small
- `yolov8m-cls.pt` - Medium
- `yolov8l-cls.pt` - Large
- `yolov8x-cls.pt` - Extra Large (slowest, highest accuracy)

### Class Names
If your predictions are inverted (showing wrong class), edit `webcam.py`:
```python
class_names = {
    0: "DEFECT",   # Swap these
    1: "OK"        # If results are inverted
}
```

### Training Parameters
Edit these in `train.py`:
```python
model.train(
    epochs=30,          # More epochs = better but slower
    batch=32,           # Larger batch = faster but needs more RAM
    imgsz=224,          # Standard for classification
    patience=10,        # Early stopping patience
    device=0,           # Use '0' for GPU, 'cpu' for CPU
)
```

### Confidence Threshold
In `webcam.py`:
```python
results = model(frame, conf=0.7, verbose=False)  # Higher = stricter
```

## 📊 Understanding the Results

### Classification Output
- **Class**: OK or DEFECT
- **Confidence**: 0-1 (higher = more confident)
- **Defect Rate**: Number of defects / Total items × 100%

### Training Metrics
- **Accuracy**: Percentage of correct predictions
- **Precision**: Of predicted defects, how many were correct
- **Recall**: Of actual defects, how many were found
- **mAP (mean Average Precision)**: Overall model performance

## ⚠️ Troubleshooting

### Model Not Found
```
❌ Error: Trained model not found!
```
**Solution:** Run `python train.py` first

### Webcam Not Opening
```
❌ Error: Could not open webcam!
```
**Solutions:**
- Check if camera is in use by another app
- Try changing device ID: `cv2.VideoCapture(1)` in webcam.py
- Check camera permissions on your OS

### Dataset Path Not Found
```
❌ Error: Dataset path 'casting_data' not found!
```
**Solution:** Ensure folder structure is correct:
```
casting_data/
├── train/
│   ├── ok_front/
│   └── def_front/
└── test/
    ├── ok_front/
    └── def_front/
```

### Out of Memory During Training
**Solutions:**
- Reduce batch size: `batch=8` (default is 16)
- Use smaller model: `yolov8n-cls.pt` (using nano already)
- Reduce image size: `imgsz=192` (from 224)
- Use CPU: `device='cpu'`

### Poor Model Accuracy
**Solutions:**
- Collect more training data (100+ per class minimum)
- Increase training epochs: `epochs=50` or higher
- Use larger model: `yolov8m-cls.pt`
- Ensure consistent image quality and lighting

### Inverted Predictions
**Solution:** Swap class_names in `webcam.py`:
```python
class_names = {
    0: "DEFECT",
    1: "OK"
}
```

## 🎯 Usage Examples

### Example 1: Basic Workflow
```bash
# Train model
python train.py

# Test with webcam
python webcam.py

# View dashboard
streamlit run app.py
```

### Example 2: Batch Processing
Modify `webcam.py` to process images from a folder instead of webcam:
```python
import glob
for image_path in glob.glob("test_images/*.jpg"):
    image = cv2.imread(image_path)
    result_class, confidence = process_image(image, model)
```

### Example 3: API Integration
Save predictions and statistics for integration with external systems:
```python
import json
with open("predictions.json", "w") as f:
    json.dump(results, f)
```

## 📚 Learning Resources

- **YOLOv8 Documentation**: https://docs.ultralytics.com/
- **OpenCV Tutorials**: https://docs.opencv.org/master/d9/df8/tutorial_root.html
- **Streamlit Documentation**: https://docs.streamlit.io/
- **Image Classification**: https://en.wikipedia.org/wiki/Image_classification

## 🐍 Python Version
- Python 3.8 or higher
- Recommended: Python 3.10+

## 🖥️ System Requirements
- **CPU**: Intel i5 or equivalent (for training on CPU)
- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: 2GB free space (for models and datasets)
- **GPU** (Optional): CUDA-capable GPU for faster training

## 📋 Checklist

Before running:
- [ ] Python 3.8+ installed
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Dataset folder structure verified
- [ ] At least 20 images per class (minimum)

When training:
- [ ] Dataset in correct location (casting_data/)
- [ ] Enough disk space (1GB+)
- [ ] No other GPU-intensive processes running

When using dashboard:
- [ ] Model trained (`models/casting_defect_model.pt` exists)
- [ ] Streamlit installed (`pip install streamlit`)
- [ ] Port 8501 available

## 🚨 Important Notes

1. **First Run**: Training takes 10-60 minutes depending on dataset size
2. **GPU**: Using GPU can reduce training time by 80-90%
3. **Accuracy**: More images = better model (aim for 500+ images per class)
4. **Webcam**: Ensure good lighting for best results
5. **Exports**: CSV exports include timestamp, class, and confidence

## 📝 Class Names Convention

The system uses folder names as class labels:
- `ok_front/` → Class: "OK"
- `def_front/` → Class: "DEFECT"

If you want different names, rename the folders or modify class_names in the code.

## 🎓 What You'll Learn

By using this system, you'll learn:
- ✅ Deep learning fundamentals
- ✅ Computer vision basics
- ✅ Model training and evaluation
- ✅ Real-time inference
- ✅ Data visualization
- ✅ Web dashboard development

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section above
2. Verify folder structure matches the template
3. Ensure all dependencies are installed
4. Check YOLOv8 documentation: https://docs.ultralytics.com/

## 📄 License

This project is provided as-is for educational and commercial use.

## 🙏 Acknowledgments

- **Ultralytics** for YOLOv8
- **OpenCV** for image processing
- **Streamlit** for the web framework
- **Community** for feedback and improvements

---

**Happy Inspecting! 🎯**

Built with ❤️ for quality assurance and defect detection.
# AI-defect-detector
