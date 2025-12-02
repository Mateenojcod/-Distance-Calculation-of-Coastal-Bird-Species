# Library and Module Guide for Coastal Bird Species Identification

This document provides detailed information about the libraries and modules used in this project for bird detection, classification, and distance calculation from camera trap footage.

## Table of Contents
1. [Overview](#overview)
2. [Video Processing Libraries](#video-processing-libraries)
3. [Machine Learning Libraries](#machine-learning-libraries)
4. [Distance Calculation Methods](#distance-calculation-methods)
5. [Installation](#installation)
6. [Usage Examples](#usage-examples)

---

## Overview

This project implements a comprehensive system for identifying coastal bird species and calculating their distance from camera traps using multiple machine learning approaches and computer vision techniques.

### Core Components
- **Bird Detection**: Locate birds in images/videos
- **Species Classification**: Identify specific bird species
- **Distance Estimation**: Calculate distance from camera to bird

---

## Video Processing Libraries

### 1. OpenCV (opencv-python)
**Primary library for computer vision tasks**

- **Purpose**: Image/video processing, camera calibration, feature detection
- **Use Cases**:
  - Reading and writing videos
  - Image preprocessing
  - Camera calibration
  - Basic object detection
  - Drawing annotations

```python
import cv2

# Read video
cap = cv2.VideoCapture('bird_video.mp4')
ret, frame = cap.read()

# Camera calibration
ret, camera_matrix, dist_coeffs = cv2.calibrateCamera(...)
```

**Pros**: Comprehensive, well-documented, industry standard
**Cons**: Can be complex for beginners

### 2. OpenCV Contrib (opencv-contrib-python)
**Extended OpenCV modules**

- **Purpose**: Additional algorithms and features
- **Use Cases**:
  - Advanced tracking algorithms
  - Structure from Motion (SfM)
  - Extra feature detectors

### 3. MoviePy
**High-level video editing**

- **Purpose**: Video manipulation and editing
- **Use Cases**:
  - Easy video concatenation
  - Adding audio/effects
  - Simple video processing

```python
from moviepy.editor import VideoFileClip

clip = VideoFileClip('input.mp4')
clip.subclip(0, 10).write_videofile('output.mp4')
```

**Pros**: User-friendly API
**Cons**: Slower than OpenCV for frame-by-frame processing

### 4. Pillow (PIL)
**Image manipulation**

- **Purpose**: Image loading, saving, and basic manipulation
- **Use Cases**:
  - Format conversion
  - Image enhancement
  - Basic transformations

---

## Machine Learning Libraries

### Object Detection

#### 1. Ultralytics (YOLOv8) ⭐ **RECOMMENDED**
**State-of-the-art object detection**

- **Purpose**: Fast and accurate bird detection
- **Advantages**:
  - Real-time performance
  - Pre-trained models available
  - Easy to fine-tune for bird species
  - Excellent documentation

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # nano, small, medium, large, xlarge
results = model('bird_image.jpg')
```

**Performance**:
- YOLOv8n: ~45 FPS (lightweight)
- YOLOv8m: ~30 FPS (balanced)
- YOLOv8l: ~20 FPS (accurate)

#### 2. Detectron2 (Facebook AI)
**Research-grade detection framework**

- **Purpose**: State-of-the-art detection and segmentation
- **Advantages**:
  - Multiple architectures (Faster R-CNN, Mask R-CNN)
  - Highly customizable
  - Excellent for research

```python
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file("faster_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
```

**Pros**: Very accurate, flexible
**Cons**: More complex setup, requires more resources

#### 3. TensorFlow Object Detection API
**Google's detection framework**

- **Purpose**: Production-ready object detection
- **Models**: SSD, Faster R-CNN, EfficientDet
- **Use Cases**: Large-scale deployments

### Classification

#### 1. TensorFlow/Keras ⭐ **RECOMMENDED**
**Deep learning framework**

- **Purpose**: Bird species classification
- **Pre-trained Models**:
  - EfficientNet (best for birds)
  - ResNet50
  - MobileNetV2 (for mobile)
  - VGG16

```python
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

base_model = EfficientNetB0(weights='imagenet', include_top=False)
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')  # 10 bird species
])
```

**Why TensorFlow**:
- Excellent transfer learning support
- Large model zoo
- Production-ready (TF Serving, TF Lite)
- Good documentation

#### 2. PyTorch with timm
**Alternative deep learning framework**

- **Purpose**: Research and custom architectures
- **timm library**: PyTorch Image Models
  - 700+ pre-trained models
  - Vision Transformer support
  - EfficientNet variants

```python
import timm

model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=10)
```

**Advantages**:
- More Pythonic
- Better for research
- Growing ecosystem

#### 3. Scikit-learn
**Traditional machine learning**

- **Purpose**: Classical ML algorithms
- **Algorithms**:
  - Random Forest
  - SVM
  - Gradient Boosting

```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=200)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

**When to use**:
- Small datasets
- Interpretable models needed
- Limited computational resources
- Feature-based classification

---

## Distance Calculation Methods

### 1. Pinhole Camera Model ⭐ **MOST COMMON**
**Basic distance estimation using camera geometry**

**Formula**: `distance = (real_height × focal_length) / pixel_height`

**Requirements**:
- Known bird size
- Camera focal length
- Clear detection

```python
from src.distance_calculation import DistanceEstimator

estimator = DistanceEstimator(
    method='pinhole',
    camera_params={'focal_length': 800}
)

distance = estimator.estimate_distance_size_based(
    bbox=[100, 100, 200, 250],
    known_height=0.4  # 40cm seagull
)
```

**Accuracy**: ±10-20% with calibration

### 2. Stereo Vision
**Using two cameras for depth perception**

**Formula**: `distance = (baseline × focal_length) / disparity`

**Requirements**:
- Two synchronized cameras
- Camera calibration
- Feature matching

```python
estimator = DistanceEstimator(
    method='stereo',
    camera_params={'focal_length': 800, 'baseline': 0.1}
)

distance = estimator.estimate_distance_stereo(
    left_point=(100, 150),
    right_point=(90, 150)
)
```

**Accuracy**: ±5-10% (more accurate)
**Complexity**: Higher (requires hardware setup)

### 3. Deep Learning Depth Estimation
**Monocular depth using neural networks**

**Model**: MiDaS (Intel)

```python
estimator = DistanceEstimator(method='deep_learning')

# Generates depth map for entire image
depth_map = estimator.estimate_depth_map(image)

# Get distance at specific location
distance = estimator.estimate_distance_from_depth_map(depth_map, bbox)
```

**Advantages**:
- Single camera
- Works on any video
- Relative depth information

**Limitations**:
- Relative depth (not absolute meters)
- Requires GPU for real-time
- Less accurate than stereo

**Libraries**:
- PyTorch (for MiDaS)
- OpenCV (for preprocessing)

### 4. Ground Plane Assumption
**Assuming birds are on ground level**

**Formula**: Uses trigonometry with camera height and angle

```python
estimator = DistanceEstimator(method='ground_plane')

distance = estimator.estimate_distance_ground_plane(
    bbox=[100, 100, 200, 250],
    camera_height=2.0,  # 2 meters high
    camera_angle=10.0   # 10 degrees down
)
```

**Best for**:
- Shore birds
- Ground-dwelling species
- Fixed camera installations

### 5. Structure from Motion (SfM)
**3D reconstruction from video sequence**

**Uses**: OpenCV's SfM module

```python
import cv2

# Detect features across frames
features = cv2.SIFT_create()
# Match features
# Estimate camera poses
# Triangulate 3D points
```

**Advantages**: Very accurate
**Disadvantages**: Computationally expensive, requires camera motion

---

## Supporting Libraries

### Tracking and Filtering

#### FilterPy
**Kalman filtering for tracking**

```python
from filterpy.kalman import KalmanFilter

kf = KalmanFilter(dim_x=4, dim_z=2)
# Track bird position over time
# Smooth noisy measurements
```

**Use**: Smooth distance measurements, predict bird positions

#### Supervision
**Tracking and annotation utilities**

```python
from supervision import Tracker

tracker = Tracker()
tracked_objects = tracker.update(detections)
```

### Scientific Computing

#### NumPy
**Numerical operations**
- Array operations
- Linear algebra
- Mathematical functions

#### SciPy
**Scientific computing**
- Spatial distance calculations
- Signal processing
- Optimization

```python
from scipy.spatial import distance

# Calculate 3D distance between birds
dist = distance.euclidean(point1, point2)
```

#### Pandas
**Data manipulation**
- Store detection results
- Analyze statistics
- Export to CSV

```python
import pandas as pd

results_df = pd.DataFrame({
    'frame': frames,
    'species': species,
    'distance': distances
})
results_df.to_csv('bird_detections.csv')
```

---

## Installation

### Full Installation
```bash
pip install -r requirements.txt
```

### Minimal Installation (Core Only)
```bash
pip install opencv-python numpy scipy scikit-learn
pip install tensorflow ultralytics
```

### Optional Components
```bash
# For PyTorch-based methods
pip install torch torchvision timm

# For Detectron2
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html

# For advanced visualization
pip install plotly seaborn
```

---

## Performance Comparison

### Detection Speed (1920x1080 video, GPU)

| Method | FPS | Accuracy | Use Case |
|--------|-----|----------|----------|
| YOLOv8n | 45 | Good | Real-time monitoring |
| YOLOv8m | 30 | Better | Balanced |
| YOLOv8l | 20 | Best | Accuracy priority |
| Detectron2 | 15 | Excellent | Research |

### Distance Estimation Accuracy

| Method | Accuracy | Requirements | Complexity |
|--------|----------|--------------|------------|
| Pinhole | ±15% | Calibration | Low |
| Stereo | ±7% | Two cameras | Medium |
| Deep Learning | ±20% | GPU | Medium |
| Ground Plane | ±25% | Known height | Low |

---

## Recommended Setup

### For Production/Field Work
```python
detector = BirdDetector(method='yolo')  # YOLOv8
classifier = BirdClassifier(method='tensorflow')  # EfficientNet
distance_estimator = DistanceEstimator(method='pinhole')
```

### For Research
```python
detector = BirdDetector(method='detectron2')
classifier = BirdClassifier(method='ensemble')  # Multiple models
distance_estimator = DistanceEstimator(method='stereo')
```

### For Real-time/Mobile
```python
detector = BirdDetector(method='yolo')  # Use 'yolov8n'
classifier = BirdClassifier(method='tensorflow')  # Use MobileNetV2
distance_estimator = DistanceEstimator(method='pinhole')
```

---

## Best Practices

1. **Always calibrate your camera** for accurate distance measurements
2. **Use transfer learning** for bird classification (don't train from scratch)
3. **Combine multiple methods** (ensemble) for better accuracy
4. **Track birds across frames** to smooth distance measurements
5. **Use GPU** for real-time processing
6. **Validate on ground truth** measurements when possible

---

## Resources

### Datasets
- **iNaturalist**: Large bird species dataset
- **COCO**: Contains bird class (ID 14)
- **Caltech-UCSD Birds (CUB-200)**: 200 bird species

### Pre-trained Models
- **YOLOv8 weights**: https://github.com/ultralytics/ultralytics
- **TensorFlow Model Zoo**: https://tfhub.dev/
- **timm models**: https://github.com/huggingface/pytorch-image-models

### Documentation
- **OpenCV**: https://docs.opencv.org/
- **TensorFlow**: https://www.tensorflow.org/
- **Ultralytics**: https://docs.ultralytics.com/

---

## Troubleshooting

### Common Issues

1. **GPU not detected**
   ```bash
   # For TensorFlow
   pip install tensorflow[and-cuda]
   
   # For PyTorch
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **ImportError for detectron2**
   - Requires compilation
   - Use pre-built wheels for your CUDA version

3. **Poor distance accuracy**
   - Calibrate camera properly
   - Ensure correct focal length
   - Use appropriate bird size values

4. **Slow processing**
   - Use smaller models (YOLOv8n vs YOLOv8l)
   - Skip frames (process every 2nd or 3rd frame)
   - Use GPU acceleration
   - Reduce input resolution

---

## Summary

This project provides a flexible, modular system for bird identification and distance calculation. You can mix and match components based on your needs:

- **Fast & Simple**: YOLO + TensorFlow + Pinhole
- **Most Accurate**: Detectron2 + Ensemble + Stereo
- **Research**: All methods with comparison

Choose libraries based on:
- Available hardware (GPU/CPU)
- Required accuracy
- Real-time needs
- Dataset size
- Deployment target
