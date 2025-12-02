"""
Example Usage of Coastal Bird Species Identification System

This script demonstrates how to use the various modules for:
1. Detecting birds in camera trap footage
2. Classifying bird species
3. Calculating distances from the camera
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
from src.bird_identification import BirdDetector, BirdClassifier
from src.distance_calculation import DistanceEstimator, CameraCalibrator


def example_image_detection():
    """Example: Detect and classify birds in a single image."""
    print("=" * 60)
    print("Example 1: Image Detection and Classification")
    print("=" * 60)
    
    # Initialize detector
    print("\n1. Initializing bird detector...")
    detector = BirdDetector(method='yolo', confidence_threshold=0.5)
    
    # Initialize classifier
    print("2. Initializing bird classifier...")
    classifier = BirdClassifier(method='tensorflow', num_classes=10)
    classifier.set_class_names([
        "Seagull", "Pelican", "Cormorant", "Tern", "Sandpiper",
        "Plover", "Oystercatcher", "Albatross", "Petrel", "Heron"
    ])
    
    # Initialize distance estimator
    print("3. Initializing distance estimator...")
    camera_params = {
        'focal_length': 800,
        'image_height': 1080,
        'image_width': 1920
    }
    distance_estimator = DistanceEstimator(method='pinhole', camera_params=camera_params)
    
    # Example: Load and process an image
    print("\n4. Processing would happen here with actual image...")
    print("   - Load image: cv2.imread('path/to/image.jpg')")
    print("   - Detect birds: detections = detector.detect(image)")
    print("   - For each detection:")
    print("     * Classify species: species, conf, probs = classifier.predict(bird_crop)")
    print("     * Calculate distance: dist_info = distance_estimator.estimate_distance(image, bbox, species)")
    
    print("\nExample output format:")
    print({
        'bbox': [100, 100, 200, 250],
        'confidence': 0.92,
        'species': 'Seagull',
        'distance_meters': 15.3,
        'position_3d': (1.2, 0.5, 15.3)
    })


def example_video_processing():
    """Example: Process video and track birds with distance calculation."""
    print("\n" + "=" * 60)
    print("Example 2: Video Processing with Distance Tracking")
    print("=" * 60)
    
    print("\nSetup:")
    print("1. detector = BirdDetector(method='yolo')")
    print("2. classifier = BirdClassifier(method='tensorflow')")
    print("3. distance_estimator = DistanceEstimator(method='pinhole')")
    
    print("\nProcessing pipeline:")
    print("for frame in video:")
    print("  1. detections = detector.detect(frame)")
    print("  2. for each detection:")
    print("     - species = classifier.predict(crop)")
    print("     - distance = distance_estimator.estimate_distance(frame, bbox, species)")
    print("     - track_id = tracker.update(detection)")
    print("  3. Draw annotations on frame")
    print("  4. Save to output video")


def example_camera_calibration():
    """Example: Camera calibration for accurate distance measurement."""
    print("\n" + "=" * 60)
    print("Example 3: Camera Calibration")
    print("=" * 60)
    
    print("\nMethod 1: From Checkerboard Images")
    print("calibrator = CameraCalibrator(checkerboard_size=(9, 6), square_size=0.025)")
    print("results = calibrator.calibrate_from_images(image_paths)")
    print("calibrator.save_calibration('camera_calibration.pkl')")
    
    print("\nMethod 2: From Field of View")
    print("calibrator = CameraCalibrator()")
    print("params = calibrator.estimate_camera_params_from_fov(")
    print("    image_width=1920,")
    print("    image_height=1080,")
    print("    horizontal_fov=60.0")
    print(")")
    
    print("\nCalibration enables:")
    print("- Lens distortion correction")
    print("- Accurate focal length measurement")
    print("- Precise distance calculations")


def example_multiple_methods():
    """Example: Using multiple ML methods and distance calculation approaches."""
    print("\n" + "=" * 60)
    print("Example 4: Multiple Methods Comparison")
    print("=" * 60)
    
    print("\nClassification Methods:")
    print("1. TensorFlow with EfficientNet: BirdClassifier(method='tensorflow')")
    print("2. PyTorch with timm models: BirdClassifier(method='pytorch')")
    print("3. Scikit-learn ensemble: BirdClassifier(method='sklearn')")
    print("4. Ensemble of all: BirdClassifier(method='ensemble')")
    
    print("\nDistance Estimation Methods:")
    print("1. Pinhole camera model: DistanceEstimator(method='pinhole')")
    print("2. Stereo vision: DistanceEstimator(method='stereo')")
    print("3. Deep learning depth: DistanceEstimator(method='deep_learning')")
    print("4. Ground plane assumption: DistanceEstimator(method='ground_plane')")
    
    print("\nDetection Methods:")
    print("1. YOLOv8: BirdDetector(method='yolo')")
    print("2. Detectron2: BirdDetector(method='detectron2')")
    print("3. TensorFlow: BirdDetector(method='tensorflow')")


def example_complete_pipeline():
    """Example: Complete pipeline from video to results."""
    print("\n" + "=" * 60)
    print("Example 5: Complete Pipeline")
    print("=" * 60)
    
    print("""
Complete workflow for processing camera trap footage:

1. SETUP PHASE
   - Load configuration from YAML
   - Initialize detector, classifier, and distance estimator
   - Load camera calibration (if available)

2. VIDEO PROCESSING PHASE
   for each frame in video:
       a. Detect birds
          detections = detector.detect(frame)
       
       b. Classify each bird
          for detection in detections:
              crop = extract_crop(frame, detection['bbox'])
              species, confidence = classifier.predict(crop)
       
       c. Calculate distances
          for detection, species in zip(detections, species_list):
              distance_info = distance_estimator.estimate_distance(
                  frame, detection['bbox'], species
              )
       
       d. Track birds across frames
          track_id = tracker.update(detection, species, distance)
       
       e. Store results
          results.append({
              'frame': frame_number,
              'species': species,
              'confidence': confidence,
              'distance': distance,
              'position_3d': position
          })

3. OUTPUT PHASE
   - Save annotated video
   - Export results to CSV/JSON
   - Generate visualization plots
   - Create summary statistics
    """)


def print_library_recommendations():
    """Print recommended libraries for each component."""
    print("\n" + "=" * 60)
    print("Recommended Libraries by Component")
    print("=" * 60)
    
    print("\n📹 VIDEO PROCESSING & COMPUTER VISION:")
    print("  • opencv-python - Core video and image processing")
    print("  • opencv-contrib-python - Additional CV algorithms")
    print("  • moviepy - High-level video editing")
    print("  • imutils - Convenience functions for OpenCV")
    
    print("\n🔍 OBJECT DETECTION:")
    print("  • ultralytics (YOLOv8) - Fast, accurate, easy to use")
    print("  • detectron2 - State-of-the-art detection (Facebook)")
    print("  • tensorflow-hub - Pre-trained TF detection models")
    print("  • supervision - Tracking and annotation utilities")
    
    print("\n🧠 MACHINE LEARNING & CLASSIFICATION:")
    print("  • tensorflow/keras - Deep learning framework")
    print("  • pytorch/torchvision - Alternative DL framework")
    print("  • scikit-learn - Traditional ML algorithms")
    print("  • timm - PyTorch image models library")
    
    print("\n📏 DISTANCE CALCULATION:")
    print("  • scipy - Scientific computing and spatial distance")
    print("  • filterpy - Kalman filtering for tracking")
    print("  • mediapipe - Lightweight ML solutions")
    print("  • numpy - Numerical computations")
    
    print("\n🎯 DEPTH ESTIMATION:")
    print("  • MiDaS (via torch.hub) - Monocular depth estimation")
    print("  • DepthAI - Hardware-accelerated depth")
    print("  • OpenCV stereo algorithms - Stereo vision depth")
    
    print("\n📊 DATA & VISUALIZATION:")
    print("  • pandas - Data manipulation")
    print("  • matplotlib - Plotting and visualization")
    print("  • seaborn - Statistical visualization")
    print("  • plotly - Interactive visualizations")


if __name__ == "__main__":
    print("╔" + "=" * 58 + "╗")
    print("║  Coastal Bird Species Identification - Example Usage  ║")
    print("╚" + "=" * 58 + "╝")
    
    # Run examples
    example_image_detection()
    example_video_processing()
    example_camera_calibration()
    example_multiple_methods()
    example_complete_pipeline()
    print_library_recommendations()
    
    print("\n" + "=" * 60)
    print("✅ Examples completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Prepare your camera trap footage")
    print("3. Optional: Calibrate your camera")
    print("4. Run detection and classification on your data")
    print("5. Analyze distance measurements and species distribution")
