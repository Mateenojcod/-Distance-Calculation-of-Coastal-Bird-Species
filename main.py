#!/usr/bin/env python3
"""
Main Entry Point for Coastal Bird Species Identification System

This script provides a command-line interface for processing camera trap footage
to detect, classify, and calculate distances to coastal bird species.
"""

import argparse
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.bird_identification import BirdDetector, BirdClassifier
from src.distance_calculation import DistanceEstimator


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def process_image(image_path, config):
    """Process a single image."""
    import cv2
    
    print(f"Processing image: {image_path}")
    
    # Initialize components
    detector = BirdDetector(
        method=config['detection']['method'],
        confidence_threshold=config['detection']['confidence_threshold']
    )
    
    classifier = BirdClassifier(
        method=config['classification']['method'],
        num_classes=config['classification']['num_classes']
    )
    classifier.set_class_names(config['classification']['class_names'])
    
    distance_estimator = DistanceEstimator(
        method=config['distance']['method'],
        camera_params=config['camera']
    )
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Detect birds
    print("Detecting birds...")
    detections = detector.detect(image)
    print(f"Found {len(detections)} bird(s)")
    
    # Process each detection
    results = []
    for i, detection in enumerate(detections):
        print(f"\nBird {i+1}:")
        print(f"  Confidence: {detection['confidence']:.2f}")
        
        # Extract bird crop for classification
        x1, y1, x2, y2 = detection['bbox']
        bird_crop = image[y1:y2, x1:x2]
        
        # Classify species
        if bird_crop.size > 0:
            species, conf, probs = classifier.predict(bird_crop)
            print(f"  Species: {species} ({conf:.2f})")
        else:
            species = "Unknown"
            conf = 0.0
        
        # Calculate distance
        distance_info = distance_estimator.estimate_distance(
            image, detection['bbox'], species
        )
        print(f"  Distance: {distance_info['distance_meters']:.2f} meters")
        
        results.append({
            'bbox': detection['bbox'],
            'detection_confidence': detection['confidence'],
            'species': species,
            'classification_confidence': conf,
            'distance_meters': distance_info['distance_meters']
        })
    
    return results, detections


def process_video(video_path, config, output_path=None):
    """Process a video file."""
    import cv2
    
    print(f"Processing video: {video_path}")
    
    # Initialize components
    detector = BirdDetector(
        method=config['detection']['method'],
        confidence_threshold=config['detection']['confidence_threshold']
    )
    
    print("Video processing would happen here...")
    print("This is a demonstration of the system structure.")
    print("\nFor full video processing, the system would:")
    print("1. Load video with cv2.VideoCapture")
    print("2. Process each frame (or skip frames for speed)")
    print("3. Detect, classify, and calculate distances")
    print("4. Track birds across frames")
    print("5. Save annotated video and results")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Coastal Bird Species Identification and Distance Calculation'
    )
    parser.add_argument(
        'input',
        help='Path to input image or video'
    )
    parser.add_argument(
        '--config',
        default='config/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output',
        help='Path to output file (for video processing)'
    )
    parser.add_argument(
        '--type',
        choices=['image', 'video'],
        help='Input type (auto-detected if not specified)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}")
        print("Using default parameters")
        config = {
            'detection': {'method': 'yolo', 'confidence_threshold': 0.5},
            'classification': {
                'method': 'tensorflow',
                'num_classes': 10,
                'class_names': [
                    "Seagull", "Pelican", "Cormorant", "Tern", "Sandpiper",
                    "Plover", "Oystercatcher", "Albatross", "Petrel", "Heron"
                ]
            },
            'distance': {'method': 'pinhole'},
            'camera': {'focal_length': 800, 'image_width': 1920, 'image_height': 1080}
        }
    else:
        config = load_config(config_path)
    
    # Determine input type
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    input_type = args.type
    if input_type is None:
        # Auto-detect based on extension
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv'}
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        ext = input_path.suffix.lower()
        if ext in video_exts:
            input_type = 'video'
        elif ext in image_exts:
            input_type = 'image'
        else:
            print(f"Error: Unknown file type: {ext}")
            return 1
    
    # Process input
    try:
        if input_type == 'image':
            process_image(str(input_path), config)
        else:
            process_video(str(input_path), config, args.output)
        
        print("\n✅ Processing complete!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
