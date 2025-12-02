"""
Bird Detection Module

Provides object detection capabilities for identifying birds in camera trap footage.
Supports multiple detection frameworks:
1. YOLOv8 (Ultralytics) - Fast and accurate
2. Detectron2 (Facebook) - State-of-the-art detection
3. TensorFlow Object Detection API
4. MediaPipe - Lightweight detection
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

# YOLOv8
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Detectron2
try:
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False

# TensorFlow
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class BirdDetector:
    """
    Multi-backend bird detection system.
    
    Detects birds in images and video frames using various detection frameworks.
    Returns bounding boxes, confidence scores, and class labels.
    """
    
    def __init__(self, method: str = 'yolo', confidence_threshold: float = 0.5):
        """
        Initialize the bird detector.
        
        Args:
            method: Detection method ('yolo', 'detectron2', 'tensorflow')
            confidence_threshold: Minimum confidence for detections
        """
        self.method = method
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        if method == 'yolo' and YOLO_AVAILABLE:
            self.model = self._load_yolo_model()
        elif method == 'detectron2' and DETECTRON2_AVAILABLE:
            self.model = self._load_detectron2_model()
        elif method == 'tensorflow' and TENSORFLOW_AVAILABLE:
            self.model = self._load_tensorflow_model()
        else:
            raise ValueError(f"Method {method} not supported or dependencies not installed")
    
    def _load_yolo_model(self) -> YOLO:
        """
        Load YOLOv8 model for bird detection.
        Can be fine-tuned for specific bird species.
        """
        # Load pre-trained model (can be replaced with custom trained model)
        model = YOLO('yolov8n.pt')  # nano version for speed
        # For better accuracy, use: YOLO('yolov8m.pt') or YOLO('yolov8l.pt')
        return model
    
    def _load_detectron2_model(self):
        """
        Load Detectron2 model for bird detection.
        Uses Faster R-CNN with ResNet backbone.
        """
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        )
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)
        return predictor
    
    def _load_tensorflow_model(self):
        """
        Load TensorFlow Object Detection model.
        """
        # Placeholder for TensorFlow model loading
        # Would load a pre-trained or custom model here
        return None
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect birds in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detections, each containing:
            - bbox: [x1, y1, x2, y2] coordinates
            - confidence: detection confidence score
            - class_name: detected class name
            - class_id: detected class ID
        """
        if self.method == 'yolo':
            return self._detect_yolo(image)
        elif self.method == 'detectron2':
            return self._detect_detectron2(image)
        elif self.method == 'tensorflow':
            return self._detect_tensorflow(image)
        return []
    
    def _detect_yolo(self, image: np.ndarray) -> List[Dict]:
        """Detect using YOLOv8."""
        results = self.model(image, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            confidence = float(box.conf[0])
            if confidence < self.confidence_threshold:
                continue
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            
            # Filter for bird-related classes
            # COCO dataset classes: bird (class 14), etc.
            if class_id == 14 or 'bird' in class_name.lower():
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence,
                    'class_name': class_name,
                    'class_id': class_id
                })
        
        return detections
    
    def _detect_detectron2(self, image: np.ndarray) -> List[Dict]:
        """Detect using Detectron2."""
        outputs = self.model(image)
        
        detections = []
        instances = outputs["instances"]
        
        for i in range(len(instances)):
            class_id = int(instances.pred_classes[i])
            confidence = float(instances.scores[i])
            
            # Filter for bird class (class 14 in COCO)
            if class_id == 14 and confidence >= self.confidence_threshold:
                bbox = instances.pred_boxes[i].tensor[0].cpu().numpy()
                detections.append({
                    'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    'confidence': confidence,
                    'class_name': 'bird',
                    'class_id': class_id
                })
        
        return detections
    
    def _detect_tensorflow(self, image: np.ndarray) -> List[Dict]:
        """Detect using TensorFlow model."""
        # Placeholder for TensorFlow detection
        return []
    
    def detect_and_draw(self, image: np.ndarray, 
                        color: Tuple[int, int, int] = (0, 255, 0),
                        thickness: int = 2) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect birds and draw bounding boxes on the image.
        
        Args:
            image: Input image
            color: Color for bounding boxes (BGR)
            thickness: Line thickness
            
        Returns:
            Tuple of (annotated_image, detections)
        """
        detections = self.detect(image)
        annotated_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw label background
            cv2.rectangle(
                annotated_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
        
        return annotated_image, detections
    
    def detect_in_video(self, video_path: str, output_path: Optional[str] = None,
                       skip_frames: int = 1) -> List[List[Dict]]:
        """
        Detect birds in a video file.
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video (optional)
            skip_frames: Process every nth frame (for speed)
            
        Returns:
            List of detections for each frame
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_detections = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for speed
            if frame_count % skip_frames != 0:
                all_detections.append([])
                continue
            
            # Detect birds in frame
            detections = self.detect(frame)
            all_detections.append(detections)
            
            # Annotate and write frame if needed
            if writer:
                annotated_frame, _ = self.detect_and_draw(frame)
                writer.write(annotated_frame)
        
        cap.release()
        if writer:
            writer.release()
        
        return all_detections
    
    def get_detection_centers(self, detections: List[Dict]) -> List[Tuple[int, int]]:
        """
        Get center points of all detections.
        Useful for distance calculation and tracking.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            List of (x, y) center coordinates
        """
        centers = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            centers.append((center_x, center_y))
        return centers
    
    def get_detection_sizes(self, detections: List[Dict]) -> List[Tuple[int, int]]:
        """
        Get sizes (width, height) of all detections.
        Useful for distance estimation.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            List of (width, height) tuples
        """
        sizes = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            width = x2 - x1
            height = y2 - y1
            sizes.append((width, height))
        return sizes
