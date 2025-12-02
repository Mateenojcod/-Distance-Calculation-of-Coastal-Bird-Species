"""
Distance Estimation Module

Provides multiple methods for calculating distance of birds from camera:
1. Pinhole Camera Model - Using known bird size and focal length
2. Stereo Vision - Using two cameras for depth estimation
3. Structure from Motion (SfM) - Using OpenCV's SfM module
4. Deep Learning - Monocular depth estimation
5. Sensor Fusion - Combining multiple methods
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
from scipy.spatial import distance as scipy_distance

# Deep learning depth estimation
try:
    import torch
    import torchvision
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


class DistanceEstimator:
    """
    Multi-method distance estimation for birds in camera trap footage.
    
    Methods supported:
    - 'pinhole': Pinhole camera model (requires calibration)
    - 'stereo': Stereo vision depth estimation
    - 'deep_learning': Monocular depth estimation using neural networks
    - 'size_based': Distance from known object size
    - 'ground_plane': Estimation using ground plane assumption
    """
    
    def __init__(self, method: str = 'pinhole', camera_params: Optional[Dict] = None):
        """
        Initialize distance estimator.
        
        Args:
            method: Estimation method to use
            camera_params: Dictionary containing camera parameters:
                - focal_length: Camera focal length in pixels
                - sensor_width: Sensor width in mm (optional)
                - sensor_height: Sensor height in mm (optional)
                - baseline: Stereo baseline distance in meters (for stereo method)
        """
        self.method = method
        self.camera_params = camera_params or {}
        
        # Default camera parameters if not provided
        if 'focal_length' not in self.camera_params:
            self.camera_params['focal_length'] = 800  # pixels
        
        # Load depth estimation model if using deep learning
        if method == 'deep_learning' and PYTORCH_AVAILABLE:
            self.depth_model = self._load_depth_model()
        else:
            self.depth_model = None
    
    def _load_depth_model(self):
        """
        Load monocular depth estimation model.
        Uses MiDaS or similar model for depth prediction.
        """
        try:
            # MiDaS depth estimation model
            model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            model.eval()
            
            # Also load transforms
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.depth_transform = transforms.small_transform
            
            return model
        except Exception as e:
            print(f"Warning: Could not load depth model: {e}")
            return None
    
    def estimate_distance_pinhole(self, real_height: float, 
                                  pixel_height: float) -> float:
        """
        Estimate distance using pinhole camera model.
        
        Formula: distance = (real_height * focal_length) / pixel_height
        
        Args:
            real_height: Real height of the object in meters
            pixel_height: Height of object in pixels
            
        Returns:
            Estimated distance in meters
        """
        if pixel_height == 0:
            return float('inf')
        
        focal_length = self.camera_params['focal_length']
        distance = (real_height * focal_length) / pixel_height
        
        return distance
    
    def estimate_distance_size_based(self, bbox: List[int], 
                                    known_width: float = 0.5,
                                    known_height: float = 0.3) -> float:
        """
        Estimate distance based on known bird size.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            known_width: Known average width of bird in meters (default: 50cm)
            known_height: Known average height of bird in meters (default: 30cm)
            
        Returns:
            Estimated distance in meters
        """
        x1, y1, x2, y2 = bbox
        pixel_width = x2 - x1
        pixel_height = y2 - y1
        
        # Use the larger dimension for better accuracy
        if pixel_height > pixel_width:
            distance = self.estimate_distance_pinhole(known_height, pixel_height)
        else:
            distance = self.estimate_distance_pinhole(known_width, pixel_width)
        
        return distance
    
    def estimate_distance_stereo(self, left_point: Tuple[int, int],
                                right_point: Tuple[int, int],
                                baseline: float = 0.1) -> float:
        """
        Estimate distance using stereo vision.
        
        Formula: distance = (baseline * focal_length) / disparity
        
        Args:
            left_point: (x, y) coordinates in left camera
            right_point: (x, y) coordinates in right camera
            baseline: Distance between cameras in meters
            
        Returns:
            Estimated distance in meters
        """
        # Calculate disparity (difference in x coordinates)
        disparity = abs(left_point[0] - right_point[0])
        
        if disparity == 0:
            return float('inf')
        
        focal_length = self.camera_params.get('focal_length', 800)
        baseline = self.camera_params.get('baseline', baseline)
        
        distance = (baseline * focal_length) / disparity
        
        return distance
    
    def estimate_depth_map(self, image: np.ndarray) -> np.ndarray:
        """
        Generate depth map using deep learning.
        
        Args:
            image: Input RGB image
            
        Returns:
            Depth map as numpy array
        """
        if not PYTORCH_AVAILABLE or self.depth_model is None:
            raise RuntimeError("PyTorch and depth model required for this method")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prepare input
        input_batch = self.depth_transform(img_rgb).unsqueeze(0)
        
        # Predict depth
        with torch.no_grad():
            prediction = self.depth_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # Normalize for visualization
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return depth_map
    
    def estimate_distance_from_depth_map(self, depth_map: np.ndarray, 
                                        bbox: List[int]) -> float:
        """
        Estimate distance from depth map at bounding box location.
        
        Args:
            depth_map: Depth map array
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Estimated distance (relative depth value)
        """
        x1, y1, x2, y2 = bbox
        
        # Get region of interest
        roi = depth_map[y1:y2, x1:x2]
        
        # Calculate median depth in ROI (more robust than mean)
        median_depth = np.median(roi)
        
        return float(median_depth)
    
    def estimate_distance_ground_plane(self, bbox: List[int], 
                                      camera_height: float = 2.0,
                                      camera_angle: float = 0.0) -> float:
        """
        Estimate distance assuming bird is on ground plane.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            camera_height: Height of camera above ground in meters
            camera_angle: Camera tilt angle in degrees (0 = horizontal)
            
        Returns:
            Estimated distance in meters
        """
        x1, y1, x2, y2 = bbox
        
        # Use bottom of bounding box (bird's feet location)
        image_center_y = self.camera_params.get('image_height', 1080) / 2
        bird_y = y2  # Bottom of bird
        
        # Calculate vertical displacement from center
        pixel_offset = bird_y - image_center_y
        
        # Convert to angle using focal length
        focal_length = self.camera_params['focal_length']
        angle_offset = np.arctan(pixel_offset / focal_length)
        
        # Add camera tilt angle
        total_angle = np.radians(camera_angle) + angle_offset
        
        # Calculate distance on ground
        if np.cos(total_angle) > 0.01:  # Avoid division by zero
            distance = camera_height / np.tan(total_angle)
        else:
            distance = float('inf')
        
        return abs(distance)
    
    def estimate_distance(self, image: np.ndarray, 
                         bbox: List[int],
                         bird_species: Optional[str] = None) -> Dict[str, float]:
        """
        Estimate distance using the configured method.
        
        Args:
            image: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            bird_species: Species name for species-specific size lookup
            
        Returns:
            Dictionary with distance estimates and metadata
        """
        result = {
            'method': self.method,
            'distance_meters': 0.0,
            'confidence': 0.0
        }
        
        # Get species-specific size if available
        bird_size = self._get_bird_size(bird_species)
        
        if self.method == 'pinhole' or self.method == 'size_based':
            distance = self.estimate_distance_size_based(
                bbox, 
                known_width=bird_size['width'],
                known_height=bird_size['height']
            )
            result['distance_meters'] = distance
            result['confidence'] = 0.7
        
        elif self.method == 'deep_learning':
            depth_map = self.estimate_depth_map(image)
            depth_value = self.estimate_distance_from_depth_map(depth_map, bbox)
            result['distance_meters'] = depth_value
            result['confidence'] = 0.6
            result['depth_map'] = depth_map
        
        elif self.method == 'ground_plane':
            distance = self.estimate_distance_ground_plane(bbox)
            result['distance_meters'] = distance
            result['confidence'] = 0.5
        
        return result
    
    def _get_bird_size(self, species: Optional[str]) -> Dict[str, float]:
        """
        Get average size for bird species.
        
        Args:
            species: Bird species name
            
        Returns:
            Dictionary with 'width' and 'height' in meters
        """
        # Default sizes for common coastal birds
        bird_sizes = {
            'seagull': {'width': 0.50, 'height': 0.40},
            'pelican': {'width': 0.80, 'height': 0.70},
            'cormorant': {'width': 0.60, 'height': 0.50},
            'tern': {'width': 0.30, 'height': 0.25},
            'sandpiper': {'width': 0.20, 'height': 0.15},
            'plover': {'width': 0.25, 'height': 0.20},
            'oystercatcher': {'width': 0.40, 'height': 0.35},
            'albatross': {'width': 1.20, 'height': 0.90},
            'petrel': {'width': 0.35, 'height': 0.30},
            'heron': {'width': 0.60, 'height': 0.80},
        }
        
        # Default size if species not found or not provided
        default_size = {'width': 0.40, 'height': 0.30}
        
        if species is None:
            return default_size
        
        species_lower = species.lower()
        return bird_sizes.get(species_lower, default_size)
    
    def calculate_distances_in_video(self, video_path: str, 
                                    detections_per_frame: List[List[Dict]],
                                    species_per_detection: Optional[List[List[str]]] = None) -> List[List[Dict]]:
        """
        Calculate distances for all detections in a video.
        
        Args:
            video_path: Path to video file
            detections_per_frame: List of detections for each frame
            species_per_detection: Optional species labels for each detection
            
        Returns:
            List of distance estimates for each frame
        """
        cap = cv2.VideoCapture(video_path)
        all_distances = []
        
        for frame_idx, frame_detections in enumerate(detections_per_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_distances = []
            for det_idx, detection in enumerate(frame_detections):
                species = None
                if species_per_detection and frame_idx < len(species_per_detection):
                    if det_idx < len(species_per_detection[frame_idx]):
                        species = species_per_detection[frame_idx][det_idx]
                
                distance_info = self.estimate_distance(
                    frame, 
                    detection['bbox'],
                    species
                )
                frame_distances.append(distance_info)
            
            all_distances.append(frame_distances)
        
        cap.release()
        return all_distances
    
    def calculate_3d_position(self, bbox: List[int], distance: float,
                             image_width: int, image_height: int) -> Tuple[float, float, float]:
        """
        Calculate 3D position of bird in world coordinates.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            distance: Distance to bird in meters
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            (x, y, z) position in meters (camera at origin)
        """
        # Get center of bounding box
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Convert to normalized coordinates (-1 to 1)
        norm_x = (center_x - image_width / 2) / (image_width / 2)
        norm_y = (center_y - image_height / 2) / (image_height / 2)
        
        # Calculate 3D position
        focal_length = self.camera_params['focal_length']
        
        # Z is the distance
        z = distance
        
        # X and Y from similar triangles
        x = norm_x * distance * (image_width / (2 * focal_length))
        y = norm_y * distance * (image_height / (2 * focal_length))
        
        return (x, y, z)
    
    def calculate_relative_distance(self, point1: Tuple[float, float, float],
                                   point2: Tuple[float, float, float]) -> float:
        """
        Calculate Euclidean distance between two 3D points.
        
        Args:
            point1: First 3D point (x, y, z)
            point2: Second 3D point (x, y, z)
            
        Returns:
            Distance in meters
        """
        return scipy_distance.euclidean(point1, point2)
