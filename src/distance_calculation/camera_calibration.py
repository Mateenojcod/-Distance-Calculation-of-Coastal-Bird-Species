"""
Camera Calibration Module

Provides camera calibration utilities for accurate distance estimation.
Uses OpenCV's calibration functions with checkerboard patterns.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import pickle
import logging


class CameraCalibrator:
    """
    Camera calibration class for obtaining camera intrinsic and extrinsic parameters.
    
    These parameters are essential for accurate distance estimation.
    """
    
    def __init__(self, checkerboard_size: Tuple[int, int] = (9, 6),
                 square_size: float = 0.025):
        """
        Initialize camera calibrator.
        
        Args:
            checkerboard_size: Number of inner corners (width, height)
            square_size: Size of checkerboard squares in meters
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        
        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.calibrated = False
        
        # Prepare object points
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
    
    def calibrate_from_images(self, image_paths: List[str]) -> Dict:
        """
        Calibrate camera from multiple checkerboard images.
        
        Args:
            image_paths: List of paths to calibration images
            
        Returns:
            Dictionary containing calibration results
        """
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        image_size = None
        
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                logging.warning(f"Could not read image {img_path}")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if image_size is None:
                image_size = gray.shape[::-1]
            
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(
                gray, 
                self.checkerboard_size, 
                None
            )
            
            if ret:
                objpoints.append(self.objp)
                
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners_refined)
            else:
                logging.warning(f"Checkerboard not found in {img_path}")
        
        if len(objpoints) < 3:
            raise ValueError("Need at least 3 successful checkerboard detections for calibration")
        
        # Calibrate camera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None
        )
        
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.calibrated = True
        
        # Calculate reprojection error
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints_reproj, _ = cv2.projectPoints(
                objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
            )
            error = cv2.norm(imgpoints[i], imgpoints_reproj, cv2.NORM_L2) / len(imgpoints_reproj)
            total_error += error
        
        mean_error = total_error / len(objpoints)
        
        return {
            'success': ret,
            'camera_matrix': camera_matrix,
            'distortion_coefficients': dist_coeffs,
            'focal_length_x': camera_matrix[0, 0],
            'focal_length_y': camera_matrix[1, 1],
            'principal_point': (camera_matrix[0, 2], camera_matrix[1, 2]),
            'reprojection_error': mean_error,
            'num_images': len(objpoints)
        }
    
    def calibrate_from_video(self, video_path: str, 
                            num_frames: int = 20,
                            frame_interval: int = 30) -> Dict:
        """
        Calibrate camera from video of checkerboard.
        
        Args:
            video_path: Path to calibration video
            num_frames: Number of frames to use
            frame_interval: Interval between frames to sample
            
        Returns:
            Dictionary containing calibration results
        """
        cap = cv2.VideoCapture(video_path)
        
        objpoints = []
        imgpoints = []
        image_size = None
        
        frame_count = 0
        frames_used = 0
        
        while frames_used < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames according to interval
            if frame_count % frame_interval != 0:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if image_size is None:
                image_size = gray.shape[::-1]
            
            # Find checkerboard corners
            ret_corners, corners = cv2.findChessboardCorners(
                gray, 
                self.checkerboard_size, 
                None
            )
            
            if ret_corners:
                objpoints.append(self.objp)
                
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners_refined)
                
                frames_used += 1
                logging.info(f"Found checkerboard in frame {frame_count} ({frames_used}/{num_frames})")
        
        cap.release()
        
        if len(objpoints) < 3:
            raise ValueError(f"Only found {len(objpoints)} valid frames. Need at least 3.")
        
        # Calibrate
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None
        )
        
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.calibrated = True
        
        # Calculate reprojection error
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints_reproj, _ = cv2.projectPoints(
                objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
            )
            error = cv2.norm(imgpoints[i], imgpoints_reproj, cv2.NORM_L2) / len(imgpoints_reproj)
            total_error += error
        
        mean_error = total_error / len(objpoints)
        
        return {
            'success': ret,
            'camera_matrix': camera_matrix,
            'distortion_coefficients': dist_coeffs,
            'focal_length_x': camera_matrix[0, 0],
            'focal_length_y': camera_matrix[1, 1],
            'principal_point': (camera_matrix[0, 2], camera_matrix[1, 2]),
            'reprojection_error': mean_error,
            'num_images': len(objpoints)
        }
    
    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        Undistort an image using calibration parameters.
        
        Args:
            image: Input distorted image
            
        Returns:
            Undistorted image
        """
        if not self.calibrated:
            raise RuntimeError("Camera not calibrated yet")
        
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        
        # Undistort
        undistorted = cv2.undistort(
            image, 
            self.camera_matrix, 
            self.dist_coeffs, 
            None, 
            new_camera_matrix
        )
        
        # Crop the image
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]
        
        return undistorted
    
    def get_focal_length(self) -> Tuple[float, float]:
        """
        Get camera focal lengths in pixels.
        
        Returns:
            Tuple of (focal_length_x, focal_length_y)
        """
        if not self.calibrated:
            raise RuntimeError("Camera not calibrated yet")
        
        return (self.camera_matrix[0, 0], self.camera_matrix[1, 1])
    
    def save_calibration(self, filepath: str):
        """
        Save calibration parameters to file.
        
        Args:
            filepath: Path to save calibration file
        """
        if not self.calibrated:
            raise RuntimeError("Camera not calibrated yet")
        
        calibration_data = {
            'camera_matrix': self.camera_matrix,
            'dist_coeffs': self.dist_coeffs,
            'rvecs': self.rvecs,
            'tvecs': self.tvecs,
            'checkerboard_size': self.checkerboard_size,
            'square_size': self.square_size
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(calibration_data, f)
        
        logging.info(f"Calibration saved to {filepath}")
    
    def load_calibration(self, filepath: str):
        """
        Load calibration parameters from file.
        
        Args:
            filepath: Path to calibration file
        """
        with open(filepath, 'rb') as f:
            calibration_data = pickle.load(f)
        
        self.camera_matrix = calibration_data['camera_matrix']
        self.dist_coeffs = calibration_data['dist_coeffs']
        self.rvecs = calibration_data['rvecs']
        self.tvecs = calibration_data['tvecs']
        self.checkerboard_size = calibration_data['checkerboard_size']
        self.square_size = calibration_data['square_size']
        self.calibrated = True
        
        logging.info(f"Calibration loaded from {filepath}")
    
    def estimate_camera_params_from_fov(self, image_width: int, 
                                       image_height: int,
                                       horizontal_fov: float = 60.0) -> Dict:
        """
        Estimate camera parameters from field of view.
        Useful when calibration is not available.
        
        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels
            horizontal_fov: Horizontal field of view in degrees
            
        Returns:
            Dictionary with estimated camera parameters
        """
        # Calculate focal length from FOV
        focal_length = image_width / (2 * np.tan(np.radians(horizontal_fov / 2)))
        
        # Create camera matrix
        camera_matrix = np.array([
            [focal_length, 0, image_width / 2],
            [0, focal_length, image_height / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Assume no distortion for estimates
        dist_coeffs = np.zeros(5, dtype=np.float32)
        
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.calibrated = True
        
        return {
            'camera_matrix': camera_matrix,
            'distortion_coefficients': dist_coeffs,
            'focal_length': focal_length,
            'method': 'estimated_from_fov'
        }
