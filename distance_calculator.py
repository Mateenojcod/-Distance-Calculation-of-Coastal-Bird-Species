"""
Distance Calculator for Coastal Bird Species
Calculates the distance of objects in video footage using computer vision techniques.
"""

import cv2
import numpy as np


class DistanceCalculator:
    """
    A class to calculate the distance of objects from the camera in video footage.
    
    Uses the pinhole camera model and known object dimensions to estimate distance.
    Formula: Distance = (Known_Width * Focal_Length) / Perceived_Width
    """
    
    def __init__(self, known_width, focal_length, frame_width=1920):
        """
        Initialize the distance calculator.
        
        Args:
            known_width (float): The actual width of the object in metres
            focal_length (float): The focal length of the camera in pixels
            frame_width (int): The width of the video frame in pixels (default: 1920)
        """
        self.known_width = known_width
        self.focal_length = focal_length
        self.frame_width = frame_width
        
    def calculate_distance(self, perceived_width):
        """
        Calculate the distance to an object based on its perceived width in pixels.
        
        Args:
            perceived_width (float): The width of the object in the frame (pixels)
            
        Returns:
            float: The estimated distance in metres
        """
        if perceived_width == 0:
            return 0
        
        distance = (self.known_width * self.focal_length) / perceived_width
        return distance
    
    def calibrate_focal_length(self, known_distance, perceived_width):
        """
        Calibrate the focal length of the camera using a known distance and perceived width.
        
        Args:
            known_distance (float): The actual distance to the object in metres
            perceived_width (float): The width of the object in the frame (pixels)
            
        Returns:
            float: The calculated focal length
        """
        self.focal_length = (perceived_width * known_distance) / self.known_width
        return self.focal_length


class VideoDistanceProcessor:
    """
    Process video files and calculate distances to detected objects.
    """
    
    # Default detection parameters
    DEFAULT_MIN_CONTOUR_AREA = 500
    DEFAULT_HISTORY = 500
    DEFAULT_VAR_THRESHOLD = 50
    
    def __init__(self, distance_calculator, min_contour_area=None, 
                 bg_history=None, bg_var_threshold=None):
        """
        Initialize the video processor.
        
        Args:
            distance_calculator (DistanceCalculator): An instance of DistanceCalculator
            min_contour_area (int, optional): Minimum contour area to filter noise (default: 500)
            bg_history (int, optional): Background subtractor history parameter (default: 500)
            bg_var_threshold (int, optional): Background subtractor variance threshold (default: 50)
        """
        self.calculator = distance_calculator
        self.min_contour_area = min_contour_area or self.DEFAULT_MIN_CONTOUR_AREA
        bg_history = bg_history or self.DEFAULT_HISTORY
        bg_var_threshold = bg_var_threshold or self.DEFAULT_VAR_THRESHOLD
        
        self.detector = cv2.createBackgroundSubtractorMOG2(
            history=bg_history, 
            varThreshold=bg_var_threshold, 
            detectShadows=True
        )
        
    def detect_object_width(self, frame):
        """
        Detect objects in the frame and return the width of the largest contour.
        
        Args:
            frame: Video frame (numpy array)
            
        Returns:
            tuple: (width in pixels, bounding box coordinates, processed frame)
        """
        # Apply background subtraction
        fg_mask = self.detector.apply(frame)
        
        # Remove shadows
        fg_mask[fg_mask == 127] = 0
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return 0, None, frame
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Filter out small contours (noise)
        if cv2.contourArea(largest_contour) < self.min_contour_area:
            return 0, None, frame
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        return w, (x, y, w, h), frame
    
    def process_video(self, video_path, output_path=None, display=True):
        """
        Process a video file and calculate distances to detected objects.
        
        Args:
            video_path (str): Path to the input video file
            output_path (str, optional): Path to save the output video
            display (bool): Whether to display the video during processing
            
        Returns:
            list: List of tuples (frame_number, distance in metres)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        distances = []
        frame_count = 0
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Detect object and get its width
            obj_width, bbox, processed_frame = self.detect_object_width(frame)
            
            # Calculate distance
            distance = 0
            if obj_width > 0:
                distance = self.calculator.calculate_distance(obj_width)
                distances.append((frame_count, distance))
                
                # Draw bounding box and distance on frame
                if bbox:
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Display distance
                    text = f"Distance: {distance:.2f}m"
                    cv2.putText(
                        frame, 
                        text, 
                        (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (0, 255, 0), 
                        2
                    )
            
            # Add frame counter
            cv2.putText(
                frame, 
                f"Frame: {frame_count}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
            
            # Write frame to output video
            if writer:
                writer.write(frame)
            
            # Display frame
            if display:
                cv2.imshow('Distance Calculation', frame)
                
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        
        print(f"\nProcessing complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Objects detected in {len(distances)} frames")
        
        return distances
    
    def process_frame(self, frame):
        """
        Process a single frame and calculate distance.
        
        Args:
            frame: Video frame (numpy array)
            
        Returns:
            tuple: (distance in metres, annotated frame)
        """
        obj_width, bbox, _ = self.detect_object_width(frame)
        
        distance = 0
        if obj_width > 0:
            distance = self.calculator.calculate_distance(obj_width)
            
            # Draw bounding box and distance
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                text = f"Distance: {distance:.2f}m"
                cv2.putText(
                    frame, 
                    text, 
                    (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 255, 0), 
                    2
                )
        
        return distance, frame


def main():
    """
    Example usage of the distance calculator.
    """
    # Example: Calibration for a typical coastal bird
    # Assume average bird width is 0.3 meters (30 cm)
    # This should be adjusted based on the specific bird species
    
    print("Distance Calculator for Coastal Bird Species")
    print("=" * 50)
    
    # Initialize the calculator
    # These values should be calibrated for your specific camera setup
    known_width = 0.3  # metres (average bird width)
    focal_length = 700  # pixels (should be calibrated)
    
    calculator = DistanceCalculator(known_width, focal_length)
    
    print(f"\nCalculator initialized:")
    print(f"  Known object width: {known_width}m")
    print(f"  Focal length: {focal_length}px")
    
    # Example: Calculate distance for different perceived widths
    print("\nExample distance calculations:")
    for perceived_width in [100, 150, 200, 250, 300]:
        distance = calculator.calculate_distance(perceived_width)
        print(f"  Perceived width: {perceived_width}px -> Distance: {distance:.2f}m")
    
    # Example: Calibrate focal length
    print("\nCalibration example:")
    print("  If you know an object is 5m away and appears 420px wide:")
    calibrated_focal_length = calculator.calibrate_focal_length(5.0, 420)
    print(f"  Calibrated focal length: {calibrated_focal_length:.2f}px")


if __name__ == "__main__":
    main()
