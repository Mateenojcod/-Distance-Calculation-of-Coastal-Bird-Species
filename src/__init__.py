"""
Coastal Bird Species Identification and Distance Calculation

A comprehensive system for identifying bird species and calculating distances
from camera trap footage using multiple machine learning approaches.
"""

__version__ = "1.0.0"
__author__ = "Coastal Bird Research Team"

from .bird_identification import BirdDetector, BirdClassifier
from .distance_calculation import DistanceEstimator, CameraCalibrator

__all__ = [
    'BirdDetector',
    'BirdClassifier', 
    'DistanceEstimator',
    'CameraCalibrator'
]
