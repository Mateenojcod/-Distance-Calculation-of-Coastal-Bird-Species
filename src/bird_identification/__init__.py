"""
Bird Identification Module
Provides machine learning-based bird species classification from camera trap footage.
"""

from .classifier import BirdClassifier
from .detector import BirdDetector

__all__ = ['BirdClassifier', 'BirdDetector']
