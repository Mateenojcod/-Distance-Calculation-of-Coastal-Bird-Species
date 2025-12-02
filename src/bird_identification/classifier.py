"""
Bird Species Classifier using Multiple ML Approaches

This module provides multiple classification algorithms for coastal bird species:
1. Convolutional Neural Networks (CNNs) - TensorFlow/Keras
2. Transfer Learning - Pre-trained models (ResNet, EfficientNet, etc.)
3. Traditional ML - SVM, Random Forest via scikit-learn
4. PyTorch-based models - Using timm library
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import cv2

# TensorFlow/Keras approach
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications import (
        ResNet50, EfficientNetB0, MobileNetV2, VGG16
    )
    from tensorflow.keras.preprocessing import image as keras_image
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# PyTorch approach
try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms
    import timm  # PyTorch Image Models
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Scikit-learn approach
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class BirdClassifier:
    """
    Main classifier class supporting multiple ML backends.
    
    Supported Methods:
    - 'tensorflow': Uses TensorFlow/Keras with transfer learning
    - 'pytorch': Uses PyTorch with timm models
    - 'sklearn': Uses traditional ML (SVM, Random Forest)
    - 'ensemble': Combines multiple models for better accuracy
    """
    
    def __init__(self, method: str = 'tensorflow', num_classes: int = 10):
        """
        Initialize the classifier.
        
        Args:
            method: Classification method ('tensorflow', 'pytorch', 'sklearn', 'ensemble')
            num_classes: Number of bird species to classify
        """
        self.method = method
        self.num_classes = num_classes
        self.model = None
        self.class_names = []
        
        if method == 'tensorflow' and TENSORFLOW_AVAILABLE:
            self.model = self._build_tensorflow_model()
        elif method == 'pytorch' and PYTORCH_AVAILABLE:
            self.model = self._build_pytorch_model()
        elif method == 'sklearn' and SKLEARN_AVAILABLE:
            self.model = self._build_sklearn_model()
        elif method == 'ensemble':
            self.models = self._build_ensemble_models()
        else:
            raise ValueError(f"Method {method} not supported or dependencies not installed")
    
    def _build_tensorflow_model(self) -> Model:
        """
        Build a CNN classifier using TensorFlow/Keras with transfer learning.
        Uses EfficientNetB0 as base model (good for bird classification).
        """
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_pytorch_model(self) -> nn.Module:
        """
        Build a classifier using PyTorch with timm library.
        Uses pre-trained models optimized for fine-grained classification.
        """
        # Using timm for advanced pre-trained models
        model = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=self.num_classes
        )
        
        # Freeze early layers
        for param in list(model.parameters())[:-10]:
            param.requires_grad = False
        
        return model
    
    def _build_sklearn_model(self) -> Dict:
        """
        Build traditional ML classifiers using scikit-learn.
        Returns dictionary of multiple models.
        """
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            )
        }
        return models
    
    def _build_ensemble_models(self) -> Dict:
        """
        Build ensemble of multiple models for better accuracy.
        """
        ensemble = {}
        if TENSORFLOW_AVAILABLE:
            ensemble['tensorflow'] = self._build_tensorflow_model()
        if PYTORCH_AVAILABLE:
            ensemble['pytorch'] = self._build_pytorch_model()
        if SKLEARN_AVAILABLE:
            ensemble['sklearn'] = self._build_sklearn_model()
        return ensemble
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for the selected model.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        if self.method == 'tensorflow':
            # Resize to 224x224
            img = cv2.resize(image, (224, 224))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)
            return img
        
        elif self.method == 'pytorch':
            # PyTorch preprocessing
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            img = transform(image)
            img = img.unsqueeze(0)  # Add batch dimension
            return img
        
        elif self.method == 'sklearn':
            # Feature extraction for traditional ML
            img = cv2.resize(image, (64, 64))
            img = img.flatten() / 255.0
            return img.reshape(1, -1)
        
        return image
    
    def predict(self, image: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict bird species from image.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            Tuple of (predicted_class, confidence, all_probabilities)
        """
        # Preprocess
        processed_image = self.preprocess_image(image)
        
        if self.method == 'tensorflow':
            predictions = self.model.predict(processed_image, verbose=0)[0]
            
        elif self.method == 'pytorch':
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(processed_image)
                predictions = torch.softmax(outputs, dim=1)[0].numpy()
        
        elif self.method == 'sklearn':
            # Use best model from sklearn ensemble
            model = self.model.get('random_forest')
            if model is None:
                raise RuntimeError("Random Forest model not found in sklearn ensemble")
            predictions = model.predict_proba(processed_image)[0]
        
        elif self.method == 'ensemble':
            # Average predictions from all models
            all_preds = []
            if 'tensorflow' in self.models:
                pred = self.models['tensorflow'].predict(processed_image, verbose=0)[0]
                all_preds.append(pred)
            # Add more ensemble predictions...
            predictions = np.mean(all_preds, axis=0)
        
        # Get top prediction
        top_idx = np.argmax(predictions)
        confidence = float(predictions[top_idx])
        
        # Create probability dictionary
        prob_dict = {
            self.class_names[i] if i < len(self.class_names) else f"class_{i}": 
            float(predictions[i]) 
            for i in range(len(predictions))
        }
        
        predicted_class = self.class_names[top_idx] if top_idx < len(self.class_names) else f"class_{top_idx}"
        
        return predicted_class, confidence, prob_dict
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 50, batch_size: int = 32):
        """
        Train the classifier.
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        if self.method == 'tensorflow':
            validation_data = (X_val, y_val) if X_val is not None else None
            history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
            return history
        
        elif self.method == 'sklearn':
            # Train all sklearn models
            for name, model in self.model.items():
                print(f"Training {name}...")
                model.fit(X_train, y_train)
        
        elif self.method == 'pytorch':
            # PyTorch training loop would go here
            pass
    
    def save_model(self, path: str):
        """Save the trained model."""
        if self.method == 'tensorflow':
            self.model.save(path)
        elif self.method == 'pytorch':
            torch.save(self.model.state_dict(), path)
        elif self.method == 'sklearn':
            import joblib
            joblib.dump(self.model, path)
    
    def load_model(self, path: str):
        """Load a trained model."""
        if self.method == 'tensorflow':
            self.model = keras.models.load_model(path)
        elif self.method == 'pytorch':
            self.model.load_state_dict(torch.load(path))
        elif self.method == 'sklearn':
            import joblib
            self.model = joblib.load(path)
    
    def set_class_names(self, names: List[str]):
        """Set the class names for prediction output."""
        self.class_names = names


# Example coastal bird species that could be classified
COASTAL_BIRD_SPECIES = [
    "Seagull",
    "Pelican",
    "Cormorant",
    "Tern",
    "Sandpiper",
    "Plover",
    "Oystercatcher",
    "Albatross",
    "Petrel",
    "Heron"
]
