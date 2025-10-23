"""
Machine Learning analyzer for Forestry Carbon ARR analysis.
"""

import logging
from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd

from ..exceptions import MLError, DependencyError
from ..utils.dependency_manager import DependencyManager

logger = logging.getLogger(__name__)


class MLAnalyzer:
    """
    Machine Learning analyzer for forestry carbon analysis.
    
    This class provides ML capabilities for land cover classification,
    carbon stock estimation, and other ML-based analysis tasks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ML analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.config = config
        self.dependency_manager = DependencyManager()
        
        # Check if ML dependencies are available
        if not self.dependency_manager.is_ml_available():
            raise DependencyError("ML dependencies not available. Please install scikit-learn, tensorflow, or torch.")
        
        # Initialize ML libraries
        self._initialize_ml_libraries()
    
    def _initialize_ml_libraries(self) -> None:
        """Initialize machine learning libraries."""
        try:
            import sklearn
            self.sklearn_available = True
            self.logger.info("scikit-learn available")
        except ImportError:
            self.sklearn_available = False
            self.logger.warning("scikit-learn not available")
        
        try:
            import tensorflow as tf
            self.tensorflow_available = True
            self.logger.info("TensorFlow available")
        except ImportError:
            self.tensorflow_available = False
            self.logger.warning("TensorFlow not available")
        
        try:
            import torch
            self.pytorch_available = True
            self.logger.info("PyTorch available")
        except ImportError:
            self.pytorch_available = False
            self.logger.warning("PyTorch not available")
    
    def classify_landcover(self, 
                          features: np.ndarray,
                          labels: np.ndarray,
                          algorithm: str = 'gbm',
                          test_size: float = 0.2) -> Dict[str, Any]:
        """
        Perform land cover classification.
        
        Args:
            features: Feature matrix
            labels: Target labels
            algorithm: ML algorithm to use
            test_size: Test set size ratio
            
        Returns:
            Classification results
        """
        if not self.sklearn_available:
            raise DependencyError("scikit-learn required for classification")
        
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
            from sklearn.svm import SVC
            from sklearn.metrics import classification_report, accuracy_score
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=test_size, random_state=42
            )
            
            # Initialize classifier
            if algorithm.lower() == 'gbm':
                classifier = GradientBoostingClassifier(random_state=42)
            elif algorithm.lower() == 'random_forest':
                classifier = RandomForestClassifier(random_state=42)
            elif algorithm.lower() == 'svm':
                classifier = SVC(random_state=42)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            # Train classifier
            classifier.fit(X_train, y_train)
            
            # Make predictions
            y_pred = classifier.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            results = {
                'classifier': classifier,
                'accuracy': accuracy,
                'classification_report': report,
                'predictions': y_pred,
                'test_labels': y_test
            }
            
            self.logger.info(f"Classification completed with {algorithm}: {accuracy:.3f} accuracy")
            return results
            
        except Exception as e:
            raise MLError(f"Classification failed: {e}")
    
    def estimate_carbon_stock(self, 
                            features: np.ndarray,
                            carbon_values: np.ndarray,
                            algorithm: str = 'random_forest') -> Dict[str, Any]:
        """
        Estimate carbon stock using regression.
        
        Args:
            features: Feature matrix
            carbon_values: Carbon stock values
            algorithm: ML algorithm to use
            
        Returns:
            Regression results
        """
        if not self.sklearn_available:
            raise DependencyError("scikit-learn required for regression")
        
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.svm import SVR
            from sklearn.metrics import mean_squared_error, r2_score
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, carbon_values, test_size=0.2, random_state=42
            )
            
            # Initialize regressor
            if algorithm.lower() == 'random_forest':
                regressor = RandomForestRegressor(random_state=42)
            elif algorithm.lower() == 'gbm':
                regressor = GradientBoostingRegressor(random_state=42)
            elif algorithm.lower() == 'svr':
                regressor = SVR()
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            # Train regressor
            regressor.fit(X_train, y_train)
            
            # Make predictions
            y_pred = regressor.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results = {
                'regressor': regressor,
                'mse': mse,
                'r2_score': r2,
                'predictions': y_pred,
                'test_values': y_test
            }
            
            self.logger.info(f"Carbon stock estimation completed with {algorithm}: R² = {r2:.3f}")
            return results
            
        except Exception as e:
            raise MLError(f"Carbon stock estimation failed: {e}")
    
    def get_feature_importance(self, model: Any) -> Dict[str, float]:
        """
        Get feature importance from trained model.
        
        Args:
            model: Trained ML model
            
        Returns:
            Dictionary with feature importance
        """
        try:
            if hasattr(model, 'feature_importances_'):
                return dict(zip(range(len(model.feature_importances_)), model.feature_importances_))
            else:
                self.logger.warning("Model does not support feature importance")
                return {}
        except Exception as e:
            self.logger.error(f"Failed to get feature importance: {e}")
            return {}
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get ML analyzer system information.
        
        Returns:
            Dictionary with system information
        """
        return {
            'sklearn_available': self.sklearn_available,
            'tensorflow_available': self.tensorflow_available,
            'pytorch_available': self.pytorch_available,
            'config': self.config.get('ml', {})
        }
