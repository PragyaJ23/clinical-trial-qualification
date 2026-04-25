"""
Model Training Module
Trains Logistic Regression and Decision Tree models for eligibility classification.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, f1_score)
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and evaluates eligibility classification models."""
    
    def __init__(self, model_type='logistic_regression', random_state=42):
        """
        Initialize model trainer.
        
        Args:
            model_type: 'logistic_regression' or 'decision_tree'
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.train_test_split_data = None
        self.evaluation_metrics = None
        
    def _initialize_model(self):
        """Initialize the model based on model_type."""
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                solver='lbfgs',
                n_jobs=-1
            )
            logger.info("Initialized Logistic Regression model")
        elif self.model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state
            )
            logger.info("Initialized Decision Tree model")
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X, y, test_size=0.2, validation_split=False):
        """
        Train the model.
        
        Args:
            X: Features
            y: Target variable
            test_size: Proportion of data for testing
            validation_split: Whether to use cross-validation
        """
        logger.info(f"Starting training with {self.model_type}")
        
        # Initialize model
        self._initialize_model()
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        self.train_test_split_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        
        # Train model
        self.model.fit(X_train, y_train)
        logger.info(f"Model training completed. Training set size: {len(X_train)}")
        
        # Cross-validation (optional)
        if validation_split:
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='f1_weighted')
            logger.info(f"Cross-validation scores: {cv_scores}")
            logger.info(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return self
    
    def evaluate(self):
        """Evaluate model performance on test set."""
        if self.model is None or self.train_test_split_data is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X_test = self.train_test_split_data['X_test']
        y_test = self.train_test_split_data['y_test']
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self.evaluation_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, zero_division=0)
        }
        
        logger.info(f"Model Evaluation Results:")
        logger.info(f"Accuracy: {self.evaluation_metrics['accuracy']:.4f}")
        logger.info(f"Precision: {self.evaluation_metrics['precision']:.4f}")
        logger.info(f"Recall: {self.evaluation_metrics['recall']:.4f}")
        logger.info(f"F1-Score: {self.evaluation_metrics['f1']:.4f}")
        
        return self.evaluation_metrics
    
    def predict(self, X):
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'eligible': predictions == 1
        }
    
    def save_model(self, filepath):
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load pre-trained model from disk."""
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return self
    
    def get_metrics(self):
        """Get evaluation metrics."""
        return self.evaluation_metrics
    
    def get_model_info(self):
        """Get model information."""
        return {
            'type': self.model_type,
            'model': self.model,
            'parameters': self.model.get_params() if self.model else None,
            'metrics': self.evaluation_metrics
        }
