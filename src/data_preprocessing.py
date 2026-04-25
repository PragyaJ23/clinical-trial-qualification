"""
Data Preprocessing Module
Handles data cleaning, normalization, and preparation for model training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocesses clinical trial patient data."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.label_encoders = {}
        self.is_fitted = False
        
    def load_data(self, filepath):
        """Load data from CSV file."""
        try:
            data = pd.read_csv(filepath)
            logger.info(f"Data loaded successfully. Shape: {data.shape}")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
    
    def handle_missing_values(self, data, strategy='mean'):
        """Handle missing values using specified strategy."""
        logger.info(f"Handling missing values with strategy: {strategy}")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        # Handle numeric columns
        if len(numeric_cols) > 0:
            data[numeric_cols] = self.imputer.fit_transform(data[numeric_cols])
        
        # Handle categorical columns
        for col in categorical_cols:
            data[col].fillna(data[col].mode()[0] if len(data[col].mode()) > 0 else 'Unknown', inplace=True)
        
        logger.info(f"Missing values handled. Remaining nulls: {data.isnull().sum().sum()}")
        return data
    
    def encode_categorical_features(self, data, fit=False):
        """Encode categorical features using LabelEncoder."""
        logger.info("Encoding categorical features")
        
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                data[col] = self.label_encoders[col].fit_transform(data[col].astype(str))
            else:
                if col in self.label_encoders:
                    data[col] = self.label_encoders[col].transform(data[col].astype(str))
        
        logger.info(f"Encoded {len(categorical_cols)} categorical features")
        return data
    
    def normalize_features(self, data, fit=False):
        """Normalize numeric features using StandardScaler."""
        logger.info("Normalizing numeric features")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if fit:
            data[numeric_cols] = self.scaler.fit_transform(data[numeric_cols])
            self.is_fitted = True
        else:
            data[numeric_cols] = self.scaler.transform(data[numeric_cols])
        
        logger.info(f"Normalized {len(numeric_cols)} numeric features")
        return data
    
    def preprocess(self, data, fit=False):
        """Complete preprocessing pipeline."""
        logger.info("Starting data preprocessing pipeline")
        
        # Step 1: Handle missing values
        data = self.handle_missing_values(data)
        
        # Step 2: Encode categorical features
        data = self.encode_categorical_features(data, fit=fit)
        
        # Step 3: Normalize features
        data = self.normalize_features(data, fit=fit)
        
        logger.info("Preprocessing pipeline completed")
        return data
    
    def get_feature_info(self, data):
        """Get information about features."""
        return {
            'total_features': data.shape[1],
            'total_samples': data.shape[0],
            'numeric_features': len(data.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(data.select_dtypes(include=['object']).columns)
        }
