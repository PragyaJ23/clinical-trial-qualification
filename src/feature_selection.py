"""
Feature Selection Module
Identifies important medical attributes for model training.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureSelector:
    """Selects most important features for clinical trial qualification."""
    
    def __init__(self, method='f_classif', k=10):
        """
        Initialize feature selector.
        
        Args:
            method: 'f_classif', 'mutual_info', or 'rf' (Random Forest)
            k: Number of features to select
        """
        self.method = method
        self.k = k
        self.selector = None
        self.selected_features = None
        self.feature_importance = None
        
    def select_kbest(self, X, y):
        """Select K best features using f_classif or mutual information."""
        if self.method == 'f_classif':
            self.selector = SelectKBest(score_func=f_classif, k=min(self.k, X.shape[1]))
        elif self.method == 'mutual_info':
            self.selector = SelectKBest(score_func=mutual_info_classif, k=min(self.k, X.shape[1]))
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.selector.fit(X, y)
        self.selected_features = X.columns[self.selector.get_support()].tolist()
        
        logger.info(f"Selected {len(self.selected_features)} features using {self.method}")
        return self.selected_features
    
    def select_random_forest(self, X, y, n_estimators=100):
        """Select features using Random Forest feature importance."""
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        self.selected_features = feature_importance_df.head(self.k)['feature'].tolist()
        self.feature_importance = feature_importance_df
        
        logger.info(f"Selected {len(self.selected_features)} features using Random Forest")
        return self.selected_features
    
    def select(self, X, y):
        """Select features based on configured method."""
        if self.method == 'rf':
            return self.select_random_forest(X, y)
        else:
            return self.select_kbest(X, y)
    
    def transform(self, X):
        """Transform data to selected features only."""
        if self.selected_features is None:
            raise ValueError("Features not selected yet. Call select() first.")
        
        return X[self.selected_features]
    
    def fit_transform(self, X, y):
        """Fit selector and transform data."""
        self.select(X, y)
        return self.transform(X)
    
    def get_feature_importance(self):
        """Get feature importance scores."""
        if self.feature_importance is not None:
            return self.feature_importance
        
        if self.selector is not None:
            scores = self.selector.scores_
            feature_importance_df = pd.DataFrame({
                'feature': self.selector.get_feature_names_out(),
                'score': scores
            }).sort_values('score', ascending=False)
            return feature_importance_df
        
        return None
    
    def get_selected_features(self):
        """Get list of selected features."""
        return self.selected_features
