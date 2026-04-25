"""
Sample Data Generator
Creates synthetic clinical trial patient data for testing and demonstration.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_sample_data(n_samples=500, random_state=42):
    """
    Generate synthetic clinical trial patient data.
    
    Args:
        n_samples: Number of patient records to generate
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with patient data and eligibility labels
    """
    np.random.seed(random_state)
    
    # Generate patient features
    data = {
        'patient_id': np.arange(1, n_samples + 1),
        'age': np.random.randint(18, 85, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'weight_kg': np.random.normal(75, 15, n_samples),
        'height_cm': np.random.normal(170, 10, n_samples),
        'blood_pressure_systolic': np.random.randint(90, 180, n_samples),
        'blood_pressure_diastolic': np.random.randint(60, 120, n_samples),
        'glucose_level': np.random.randint(70, 200, n_samples),
        'creatinine_level': np.random.uniform(0.5, 2.0, n_samples),
        'hemoglobin_level': np.random.uniform(10, 18, n_samples),
        'platelets_count': np.random.randint(150000, 450000, n_samples),
        'has_diabetes': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'has_hypertension': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'has_heart_disease': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'smoking_status': np.random.choice(['never', 'former', 'current'], n_samples),
        'alcohol_consumption': np.random.choice(['none', 'moderate', 'heavy'], n_samples),
        'previous_treatment': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    }
    
    df = pd.DataFrame(data)
    
    # Generate eligibility labels based on eligibility criteria
    # Eligibility criteria:
    # - Age between 30-75
    # - BMI between 18.5-30
    # - Blood pressure systolic < 160
    # - Glucose level < 150
    # - No serious heart disease
    # - Creatinine level < 1.5
    
    df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)
    
    eligible = (
        (df['age'] >= 30) & (df['age'] <= 75) &
        (df['bmi'] >= 18.5) & (df['bmi'] <= 30) &
        (df['blood_pressure_systolic'] < 160) &
        (df['glucose_level'] < 150) &
        (df['has_heart_disease'] == 0) &
        (df['creatinine_level'] < 1.5) &
        (df['hemoglobin_level'] >= 12)
    ).astype(int)
    
    df['eligible'] = eligible
    
    return df


def save_sample_data(filepath, n_samples=500):
    """Generate and save sample data to CSV."""
    df = generate_sample_data(n_samples=n_samples)
    df.to_csv(filepath, index=False)
    print(f"Sample data saved to {filepath}")
    print(f"Total samples: {len(df)}")
    print(f"Eligible patients: {df['eligible'].sum()}")
    print(f"Ineligible patients: {(1-df['eligible']).sum()}")
    return df


if __name__ == "__main__":
    save_sample_data('clinical_trial_data.csv', n_samples=500)
