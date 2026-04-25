````markdown name=README.md
# Clinical Trial Qualification ML Model

A machine learning system that analyzes structured patient data and compares it with predefined eligibility criteria to classify patients as eligible or not eligible for clinical trials.

## 🎯 Overview

This project implements a comprehensive machine learning pipeline for clinical trial patient eligibility classification. It uses structured patient health data and applies data preprocessing, feature selection, and machine learning models to predict trial eligibility.

### Key Features

- **Data Preprocessing**: Cleaning, normalization, and handling missing values
- **Feature Selection**: Identifies important medical attributes using multiple techniques
- **Multiple Models**: Logistic Regression and Decision Tree classifiers
- **Complete Pipeline**: End-to-end workflow from raw data to predictions
- **Evaluation Metrics**: Comprehensive performance assessment
- **Model Persistence**: Save and load trained models

## 📊 Project Workflow

```
1. Data Collection (Healthcare datasets)
   ↓
2. Data Preprocessing (cleaning, normalization)
   ↓
3. Feature Selection (important medical attributes)
   ↓
4. Model Training (Logistic Regression / Decision Tree)
   ↓
5. Prediction (Eligibility classification)
```

## 🏗️ Project Structure

```
clinical-trial-qualification/
├── data/
│   ├── sample_data.py          # Sample data generator
│   └── clinical_trial_data.csv # Generated sample data
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data preprocessing module
│   ├── feature_selection.py    # Feature selection module
│   ├── model_training.py       # Model training module
│   └── predictor.py            # Complete prediction pipeline
├── models/
│   ├── trial_eligibility_lr.pkl    # Trained LR model
│   └── trial_eligibility_dt.pkl    # Trained DT model
├── notebooks/
│   └── example_usage.py        # Example usage script
├── tests/
│   ├── test_preprocessing.py   # Preprocessing tests
│   └── test_model_training.py  # Model training tests
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/PragyaJ23/clinical-trial-qualification.git
cd clinical-trial-qualification
```

2. Create virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📚 Module Documentation

### 1. Data Preprocessing (`src/data_preprocessing.py`)

Handles data cleaning and preparation:

```python
from src.data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
data = preprocessor.load_data('path/to/data.csv')
processed_data = preprocessor.preprocess(data, fit=True)
```

**Key Methods:**
- `load_data()`: Load data from CSV
- `handle_missing_values()`: Handle missing data
- `encode_categorical_features()`: Encode categorical variables
- `normalize_features()`: Normalize numeric features
- `preprocess()`: Complete pipeline

### 2. Feature Selection (`src/feature_selection.py`)

Identifies most important features:

```python
from src.feature_selection import FeatureSelector

selector = FeatureSelector(method='rf', k=15)
selected_features = selector.fit_transform(X, y)
```

**Supported Methods:**
- `f_classif`: ANOVA F-test
- `mutual_info`: Mutual information
- `rf`: Random Forest importance

### 3. Model Training (`src/model_training.py`)

Trains and evaluates models:

```python
from src.model_training import ModelTrainer

trainer = ModelTrainer(model_type='logistic_regression')
trainer.train(X, y, test_size=0.2, validation_split=True)
metrics = trainer.evaluate()
predictions = trainer.predict(X_new)
```

**Supported Models:**
- Logistic Regression
- Decision Tree Classifier

### 4. Prediction Pipeline (`src/predictor.py`)

Complete end-to-end workflow:

```python
from src.predictor import TrialEligibilityPredictor

predictor = TrialEligibilityPredictor(model_type='logistic_regression')
metrics = predictor.fit(X, y, feature_selection_k=12)
predictions = predictor.predict(X_new)
```

## 🚀 Quick Start

### 1. Generate Sample Data

```python
from data.sample_data import generate_sample_data

df = generate_sample_data(n_samples=500)
```

### 2. Train Model

```python
from src.predictor import TrialEligibilityPredictor
import pandas as pd

# Load data
df = pd.read_csv('data/clinical_trial_data.csv')
X = df.drop(['patient_id', 'eligible'], axis=1)
y = df['eligible']

# Create and train predictor
predictor = TrialEligibilityPredictor(model_type='logistic_regression')
metrics = predictor.fit(X, y, feature_selection_k=12)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1-Score: {metrics['f1']:.4f}")
```

### 3. Make Predictions

```python
# Predict for new patients
new_patients = X.iloc[:10]
predictions = predictor.predict(new_patients)

print(predictions)
# Output:
#    prediction eligible  probability_not_eligible  probability_eligible
# 0           1      True                    0.1234                0.8766
# 1           0     False                    0.7891                0.2109
```

### 4. Save and Load Model

```python
# Save model
predictor.save_pipeline('models/my_model.pkl')

# Load model
predictor.load_pipeline('models/my_model.pkl')
predictions = predictor.predict(new_patients)
```

## 📊 Model Performance

### Evaluation Metrics

Models are evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Classification breakdown
- **Classification Report**: Detailed per-class metrics

### Cross-Validation

5-fold cross-validation is performed during training for robust evaluation.

## 🧪 Testing

Run tests using pytest:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_preprocessing.py

# Run with verbose output
pytest -v tests/
```

## 📈 Sample Patient Data

The sample data generator creates synthetic patient records with features:

- **Demographics**: Age, Gender
- **Vitals**: Weight, Height, Blood Pressure
- **Labs**: Glucose, Creatinine, Hemoglobin, Platelets
- **Medical History**: Diabetes, Hypertension, Heart Disease
- **Lifestyle**: Smoking Status, Alcohol Consumption

### Eligibility Criteria

Patients are considered eligible if:
- Age between 30-75 years
- BMI between 18.5-30
- Systolic BP < 160 mmHg
- Glucose level < 150 mg/dL
- No serious heart disease
- Creatinine level < 1.5 mg/dL
- Hemoglobin level ≥ 12 g/dL

## 🔮 Future Enhancements

- Integration of NLP for processing clinical notes
- Advanced deep learning models (Neural Networks, LSTM)
- Hyperparameter optimization (GridSearch, RandomSearch)
- Ensemble methods combining multiple models
- Real-time prediction API
- Interactive dashboard for predictions
- Model explainability (SHAP, LIME)
- Handling imbalanced datasets with SMOTE
- Cross-validation with stratified folds
- Performance monitoring and drift detection

## 📝 Example Usage Script

See `notebooks/example_usage.py` for complete workflow example.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**Pragya J** - PragyaJ23

## 📞 Support

For issues, questions, or suggestions, please open an issue on GitHub.

## 🙏 Acknowledgments

- scikit-learn for ML algorithms
- pandas and numpy for data manipulation
- Healthcare datasets community

---

**Disclaimer**: This model is for educational and research purposes. It should not be used for actual clinical trial decisions without proper validation by medical professionals and regulatory review.
````
