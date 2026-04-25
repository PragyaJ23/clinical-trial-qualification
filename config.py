"""
Configuration file for Clinical Trial Qualification Model
"""

# Model Configuration
MODEL_TYPES = ['logistic_regression', 'decision_tree']
DEFAULT_MODEL = 'logistic_regression'

# Training Configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Feature Selection
DEFAULT_FEATURE_SELECTION_METHOD = 'rf'  # 'f_classif', 'mutual_info', 'rf'
DEFAULT_K_FEATURES = 15

# Data Preprocessing
MISSING_VALUE_STRATEGY = 'mean'  # 'mean', 'median', 'most_frequent'
NORMALIZE_FEATURES = True

# Model Parameters
LOGISTIC_REGRESSION_PARAMS = {
    'max_iter': 1000,
    'solver': 'lbfgs',
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

DECISION_TREE_PARAMS = {
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE
}

# Paths
DATA_DIR = 'data/'
MODEL_DIR = 'models/'
LOGS_DIR = 'logs/'

# Eligibility Criteria Thresholds
ELIGIBILITY_CRITERIA = {
    'age_min': 30,
    'age_max': 75,
    'bmi_min': 18.5,
    'bmi_max': 30,
    'systolic_bp_max': 160,
    'glucose_max': 150,
    'creatinine_max': 1.5,
    'hemoglobin_min': 12,
    'has_heart_disease': False,
}

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Sample Data Configuration
SAMPLE_DATA_SIZE = 500
SAMPLE_DATA_RANDOM_STATE = 42
