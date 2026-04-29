"""
Simplified AI Clinical Trial Qualification System
A streamlined model for patient-trial matching
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
from enum import Enum


class EligibilityStatus(Enum):
    ELIGIBLE = "eligible"
    NOT_ELIGIBLE = "not_eligible"
    PARTIAL = "partial"


@dataclass
class PatientData:
    """Patient clinical information"""
    patient_id: str
    age: int
    conditions: List[str]
    medications: List[str]
    lab_values: Dict[str, float]
    comorbidities: List[str]


@dataclass
class TrialCriteria:
    """Clinical trial eligibility criteria"""
    trial_id: str
    trial_name: str
    inclusion_criteria: Dict[str, any]
    exclusion_criteria: Dict[str, any]
    required_conditions: List[str]


@dataclass
class QualificationResult:
    """Result of trial qualification assessment"""
    patient_id: str
    trial_id: str
    status: EligibilityStatus
    matching_score: float
    reasons: List[str]


class DataPreprocessor:
    """Stage 1: Clean and standardize patient data"""
    
    @staticmethod
    def preprocess(patient_data: PatientData) -> PatientData:
        """
        Clean patient data:
        - Normalize conditions (lowercase, remove duplicates)
        - Standardize medication names
        - Validate lab values
        """
        cleaned_data = PatientData(
            patient_id=patient_data.patient_id,
            age=patient_data.age,
            conditions=[c.lower().strip() for c in set(patient_data.conditions)],
            medications=[m.lower().strip() for m in set(patient_data.medications)],
            lab_values={k: v for k, v in patient_data.lab_values.items() if v >= 0},
            comorbidities=[c.lower().strip() for c in set(patient_data.comorbidities)]
        )
        return cleaned_data


class TrialMatcher:
    """Stage 2: Match patient against trial criteria"""
    
    @staticmethod
    def check_inclusion(patient: PatientData, trial: TrialCriteria) -> Tuple[bool, List[str]]:
        """Check if patient meets inclusion criteria"""
        reasons = []
        passed = True
        
        # Check required conditions
        patient_conditions = set(patient.conditions)
        required = set(trial.required_conditions)
        
        if not patient_conditions.intersection(required):
            passed = False
            reasons.append(f"Patient missing required conditions: {required}")
        
        # Check age criteria
        age_range = trial.inclusion_criteria.get("age_range", (0, 150))
        if not (age_range[0] <= patient.age <= age_range[1]):
            passed = False
            reasons.append(f"Age {patient.age} outside range {age_range}")
        
        # Check lab values
        lab_criteria = trial.inclusion_criteria.get("lab_values", {})
        for lab_name, (min_val, max_val) in lab_criteria.items():
            if lab_name in patient.lab_values:
                value = patient.lab_values[lab_name]
                if not (min_val <= value <= max_val):
                    passed = False
                    reasons.append(f"{lab_name}: {value} outside range [{min_val}, {max_val}]")
        
        return passed, reasons
    
    @staticmethod
    def check_exclusion(patient: PatientData, trial: TrialCriteria) -> Tuple[bool, List[str]]:
        """Check if patient meets exclusion criteria (returns True if excluded)"""
        reasons = []
        excluded = False
        
        # Check excluded conditions
        excluded_conditions = set(trial.exclusion_criteria.get("conditions", []))
        patient_conditions = set(patient.conditions)
        
        if patient_conditions.intersection(excluded_conditions):
            excluded = True
            reasons.append(f"Patient has excluded conditions: {patient_conditions.intersection(excluded_conditions)}")
        
        # Check excluded medications
        excluded_meds = set(trial.exclusion_criteria.get("medications", []))
        patient_meds = set(patient.medications)
        
        if patient_meds.intersection(excluded_meds):
            excluded = True
            reasons.append(f"Patient taking excluded medications: {patient_meds.intersection(excluded_meds)}")
        
        # Check comorbidities
        excluded_comorbidities = set(trial.exclusion_criteria.get("comorbidities", []))
        patient_comorbidities = set(patient.comorbidities)
        
        if patient_comorbidities.intersection(excluded_comorbidities):
            excluded = True
            reasons.append(f"Patient has excluded comorbidities: {patient_comorbidities.intersection(excluded_comorbidities)}")
        
        return excluded, reasons


class QualificationEngine:
    """Stage 3: Generate qualification result"""
    
    @staticmethod
    def qualify(patient: PatientData, trial: TrialCriteria) -> QualificationResult:
        """
        Main qualification logic:
        1. Check inclusion criteria
        2. Check exclusion criteria
        3. Calculate matching score
        4. Return result
        """
        
        # Check inclusion
        inclusion_passed, inclusion_reasons = TrialMatcher.check_inclusion(patient, trial)
        
        # Check exclusion
        is_excluded, exclusion_reasons = TrialMatcher.check_exclusion(patient, trial)
        
        # Determine status and score
        all_reasons = inclusion_reasons + exclusion_reasons
        
        if is_excluded:
            status = EligibilityStatus.NOT_ELIGIBLE
            score = 0.0
        elif inclusion_passed:
            status = EligibilityStatus.ELIGIBLE
            score = 1.0
        else:
            status = EligibilityStatus.PARTIAL
            score = 0.5
        
        result = QualificationResult(
            patient_id=patient.patient_id,
            trial_id=trial.trial_id,
            status=status,
            matching_score=score,
            reasons=all_reasons if all_reasons else ["All criteria met"]
        )
        
        return result


class TrialQualificationSystem:
    """Main system orchestrating the qualification process"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.engine = QualificationEngine()
    
    def assess_patient_trial_match(self, patient: PatientData, trial: TrialCriteria) -> QualificationResult:
        """
        Complete qualification pipeline:
        Stage 1: Preprocess patient data
        Stage 2: Match against criteria
        Stage 3: Generate result
        """
        # Stage 1: Preprocess
        cleaned_patient = self.preprocessor.preprocess(patient)
        
        # Stage 2 & 3: Qualify
        result = self.engine.qualify(cleaned_patient, trial)
        
        return result
    
    def batch_assess(self, patient: PatientData, trials: List[TrialCriteria]) -> List[QualificationResult]:
        """Assess patient against multiple trials"""
        results = [self.assess_patient_trial_match(patient, trial) for trial in trials]
        return sorted(results, key=lambda r: r.matching_score, reverse=True)


# Example usage
if __name__ == "__main__":
    # Initialize system
    system = TrialQualificationSystem()
    
    # Create sample patient
    patient = PatientData(
        patient_id="P001",
        age=55,
        conditions=["diabetes", "hypertension"],
        medications=["metformin", "lisinopril"],
        lab_values={"glucose": 145, "hemoglobin": 7.2},
        comorbidities=["obesity"]
    )
    
    # Create sample trials
    trial1 = TrialCriteria(
        trial_id="T001",
        trial_name="Diabetes Management Study",
        inclusion_criteria={
            "age_range": (40, 70),
            "lab_values": {"glucose": (130, 200), "hemoglobin": (6.5, 9.0)}
        },
        exclusion_criteria={
            "conditions": ["kidney_disease"],
            "medications": ["insulin"],
            "comorbidities": []
        },
        required_conditions=["diabetes"]
    )
    
    trial2 = TrialCriteria(
        trial_id="T002",
        trial_name="Hypertension Control Trial",
        inclusion_criteria={
            "age_range": (30, 75),
            "lab_values": {}
        },
        exclusion_criteria={
            "conditions": ["heart_failure"],
            "medications": [],
            "comorbidities": ["severe_obesity"]
        },
        required_conditions=["hypertension"]
    )
    
    # Run assessments
    result1 = system.assess_patient_trial_match(patient, trial1)
    result2 = system.assess_patient_trial_match(patient, trial2)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"Patient: {patient.patient_id}")
    print(f"{'='*60}")
    
    for result in [result1, result2]:
        print(f"\nTrial: {result.trial_id}")
        print(f"Status: {result.status.value}")
        print(f"Matching Score: {result.matching_score:.2%}")
        print(f"Reasons: {result.reasons}")
