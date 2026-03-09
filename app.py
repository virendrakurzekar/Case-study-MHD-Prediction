
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conint
import joblib
import numpy as np
from typing import List
from scipy.stats import ks_2samp

# FastAPI App
app = FastAPI(title="Mental Health Prediction API", version="1.0")

MODEL_PATH = "final_dass_system.pkl"
REFERENCE_PATH = "reference_stats.pkl"

try:
    bundle = joblib.load(MODEL_PATH)
except FileNotFoundError:
    bundle = None

# Input Schema
class DASSAssessment(BaseModel):
    answers: List[conint(ge=0, le=3)] = Field(..., min_length=42, max_length=42)
    education: int
    urban: int
    gender: int
    engnat: int
    age: int
    screensize: float
    religion: int
    orientation: int
    race: int
    married: int
    familysize: int


# Output Schema
class DASSResponse(BaseModel):
    depression_level: int
    anxiety_level: int
    stress_level: int
    drift_detected: bool


# Question index mapping
DEP_IDX = [2,4,9,12,15,16,20,23,25,30,33,36,37,41]
ANX_IDX = [1,3,6,8,14,18,19,22,24,27,29,35,39,40]
STR_IDX = [0,5,7,10,11,13,17,21,26,28,31,32,34,38]

# Drift Monitoring Class
class DriftMonitor:

    def __init__(self, reference_path: str, psi_threshold: float = 0.25):
        try:
            self.reference = joblib.load(reference_path)
        except Exception:
            self.reference = None
        self.psi_threshold = psi_threshold

    # Calculate PSI
    def calculate_psi(self, expected, actual, bins=10):

        min_val = min(np.min(expected), np.min(actual))
        max_val = max(np.max(expected), np.max(actual))

        bins_edges = np.linspace(min_val, max_val, bins + 1)

        expected_counts, _ = np.histogram(expected, bins=bins_edges)
        actual_counts, _ = np.histogram(actual, bins=bins_edges)

        expected_perc = expected_counts / (len(expected) + 1e-6)
        actual_perc = actual_counts / (len(actual) + 1e-6)

        psi = np.sum(
            (actual_perc - expected_perc)
            * np.log((actual_perc + 1e-6) / (expected_perc + 1e-6))
        )

        return psi

    # Check if drift exists
    def check_drift(self, train_feature, new_feature):
        psi_value = self.calculate_psi(train_feature, new_feature)
        _, p_value = ks_2samp(train_feature, new_feature)
        drift_detected = psi_value > self.psi_threshold or p_value < 0.05

        return drift_detected

drift_monitor = DriftMonitor(REFERENCE_PATH)


# Prediction Endpoint
# -----------------------------
@app.post("/predict", response_model=DASSResponse)
def predict(payload: DASSAssessment):

    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    dv = [payload.education, payload.urban, payload.gender, payload.engnat,
          payload.age, payload.screensize, payload.religion, payload.orientation,
          payload.race, payload.married, payload.familysize]

    dep_answers = [payload.answers[i] for i in DEP_IDX]
    anx_answers = [payload.answers[i] for i in ANX_IDX]
    str_answers = [payload.answers[i] for i in STR_IDX]
    # feature for dataset
    dep_feat = np.array([dep_answers + dv])
    anx_feat = np.array([anx_answers + dv])
    str_feat = np.array([str_answers + dv])
    # Scaling 
    dep_scaled = bundle["dep_scaler"].transform(dep_feat)
    anx_scaled = bundle["anx_scaler"].transform(anx_feat)
    str_scaled = bundle["str_scaler"].transform(str_feat)
    # Predictions
    dep_level = int(bundle["depression_model"].predict(dep_scaled)[0])
    anx_level = int(bundle["anxiety_model"].predict(anx_scaled)[0])
    str_level = int(bundle["stress_model"].predict(str_scaled)[0])


    drift_flag = False

    if drift_monitor.reference:

        ref_dep = drift_monitor.reference.get("dep_train_scaled_flattened")
        ref_anx = drift_monitor.reference.get("anx_train_scaled_flattened")
        ref_str = drift_monitor.reference.get("str_train_scaled_flattened")

        drift_dep = drift_monitor.check_drift(ref_dep, dep_scaled.flatten())
        drift_anx = drift_monitor.check_drift(ref_anx, anx_scaled.flatten())
        drift_str = drift_monitor.check_drift(ref_str, str_scaled.flatten())

        drift_flag = drift_dep or drift_anx or drift_str

    return DASSResponse(
        depression_level=dep_level,
        anxiety_level=anx_level,
        stress_level=str_level,
        drift_detected=drift_flag
    )


@app.get("/")
def health():
    return {"status": "API Running"}
