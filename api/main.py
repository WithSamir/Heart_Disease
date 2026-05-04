"""
api/main.py — FastAPI REST API for Heart Disease Prediction
Run: uvicorn api.main:app --reload  (from heart_disease_project/)
Docs: http://localhost:8000/docs
"""

import os, sys, json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MDL_DIR  = os.path.join(BASE_DIR, "models")

app = FastAPI(
    title="CardioAI API",
    description="Heart Disease Risk Prediction REST API — powered by ML",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Load models ────────────────────────────────────────────────────────────────
try:
    MODELS = {
        "random_forest":        joblib.load(os.path.join(MDL_DIR, "random_forest_model.pkl")),
        "xgboost":              joblib.load(os.path.join(MDL_DIR, "xgboost_model.pkl")),
        "logistic_regression":  joblib.load(os.path.join(MDL_DIR, "logistic_regression_model.pkl")),
    }
    MODEL_COLS = joblib.load(os.path.join(MDL_DIR, "model_columns.pkl"))
except Exception as e:
    MODELS, MODEL_COLS = {}, []
    print(f"WARNING: Could not load models — {e}")

# ── Schemas ────────────────────────────────────────────────────────────────────
class PatientInput(BaseModel):
    age:            int   = Field(..., ge=1,   le=120,  example=52)
    gender:         str   = Field(...,                   example="Male")
    cholesterol:    int   = Field(..., ge=100, le=600,   example=230)
    blood_pressure: int   = Field(..., ge=60,  le=220,   example=130)
    heart_rate:     int   = Field(..., ge=30,  le=220,   example=80)
    blood_sugar:    int   = Field(..., ge=50,  le=500,   example=100)
    smoking:        str   = Field(...,                   example="Never")   # Never/Current/Former
    alcohol_intake: str   = Field(...,                   example="None")    # None/Moderate/Heavy
    exercise_hours: int   = Field(..., ge=0,  le=20,    example=3)
    stress_level:   int   = Field(..., ge=1,  le=10,    example=5)
    family_history: str   = Field(...,                   example="No")
    diabetes:       str   = Field(...,                   example="No")
    obesity:        str   = Field(...,                   example="No")
    exercise_induced_angina: str = Field(...,            example="No")
    chest_pain_type: str  = Field(...,                   example="Typical Angina")
    model_name:     str   = Field("random_forest",       example="random_forest")

class PredictionResponse(BaseModel):
    model_used:       str
    risk_label:       str
    risk_probability: float
    risk_percent:     str

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "CardioAI API is running 🫀"}

@app.get("/models", tags=["Info"])
def list_models():
    """List available models and their performance metrics."""
    metrics_path = os.path.join(MDL_DIR, "model_metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    return {"available_models": list(MODELS.keys()), "metrics": metrics}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(patient: PatientInput):
    """Predict heart disease risk for a patient."""
    if patient.model_name not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{patient.model_name}' not found. Available: {list(MODELS.keys())}"
        )

    raw = {
        "Age": patient.age, "Cholesterol": patient.cholesterol,
        "Blood_Pressure": patient.blood_pressure, "Heart_Rate": patient.heart_rate,
        "Exercise_Hours": patient.exercise_hours, "Stress_Level": patient.stress_level,
        "Blood_Sugar": patient.blood_sugar,
        "Gender_Male":          1 if patient.gender == "Male" else 0,
        "Smoking_Current":      1 if patient.smoking == "Current" else 0,
        "Smoking_Former":       1 if patient.smoking == "Former"  else 0,
        "Alcohol_Intake_Heavy":    1 if patient.alcohol_intake == "Heavy"    else 0,
        "Alcohol_Intake_Moderate": 1 if patient.alcohol_intake == "Moderate" else 0,
        "Family_History_Yes":  1 if patient.family_history == "Yes" else 0,
        "Diabetes_Yes":        1 if patient.diabetes == "Yes" else 0,
        "Obesity_Yes":         1 if patient.obesity == "Yes" else 0,
        "Exercise_Induced_Angina_Yes": 1 if patient.exercise_induced_angina == "Yes" else 0,
        "Chest_Pain_Type_Non-Anginal Pain": 1 if patient.chest_pain_type == "Non-Anginal Pain" else 0,
        "Chest_Pain_Type_Typical Angina":   1 if patient.chest_pain_type == "Typical Angina"   else 0,
        "Chest_Pain_Type_Atypical Angina":  1 if patient.chest_pain_type == "Atypical Angina"  else 0,
    }

    df = pd.DataFrame([raw]).reindex(columns=MODEL_COLS, fill_value=0)
    clf  = MODELS[patient.model_name]
    prob = float(clf.predict_proba(df)[0][1])
    label = "HIGH RISK" if prob > 0.5 else "LOW RISK"

    return PredictionResponse(
        model_used=patient.model_name,
        risk_label=label,
        risk_probability=round(prob, 4),
        risk_percent=f"{prob:.1%}",
    )
