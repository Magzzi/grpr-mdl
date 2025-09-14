import pandas as pd
import os
import joblib

# ---------- CONFIG ----------
MODEL_PATH = "models/student_score_model.pkl"  # adjust if needed

# ---------- LOAD MODEL ----------
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
else:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train the model first.")

# Create DataFrame with correct feature names
test_students = pd.DataFrame([
    [2, 6, 5],   # low study, average sleep, high social media
    [5, 7, 2],   # moderate study, good sleep, some social media
    [8, 6, 1],   # high study, average sleep, very little social media
    [10, 8, 0],  # max study, good sleep, no social media
    [4, 5, 4],   # low study, poor sleep, high social media
], columns=["Hours_Studied", "Sleep_Hours", "Social_Media_Hours"])

# Predict with your loaded model
predictions = model.predict(test_students)
for i, score in enumerate(predictions):
    print(f"Student {i+1} predicted score: {round(score, 2)}")
