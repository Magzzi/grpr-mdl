import pandas as pd

import joblib

import os

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ---------- CONFIG ----------
DATA_PATH = "data/student_scores.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "student_score_model.pkl")

# ---------- LOAD DATA ----------
df = pd.read_csv(DATA_PATH)

# Define X (features) and y (target)
X = df[["Hours_Studied", "Sleep_Hours","Social_Media_Hours"]]
y = df["Exam_Score"]

# Split into training and testing sets (80% train, 20% test)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a model

model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions on Testing Set
y_pred = model.predict(X_test)

# Evaluate the model
print("Predictions: ", y_pred)
print("Actual: ", y_test.values)

# Metrics
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
# print("R² Score:", r2_score(y_test, y_pred))

# Step 6: Show learned coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Save the model
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# ---------- LOAD MODEL (example) ----------
loaded_model = joblib.load(MODEL_PATH)


# ---------- PLOTS ----------

# 1. Predicted vs Actual
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, color="blue", label="Predicted vs Actual")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label="Perfect Prediction")
plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.title("Predicted vs Actual Exam Scores")
plt.legend()
plt.show()

# 2. Feature Importance (Bar Chart)
features = X.columns
coefficients = model.coef_

plt.figure(figsize=(6,4))
plt.bar(features, coefficients, color="green")
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.title("Feature Importance in Predicting Exam Score")
plt.show()