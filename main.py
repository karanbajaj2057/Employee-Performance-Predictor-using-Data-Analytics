# ============================================
# Employee Performance Predictor (FULL PROJECT)
# ============================================

# Import Libraries
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# ============================================
# 1. CREATE FOLDERS (AUTO)
# ============================================
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ============================================
# 2. GENERATE SYNTHETIC DATA (NO EXTERNAL FILE)
# ============================================
np.random.seed(42)
n = 500

data = pd.DataFrame({
    "age": np.random.randint(22, 50, n),
    "experience": np.random.randint(1, 25, n),
    "salary": np.random.randint(20000, 150000, n),
    "department": np.random.choice(["IT", "HR", "Sales"], n),
    "training_hours": np.random.randint(10, 100, n),
    "projects": np.random.randint(1, 10, n),
})

# Performance Logic (Realistic Simulation)
def assign_performance(row):
    if row["experience"] > 12 and row["training_hours"] > 60:
        return "High"
    elif row["projects"] > 5:
        return "Medium"
    else:
        return "Low"

data["performance"] = data.apply(assign_performance, axis=1)

# Save dataset
data.to_csv("data/employee_data.csv", index=False)

print("\n✅ Dataset Created Successfully!\n")
print(data.head())

# ============================================
# 3. PREPROCESSING
# ============================================
le_dept = LabelEncoder()
le_perf = LabelEncoder()

data["department"] = le_dept.fit_transform(data["department"])
data["performance"] = le_perf.fit_transform(data["performance"])

# ============================================
# 4. SPLIT DATA
# ============================================
X = data.drop("performance", axis=1)
y = data["performance"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# 5. TRAIN MODEL
# ============================================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ============================================
# 6. PREDICTION
# ============================================
y_pred = model.predict(X_test)

# ============================================
# 7. EVALUATION
# ============================================
accuracy = accuracy_score(y_test, y_pred)

print("\n🎯 Model Accuracy:", round(accuracy * 100, 2), "%")

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# ============================================
# 8. CONFUSION MATRIX
# ============================================
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/confusion_matrix.png")
plt.close()

# ============================================
# 9. FEATURE IMPORTANCE
# ============================================
importance = model.feature_importances_
features = X.columns

plt.figure()
plt.barh(features, importance)
plt.title("Feature Importance")
plt.savefig("outputs/feature_importance.png")
plt.close()

# ============================================
# 10. SAVE MODEL
# ============================================
joblib.dump(model, "models/model.pkl")

print("\n💾 Model saved at: models/model.pkl")

# ============================================
# 11. SAMPLE PREDICTION (REAL DEMO)
# ============================================
sample = pd.DataFrame({
    "age": [30],
    "experience": [10],
    "salary": [50000],
    "department": [le_dept.transform(["IT"])[0]],
    "training_hours": [70],
    "projects": [6]
})

prediction = model.predict(sample)
label = le_perf.inverse_transform(prediction)

print("\n🧪 Sample Prediction:", label[0])

# ============================================
# END
# ============================================
print("\n✅ PROJECT RUN SUCCESSFULLY 🚀")