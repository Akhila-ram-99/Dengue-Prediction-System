import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ---------------- LOAD DATASET ----------------

data = pd.read_csv("dengue.csv")

print("Dataset loaded successfully.")
print("Total records:", len(data))


# ---------------- INPUT FEATURES ----------------

features = [
    "Fever",
    "Headache",
    "JointPain",
    "Vomiting",
    "Rash",
    "EyePain",
    "Fatigue",
    "MusclePain",
    "Nausea"
]

X = data[features]
y = data["Result"]


# ---------------- TRAIN / TEST SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------- RANDOM FOREST MODEL ----------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42
)


# ---------------- TRAIN MODEL ----------------

model.fit(X_train, y_train)


# ---------------- TEST MODEL ----------------

y_pred = model.predict(X_test)


# ---------------- MODEL PERFORMANCE ----------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    pos_label="Yes",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label="Yes",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label="Yes",
    zero_division=0
)


print("\n========== MODEL PERFORMANCE ==========")

print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1 Score :", round(f1 * 100, 2), "%")


# ---------------- CONFUSION MATRIX ----------------

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ---------------- CLASSIFICATION REPORT ----------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ---------------- 5-FOLD CROSS VALIDATION ----------------

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\n========== CROSS VALIDATION ==========")

print("CV Scores:", cv_scores)

print(
    "Average CV Accuracy:",
    round(cv_scores.mean() * 100, 2),
    "%"
)


# ---------------- SAVE MODEL ----------------

joblib.dump(model, "dengue_model.pkl")

print("\n======================================")
print("Model trained successfully.")
print("Model saved as: dengue_model.pkl")
print("======================================")
