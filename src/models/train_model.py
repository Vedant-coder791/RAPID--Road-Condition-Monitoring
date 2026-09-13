import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 1. Load training and testing data
# ============================================================

train_data = pd.read_csv(
    "data/processed/train_impact_dataset.csv"
)

test_data = pd.read_csv(
    "data/processed/test_impact_dataset.csv"
)


# ============================================================
# 2. Select the features
# ============================================================

features = [
    "peak",
    "rms",
    "energy"
]


X_train = train_data[features]
y_train = train_data["label"]

X_test = test_data[features]
y_test = test_data["label"]


# ============================================================
# 3. Create the Random Forest model
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ============================================================
# 4. Train the model
# ============================================================

print("Training RAPID model...")

model.fit(
    X_train,
    y_train
)


# ============================================================
# 5. Make predictions
# ============================================================

predictions = model.predict(
    X_test
)


# ============================================================
# 6. Evaluate the model
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nModel evaluation")
print("=" * 40)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# 7. Classification report
# ============================================================

print("\nClassification report:")

print(
    classification_report(
        y_test,
        predictions
    )
)


# ============================================================
# 8. Confusion matrix
# ============================================================

print("\nConfusion matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ============================================================
# 9. Save the trained model
# ============================================================

model_file = (
    "data/processed/rapid_random_forest.pkl"
)

joblib.dump(
    model,
    model_file
)


print("\nModel saved successfully!")
print(
    f"Saved to: {model_file}"
)