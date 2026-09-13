import joblib
import pandas as pd


# ============================================================
# 1. Load the trained RAPID model
# ============================================================

model_file = (
    "data/models/rapid_random_forest.pkl"
)

model = joblib.load(model_file)


# ============================================================
# 2. Enter measurements from a new impact
# ============================================================

peak = float(input("Enter peak vibration (m/s²): "))
rms = float(input("Enter RMS vibration (m/s²): "))
energy = float(input("Enter impact energy: "))


# ============================================================
# 3. Create the input data
# ============================================================

new_impact = pd.DataFrame({
    "peak": [peak],
    "rms": [rms],
    "energy": [energy]
})


# ============================================================
# 4. Ask the model for a prediction
# ============================================================

prediction = model.predict(new_impact)


# ============================================================
# 5. Display the result
# ============================================================

print("\nRAPID prediction")
print("=" * 30)

print(
    f"Road impact classified as: {prediction[0].upper()}"
)
