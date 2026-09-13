import pandas as pd
import numpy as np


def calculate_features(signal):
    """Calculate basic vibration features."""

    mean = np.mean(signal)
    std = np.std(signal)
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))

    return {
        "mean": mean,
        "std": std,
        "rms": rms,
        "peak": peak
    }


# Load processed data
data = pd.read_csv("data/raw/processed_accelerometer_data.csv")


# Select the three road conditions
conditions = {
    "Smooth": data["smooth_vibration"],
    "Rough": data["rough_vibration"],
    "Pothole": data["pothole_vibration"]
}


# Calculate and display features
for condition, signal in conditions.items():

    features = calculate_features(signal)

    print(f"\n{condition} road")
    print("-" * 30)

    for name, value in features.items():
        print(f"{name}: {value:.3f}")