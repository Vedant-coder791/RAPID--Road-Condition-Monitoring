import pandas as pd
import numpy as np


# Load processed vibration data
data = pd.read_csv("data/raw/processed_accelerometer_data.csv")


def detect_impacts(signal, time, threshold=1.0, min_gap=0.2):
    """
    Detect individual impact events.

    threshold:
        Minimum vibration magnitude required to detect an impact.

    min_gap:
        Minimum time between two separate impacts.
    """

    # Find all samples above the threshold
    candidate_indices = np.where(np.abs(signal) > threshold)[0]

    impact_indices = []

    for index in candidate_indices:

        # First detected impact
        if len(impact_indices) == 0:
            impact_indices.append(index)
            continue

        # Time since previous detected impact
        time_difference = time[index] - time[impact_indices[-1]]

        # Only count it as a new impact if enough time has passed
        if time_difference >= min_gap:
            impact_indices.append(index)

    return impact_indices


conditions = {
    "Smooth": data["smooth_vibration"].values,
    "Rough": data["rough_vibration"].values,
    "Pothole": data["pothole_vibration"].values
}


for condition, signal in conditions.items():

    impacts = detect_impacts(
        signal,
        data["time"].values,
        threshold=1.0,
        min_gap=0.2
    )

    print(f"\n{condition} road")
    print("-" * 30)

    print(f"Number of detected impacts: {len(impacts)}")

    if len(impacts) > 0:

        print("Impact times:")

        for index in impacts:
            print(f"{data['time'].iloc[index]:.2f} seconds")