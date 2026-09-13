import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv("data/raw/processed_accelerometer_data.csv")


def detect_impacts(signal, time, threshold=1.0, min_gap=0.2):

    candidate_indices = np.where(np.abs(signal) > threshold)[0]

    impact_indices = []

    for index in candidate_indices:

        if len(impact_indices) == 0:
            impact_indices.append(index)
            continue

        if time[index] - time[impact_indices[-1]] >= min_gap:
            impact_indices.append(index)

    return impact_indices


def get_peak_features(signal, time):

    impacts = detect_impacts(signal, time)

    peaks = []

    for index in impacts:

        start = max(0, index - 20)
        end = min(len(signal), index + 20)

        window = signal[start:end]

        peak = np.max(np.abs(window))

        peaks.append(peak)

    return peaks


time = data["time"].values

rough_peaks = get_peak_features(
    data["rough_vibration"].values,
    time
)

pothole_peaks = get_peak_features(
    data["pothole_vibration"].values,
    time
)


# Plot the comparison
plt.figure(figsize=(10, 6))

plt.scatter(
    range(len(rough_peaks)),
    rough_peaks,
    label="Rough road"
)

plt.scatter(
    range(len(pothole_peaks)),
    pothole_peaks,
    label="Pothole road"
)

plt.axhline(
    2.0,
    linestyle="--",
    label="Example threshold"
)

plt.xlabel("Impact number")
plt.ylabel("Peak vibration (m/s²)")
plt.title("RAPID - Impact Peak Comparison")
plt.legend()
plt.grid(True)

plt.show()
