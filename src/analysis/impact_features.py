import pandas as pd
import numpy as np


# Load processed data
data = pd.read_csv("data/raw/processed_accelerometer_data.csv")


def detect_impacts(signal, time, threshold=1.0, min_gap=0.2):

    candidate_indices = np.where(np.abs(signal) > threshold)[0]

    impact_indices = []

    for index in candidate_indices:

        if len(impact_indices) == 0:
            impact_indices.append(index)
            continue

        time_difference = time[index] - time[impact_indices[-1]]

        if time_difference >= min_gap:
            impact_indices.append(index)

    return impact_indices


def calculate_impact_features(signal, time, impact_index):

    # Examine a small window around the impact
    window_size = 20

    start = max(0, impact_index - window_size)
    end = min(len(signal), impact_index + window_size)

    window = signal[start:end]

    peak = np.max(np.abs(window))
    rms = np.sqrt(np.mean(window ** 2))
    energy = np.sum(window ** 2)

    duration = time[end - 1] - time[start]

    return {
        "time": time[impact_index],
        "peak": peak,
        "rms": rms,
        "energy": energy,
        "duration": duration
    }


conditions = {
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
    print("=" * 40)

    for index in impacts:

        features = calculate_impact_features(
            signal,
            data["time"].values,
            index
        )

        print(
            f"Time: {features['time']:.2f}s | "
            f"Peak: {features['peak']:.3f} | "
            f"RMS: {features['rms']:.3f} | "
            f"Energy: {features['energy']:.3f} | "
            f"Duration: {features['duration']:.2f}s"
        )