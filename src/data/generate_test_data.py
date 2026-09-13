import numpy as np
import pandas as pd

# Make results reproducible
np.random.seed(42)

# Sampling settings
sampling_rate = 100  # samples per second
duration = 30        # seconds
n_samples = sampling_rate * duration

time = np.arange(n_samples) / sampling_rate


def generate_signal(condition):
    """
    Generate simulated vertical acceleration for a road condition.
    """

    # Gravity is approximately 9.81 m/s²
    gravity = 9.81

    # Normal small vibrations
    signal = gravity + np.random.normal(0, 0.05, n_samples)

    if condition == "smooth":
        # Very small road vibrations
        signal += np.random.normal(0, 0.03, n_samples)

    elif condition == "rough":
        # Larger continuous vibrations
        signal += np.random.normal(0, 0.20, n_samples)

        # Add several moderate bumps
        for _ in range(15):
            position = np.random.randint(0, n_samples)
            signal[position:position + 5] += np.random.uniform(0.5, 1.5)

    elif condition == "pothole":
        # Start with rough-road vibration
        signal += np.random.normal(0, 0.20, n_samples)

        # Add sharp impacts
        for _ in range(6):
            position = np.random.randint(0, n_samples)
            signal[position] += np.random.uniform(2, 4)

    return signal


# Generate three road conditions
smooth = generate_signal("smooth")
rough = generate_signal("rough")
pothole = generate_signal("pothole")


# Put everything into a dataframe
data = pd.DataFrame({
    "time": time,
    "smooth_acceleration": smooth,
    "rough_acceleration": rough,
    "pothole_acceleration": pothole
})


# Save the data
output_file = "data/raw/test_accelerometer_data.csv"
data.to_csv(output_file, index=False)

print(f"Saved test data to: {output_file}")
print(data.head())
