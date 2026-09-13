
import pandas as pd

# Load the raw accelerometer data
input_file = "data/raw/test_accelerometer_data.csv"
data = pd.read_csv(input_file)

# Gravity is approximately 9.81 m/s²
gravity = 9.81

# Remove the effect of gravity
data["smooth_vibration"] = data["smooth_acceleration"] - gravity
data["rough_vibration"] = data["rough_acceleration"] - gravity
data["pothole_vibration"] = data["pothole_acceleration"] - gravity

# Save the processed data
output_file = "data/raw/processed_accelerometer_data.csv"
data.to_csv(output_file, index=False)

# Display the first few processed values
print("Processed acceleration data:")
print(
    data[
        [
            "time",
            "smooth_vibration",
            "rough_vibration",
            "pothole_vibration"
        ]
    ].head()
)

print(f"\nSaved processed data to: {output_file}")

