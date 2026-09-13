import pandas as pd
import matplotlib.pyplot as plt

# Load the test data
data = pd.read_csv("data/raw/test_accelerometer_data.csv")

# Create the plot
plt.figure(figsize=(12, 6))

plt.plot(data["time"], data["smooth_acceleration"], label="Smooth Road")
plt.plot(data["time"], data["rough_acceleration"], label="Rough Road")
plt.plot(data["time"], data["pothole_acceleration"], label="Pothole")

plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration (m/s²)")
plt.title("RAPID - Simulated Road Conditions")
plt.legend()
plt.grid(True)

plt.show()
