import pandas as pd
import matplotlib.pyplot as plt


# Load the impact dataset
data = pd.read_csv(
    "data/processed/impact_dataset.csv"
)


# Plot peak vibration
plt.figure(figsize=(10, 6))

for label in data["label"].unique():

    subset = data[data["label"] == label]

    plt.scatter(
        range(len(subset)),
        subset["peak"],
        label=label,
        alpha=0.6
    )


plt.xlabel("Impact number")
plt.ylabel("Peak vibration (m/s²)")
plt.title("RAPID - Impact Peak Distribution")

plt.legend()
plt.grid(True)

plt.show()