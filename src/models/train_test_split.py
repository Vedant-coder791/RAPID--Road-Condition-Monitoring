import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 1. Load the impact dataset
# ============================================================

data = pd.read_csv(
    "data/processed/impact_dataset.csv"
)


# ============================================================
# 2. Get the unique recording IDs
# ============================================================

recordings = data[
    ["recording_id", "label"]
].drop_duplicates()


print("Total recordings:", len(recordings))


# ============================================================
# 3. Split RECORDINGS, not individual impacts
# ============================================================

train_recordings, test_recordings = train_test_split(
    recordings,
    test_size=0.20,
    random_state=42,
    stratify=recordings["label"]
)


# ============================================================
# 4. Get the actual impact rows belonging to each recording
# ============================================================

train_data = data[
    data["recording_id"].isin(
        train_recordings["recording_id"]
    )
]

test_data = data[
    data["recording_id"].isin(
        test_recordings["recording_id"]
    )
]


# ============================================================
# 5. Save the two datasets
# ============================================================

train_file = (
    "data/processed/train_impact_dataset.csv"
)

test_file = (
    "data/processed/test_impact_dataset.csv"
)

train_data.to_csv(
    train_file,
    index=False
)

test_data.to_csv(
    test_file,
    index=False
)


# ============================================================
# 6. Display the results
# ============================================================

print("\nTrain/Test split complete!")

print("\nTraining recordings:")
print(
    train_recordings["label"].value_counts()
)

print("\nTesting recordings:")
print(
    test_recordings["label"].value_counts()
)

print("\nTraining impacts:")
print(
    len(train_data)
)

print("\nTesting impacts:")
print(
    len(test_data)
)

print("\nSaved:")
print(train_file)
print(test_file)