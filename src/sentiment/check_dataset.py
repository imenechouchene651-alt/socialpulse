
import pandas as pd

# Load the CSV file
df = pd.read_csv("data/labeled_comments.csv")

# Display the first rows
print("FIRST COMMENTS:")
print(df.head())

# Display dataset information
print("\nDATASET INFORMATION:")
print(df.info())

# Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Check sentiment distribution
print("\nSENTIMENT DISTRIBUTION:")
print(df["sentiment"].value_counts())

# Check assistant distribution
print("\nASSISTANT DISTRIBUTION:")
print(df["assistant"].value_counts())

# Check invalid labels
valid_labels = [
    "Positive",
    "Negative",
    "Neutral",
    "Mixed",
    "Unclear"
]

invalid = df[~df["sentiment"].isin(valid_labels)]

print("\nINVALID LABELS:")
print(invalid)

print("\nDataset verification completed.")
