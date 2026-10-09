import pandas as pd

# Load dataset
df = pd.read_csv("data/tn-road-crashes-1993-2025.csv")

# Basic information
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())