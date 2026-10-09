import pandas as pd

file_path = "data/US_Accidents_March23.csv/US_Accidents_March23.csv"

# Read only the first 10,000 rows
df = pd.read_csv(file_path, nrows=10000)

print("Shape of sample:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset sample loaded successfully!")