import pandas as pd

file_path = "data/US_Accidents_March23.csv/US_Accidents_March23.csv"

df = pd.read_csv(file_path, nrows=100000)

print("Dataset shape:", df.shape)

print("\nSeverity counts:")
print(df["Severity"].value_counts().sort_index())

print("\nSeverity percentages:")
print(df["Severity"].value_counts(normalize=True).sort_index() * 100)