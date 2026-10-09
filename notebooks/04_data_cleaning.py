import pandas as pd

file_path = "data/US_Accidents_March23.csv/US_Accidents_March23.csv"

# Load 100,000 rows for now
df = pd.read_csv(file_path, nrows=100000)

print("Original shape:", df.shape)

# -----------------------------------
# 1. Convert date/time columns
# -----------------------------------

df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
df["End_Time"] = pd.to_datetime(df["End_Time"], errors="coerce")

# -----------------------------------
# 2. Create useful time features
# -----------------------------------

df["Hour"] = df["Start_Time"].dt.hour
df["Day_of_Week"] = df["Start_Time"].dt.dayofweek
df["Month"] = df["Start_Time"].dt.month

# -----------------------------------
# 3. Select useful columns
# -----------------------------------

features = [
    "Severity",
    "Start_Lat",
    "Start_Lng",
    "Distance(mi)",
    "Temperature(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)",
    "Hour",
    "Day_of_Week",
    "Month",
    "Amenity",
    "Bump",
    "Crossing",
    "Give_Way",
    "Junction",
    "Railway",
    "Roundabout",
    "Station",
    "Stop",
    "Traffic_Calming",
    "Traffic_Signal"
]

df_clean = df[features].copy()

# -----------------------------------
# 4. Check missing values
# -----------------------------------

print("\nMissing values BEFORE cleaning:")
print(df_clean.isnull().sum())

# -----------------------------------
# 5. Fill numerical missing values
# -----------------------------------

numeric_columns = df_clean.select_dtypes(
    include=["float64", "int64"]
).columns

for column in numeric_columns:
    df_clean[column] = df_clean[column].fillna(
        df_clean[column].median()
    )

# -----------------------------------
# 6. Check missing values again
# -----------------------------------

print("\nMissing values AFTER cleaning:")
print(df_clean.isnull().sum())

print("\nCleaned shape:", df_clean.shape)

print("\nFirst 5 cleaned rows:")
print(df_clean.head())

print("\nData cleaning completed successfully!")