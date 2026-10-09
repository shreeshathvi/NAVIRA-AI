import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------------
# 1. Load dataset
# -----------------------------------

file_path = "data/US_Accidents_March23.csv/US_Accidents_March23.csv"

df = pd.read_csv(file_path, nrows=100000)

print("Original shape:", df.shape)

# -----------------------------------
# 2. Convert date/time
# -----------------------------------

df["Start_Time"] = pd.to_datetime(
    df["Start_Time"],
    errors="coerce"
)

# -----------------------------------
# 3. Create time features
# -----------------------------------

df["Hour"] = df["Start_Time"].dt.hour
df["Day_of_Week"] = df["Start_Time"].dt.dayofweek
df["Month"] = df["Start_Time"].dt.month

# -----------------------------------
# 4. Select features
# -----------------------------------

features = [
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

X = df[features].copy()
y = df["Severity"]

# -----------------------------------
# 5. Handle missing values
# -----------------------------------

for column in X.select_dtypes(
    include=["float64", "int64"]
).columns:

    X[column] = X[column].fillna(
        X[column].median()
    )

# -----------------------------------
# 6. Train/Test split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# -----------------------------------
# 7. Create Random Forest
# -----------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# -----------------------------------
# 8. Train model
# -----------------------------------

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training completed!")

# -----------------------------------
# 9. Make predictions
# -----------------------------------

y_pred = model.predict(X_test)

# -----------------------------------
# 10. Evaluate
# -----------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------------
# 11. Feature importance
# -----------------------------------

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

print("\nRandom Forest model completed successfully!")