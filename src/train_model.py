import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib


# Load dataset
data = pd.read_csv(
    "data/f1_data.csv"
)


print("Dataset loaded")
print(data.head())


# Create target column
# For MVP:
# If tyre is old -> possible pit

data["Pit"] = (
    data["TyreLife"] > 25
).astype(int)



# Convert tyre compound text into numbers

data = pd.get_dummies(
    data,
    columns=["Compound"]
)



# Remove unwanted columns

X = data.drop(
    [
        "Driver",
        "Pit"
    ],
    axis=1
)


y = data["Pit"]



# Create AI model

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)



# Train

model.fit(
    X,
    y
)



# Save model

joblib.dump(
    model,
    "models/pit_model.pkl"
)


print("🏎️ RaceOS AI trained successfully!")