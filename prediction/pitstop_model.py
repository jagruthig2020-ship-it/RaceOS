import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("../data/race_data.csv")

# Create target:
# 1 = pit stop happened
# 0 = no pit stop
data["PitStop"] = data["PitInTime"].notna().astype(int)

# Select useful columns
features = [
    "LapNumber",
    "Stint",
    "TyreLife",
    "Compound",
    "Driver",
    "Team",
    "Position"
]

X = data[features].copy()
y = data["PitStop"]

# Convert text columns to numbers
encoder = LabelEncoder()

for col in ["Compound", "Driver", "Team"]:
    X[col] = encoder.fit_transform(X[col].astype(str))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Test
prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("Model accuracy:", accuracy)

# Example prediction
sample = X_test.iloc[0:1]

result = model.predict(sample)

if result[0] == 1:
    print("Pit stop likely")
else:
    print("No pit stop likely")