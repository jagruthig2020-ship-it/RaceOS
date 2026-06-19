import pandas as pd
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

print("Loading YOLO model...")
yolo_model = YOLO("vision/runs/detect/train-5/weights/last.pt")

print("Loading race data...")
data = pd.read_csv("data/race_data.csv")

# Create target
data["PitStop"] = data["PitInTime"].notna().astype(int)

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

encoder = LabelEncoder()

for col in ["Compound", "Driver", "Team"]:
    X[col] = encoder.fit_transform(X[col].astype(str))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pit_model = RandomForestClassifier(n_estimators=100, random_state=42)
pit_model.fit(X_train, y_train)

print("Checking image...")

results = yolo_model.predict(
    source="vision/dataset/test/images/_0_9991_jpg.rf.9616a1ec9466314815ecef52e1dabaec.jpg",
    imgsz=640,
    conf=0.01
)

for r in results:
    if len(r.boxes) > 0:
        print("🚗 Car detected")

        sample = X_test.iloc[0:1]
        prediction = pit_model.predict(sample)

        if prediction[0] == 1:
            print("🛞 Pit stop likely")
        else:
            print("✅ No pit stop likely")

    else:
        print("No car detected")