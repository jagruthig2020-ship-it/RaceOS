import pandas as pd
import joblib


# Load trained RaceOS model

model = joblib.load(
    "models/pit_model.pkl"
)


print("🏎️ RaceOS Strategy Engine")


# Current race situation

lap = int(input("Current lap: "))

tyre_age = int(input("Tyre age: "))

lap_time = float(input("Last lap time (seconds): "))



# Create input for AI

race_state = pd.DataFrame({

    "LapNumber":[lap],

    "LapTime":[lap_time],

    "TyreLife":[tyre_age],

    "Compound_HARD":[0],

    "Compound_MEDIUM":[1],

    "Compound_SOFT":[0]

})



# Prediction

prediction = model.predict_proba(
    race_state
)



pit_probability = prediction[0][1]



print()

print(
    f"Pit Probability: {pit_probability*100:.2f}%"
)



if pit_probability >= 0.7:

    print(
        "🚨 RaceOS Decision: BOX BOX BOX"
    )

else:

    print(
        "✅ RaceOS Decision: STAY OUT"
    )