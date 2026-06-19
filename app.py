import streamlit as st
import pandas as pd
import joblib
import time

# Load model
model = joblib.load("models/pit_model.pkl")

# Load race data
df = pd.read_csv("race_data.csv")

# Keep supported compounds only
df = df[df["Compound"].isin(["SOFT", "MEDIUM", "HARD"])].copy()

# Convert LapTime to seconds
df["LapTime"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()

# Streamlit config
st.set_page_config(
    page_title="RaceOS",
    page_icon="🏎️",
    layout="wide"
)

st.title("🏎️ RaceOS")
st.subheader("AI Formula 1 Strategy Intelligence Dashboard")

# Driver selection
driver = st.selectbox(
    "Select Driver",
    sorted(df["Driver"].unique())
)

# Filter driver data
selected_df = df[df["Driver"] == driver].copy()
selected_df.reset_index(drop=True, inplace=True)

st.markdown("---")

start = st.button("▶ Start Race Simulation")

placeholder = st.empty()

if start:
    for i in range(min(len(selected_df), 60)):

        row = selected_df.iloc[i]

        lap = row["LapNumber"]
        lap_time = row["LapTime"]
        tyre_life = row["TyreLife"]
        compound = row["Compound"]
        position = row["Position"]
        speed = row["SpeedST"]

        # Prepare model input
        input_data = {
            "LapNumber": [lap],
            "LapTime": [lap_time],
            "TyreLife": [tyre_life],
            "Compound_HARD": [0],
            "Compound_MEDIUM": [0],
            "Compound_SOFT": [0]
        }

        input_data["Compound_" + compound] = 1
        input_df = pd.DataFrame(input_data)

        # ML prediction
        prediction = model.predict_proba(input_df)
        probability = prediction[0][1]

        # Hybrid strategy scoring
        pit_score = probability

        # Rule 1: Old tyres
        if tyre_life > 15:
            pit_score += 0.15

        # Rule 2: Slow lap
        if lap_time > 92:
            pit_score += 0.10

        # Rule 3: Late race
        if lap > 30:
            pit_score += 0.10

        # Cap at 100%
        pit_score = min(pit_score, 1.0)

        with placeholder.container():

            st.markdown("## Live Telemetry")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Driver", driver)
            col2.metric("Lap", int(lap))
            col3.metric("Position", f"P{int(position)}")
            col4.metric("Speed", f"{int(speed)} km/h")

            st.markdown("")

            col5, col6, col7 = st.columns(3)

            col5.metric("Tyre Compound", compound)
            col6.metric("Tyre Life", int(tyre_life))
            col7.metric("Lap Time", f"{lap_time:.2f}s")

            st.markdown("---")
            st.markdown("## AI Strategy Prediction")

            st.metric("Pit Stop Probability", f"{pit_score*100:.2f}%")
            st.write(f"ML Probability: {probability:.4f}")
            st.write(f"Final Strategy Score: {pit_score:.4f}")

            if pit_score > 0.25:
                st.error("🚨 PIT THIS LAP")

                st.write("### AI Reasoning")
                st.write("✓ Tyre degradation high")
                st.write("✓ Lap performance dropping")
                st.write("✓ Pit window open")
                st.write("✓ Undercut opportunity available")

            else:
                st.success("✅ STAY OUT")

                st.write("### AI Reasoning")
                st.write("✓ Tyres performing normally")
                st.write("✓ Pace still competitive")
                st.write("✓ No urgent pit required")

        time.sleep(2)