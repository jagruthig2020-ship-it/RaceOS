import streamlit as st
import pandas as pd
import joblib
import time

# Load ML model
model = joblib.load("models/pit_model.pkl")

# Load race telemetry data
df = pd.read_csv("race_data.csv")

# Keep only supported compounds
df = df[df["Compound"].isin(["SOFT", "MEDIUM", "HARD"])].copy()

# Convert lap time to seconds
df["LapTime"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()

# Streamlit config
st.set_page_config(
    page_title="RaceOS",
    page_icon="🏎️",
    layout="wide"
)

# Header
st.title("🏎️ RaceOS")
st.subheader("AI Formula 1 Strategy Intelligence Dashboard")

st.info("""
🏎️ RaceOS uses Formula 1 telemetry, machine learning, and strategy heuristics 
to predict optimal pit stop windows in real time.
""")

# Driver selection
driver = st.selectbox(
    "Select Driver",
    sorted(df["Driver"].unique())
)

# Filter selected driver
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

        # ML input
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

        # Hybrid scoring
        pit_score = probability

        # Strategy heuristics
        if tyre_life > 15:
            pit_score += 0.25

        if lap_time > 92:
            pit_score += 0.20

        if lap > 30:
            pit_score += 0.15

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
            st.progress(float(pit_score))

            if pit_score > 0.60:
                st.error("🚨 PIT THIS LAP")

                st.write("### AI Strategy Reasoning")
                st.write("• High tyre degradation detected")
                st.write("• Performance drop observed")
                st.write("• Strategic pit window open")
                st.write("• Undercut opportunity available")

            else:
                st.success("✅ STAY OUT")

                st.write("### AI Strategy Reasoning")
                st.write("• Tyres within optimal range")
                st.write("• Competitive lap pace")
                st.write("• No immediate pit advantage")

        time.sleep(2)