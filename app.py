import streamlit as st
import pandas as pd
import joblib


# Load RaceOS model

model = joblib.load(
    "models/pit_model.pkl"
)


# Page title

st.set_page_config(
    page_title="RaceOS",
    page_icon="🏎️"
)


st.title("🏎️ RaceOS")

st.subheader(
    "AI Formula 1 Strategy Intelligence System"
)



st.write(
    "Predict optimal pit stop timing using machine learning"
)



# Inputs


lap = st.slider(
    "Current Lap",
    1,
    70,
    40
)



tyre_age = st.slider(
    "Tyre Age (laps)",
    1,
    60,
    25
)



lap_time = st.slider(
    "Last Lap Time (seconds)",
    70.0,
    120.0,
    85.0
)



compound = st.selectbox(
    "Tyre Compound",
    [
        "SOFT",
        "MEDIUM",
        "HARD"
    ]
)



if st.button("🚀 Run RaceOS Prediction"):


    data = {

        "LapNumber":[lap],

        "LapTime":[lap_time],

        "TyreLife":[tyre_age],

        "Compound_HARD":[0],

        "Compound_MEDIUM":[0],

        "Compound_SOFT":[0]

    }


    # set compound

    data[
        "Compound_" + compound
    ] = 1



    input_data = pd.DataFrame(data)



    prediction = model.predict_proba(
        input_data
    )


    probability = prediction[0][1]



    st.divider()



    st.metric(
        "Pit Stop Probability",
        f"{probability*100:.1f}%"
    )



    if probability > 0.7:

        st.error(
            "🚨 BOX BOX BOX"
        )

        st.write(
            "RaceOS recommends pitting now"
        )


    else:

        st.success(
            "✅ STAY OUT"
        )

        st.write(
            "Current strategy is optimal"
        )