import streamlit as st
from PIL import Image

st.set_page_config(page_title="RaceOS", layout="wide")

st.title("🏎 RaceOS - F1 Pit Stop Predictor")
st.subheader("AI-powered Computer Vision + Pit Strategy Prediction")

st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload a race image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.markdown("---")
    st.header("Detection Status")
    st.success("🚗 Car Detected")

    st.header("Pit Stop Prediction")
    st.warning("🛞 Pit Stop Likely")

    st.markdown("---")
    st.header("Race Stats")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Driver", "HAM")

    with col2:
        st.metric("Lap", "34")

    with col3:
        st.metric("Tyre Life", "18")

    with col4:
        st.metric("Position", "4")
else:
    st.info("Please upload an image to begin.")