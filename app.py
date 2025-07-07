import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("asthma_model.pkl")

st.title("Asthma Diagnosis & Recommendation AI")

st.subheader("Input Patient Symptoms")

cough = st.checkbox("Coughing")
wheezing = st.checkbox("Wheezing")
short_breath = st.checkbox("Shortness of breath")

if st.button("Predict"):
    symptoms = np.array([[int(cough), int(wheezing), int(short_breath)]])
    prediction = model.predict(symptoms)[0]
    
    severity_map = {0: "Low", 1: "Moderate", 2: "High"}
    recommendation_map = {
        0: "Use prescribed inhaler occasionally.",
        1: "Regular inhaler use and follow-up in 1 week.",
        2: "Seek immediate medical attention."
    }

    st.success(f"Predicted Severity: {severity_map[prediction]}")
    st.info(f"Recommendation: {recommendation_map[prediction]}")
