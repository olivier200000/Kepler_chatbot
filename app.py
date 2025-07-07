import streamlit as st
import joblib
import numpy as np
import openai
import os

# Set your OpenAI API Key securely
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load model
model = joblib.load("asthma_model.pkl")

st.title("Asthma Diagnosis & Recommendation AI")

st.subheader("Input Patient Symptoms")

# User inputs
cough = st.checkbox("Coughing")
wheezing = st.checkbox("Wheezing")
short_breath = st.checkbox("Shortness of breath")

if st.button("Predict"):
    symptoms = np.array([[int(cough), int(wheezing), int(short_breath)]])
    prediction = model.predict(symptoms)[0]
    
    # Static maps
    severity_map = {0: "Low", 1: "Moderate", 2: "High"}
    recommendation_map = {
        0: "Use prescribed inhaler occasionally.",
        1: "Regular inhaler use and follow-up in 1 week.",
        2: "Seek immediate medical attention."
    }

    severity = severity_map[prediction]
    recommendation = recommendation_map[prediction]

    st.success(f"Predicted Severity: {severity}")
    st.info(f"AI Recommendation: {recommendation}")

    # Prompt GPT for more detailed advice
    prompt = (
        f"A patient has asthma symptoms with a predicted severity of '{severity}'. "
        f"The current recommendation is: '{recommendation}'. "
        f"Provide additional advice to the patient in simple language."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if you have access
            messages=[
                {"role": "system", "content": "You are a medical assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )

        gpt_reply = response["choices"][0]["message"]["content"]
        st.markdown("### OpenAI GPT Response:")
        st.write(gpt_reply)

    except Exception as e:
        st.error(f"OpenAI error: {e}")
