import streamlit as st
import openai
import pandas as pd
import PyPDF2
from io import StringIO

# 🔐 Secure OpenAI API Key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# 📋 Page Config
st.set_page_config(page_title="Doctor Assistant AI", layout="wide")

# 🏥 Header
st.markdown("""
    <div style='text-align:center; padding: 10px'>
        <h1 style='color: #0078D4;'>🩺 Doctor Assistant AI</h1>
        <p style='font-size: 18px;'>Upload lab results, explore disease symptoms, or chat with your AI medical assistant.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar: Navigation
option = st.sidebar.radio("🧭 Navigate", ["🏥 AI Chat", "📄 Upload Lab Results", "🦠 Disease Symptoms"])

# ------------------------ CHAT WITH AI ------------------------
if option == "🏥 AI Chat":
    st.subheader("💬 Ask the Doctor Assistant")
    st.markdown("Type any question related to symptoms, treatment, medication, or patient care.")

    with st.container():
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input("👨‍⚕️ Enter your question here:", key="ai_chat_input")
        with col2:
            ask_button = st.button("Ask")

    if ask_button and user_input:
        with st.spinner("Thinking..."):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful and professional medical assistant."},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=300
            )
            st.success("🤖 AI Response")
            st.write(response.choices[0].message.content)

# ------------------------ UPLOAD LAB RESULTS ------------------------
elif option == "📄 Upload Lab Results":
    st.subheader("📋 Upload and Interpret Lab Report")
    uploaded_file = st.file_uploader("Upload a PDF or CSV lab report", type=["pdf", "csv"])

    if uploaded_file:
        if uploaded_file.name.endswith(".pdf"):
            st.info("🧾 PDF file uploaded successfully.")
            reader = PyPDF2.PdfReader(uploaded_file)
            pdf_text = "\n".join([page.extract_text() for page in reader.pages])
            st.text_area("📃 Extracted Text", pdf_text, height=250)

            if st.button("🧠 Interpret with AI"):
                with st.spinner("Interpreting..."):
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful AI doctor that interprets lab results."},
                            {"role": "user", "content": f"Interpret this lab report:\n\n{pdf_text}"}
                        ],
                        max_tokens=300
                    )
                    st.success("🤖 AI Interpretation")
                    st.write(response.choices[0].message.content)

        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            st.dataframe(df)

            if st.button("🧠 Analyze CSV with AI"):
                with st.spinner("Analyzing..."):
                    csv_text = df.to_string()
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You're a helpful AI doctor assistant that explains CSV lab results."},
                            {"role": "user", "content": f"Analyze this lab data:\n\n{csv_text}"}
                        ],
                        max_tokens=300
                    )
                    st.success("🤖 AI Analysis")
                    st.write(response.choices[0].message.content)

# ------------------------ DISEASE SYMPTOMS ------------------------
elif option == "🦠 Disease Symptoms":
    st.subheader("🔍 Disease Symptom Checker")
    st.markdown("Select a disease to view its common symptoms and medical advice.")

    disease = st.selectbox("Select disease", ["Malaria", "Asthma", "Typhoid", "Tuberculosis", "COVID-19"])

    if st.button("🔬 Get Symptoms"):
        with st.spinner("Fetching symptom data..."):
            prompt = f"What are the major symptoms and advice for treating {disease}?"
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a medical expert giving symptom lists and basic advice."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250
            )
            st.success(f"🧾 Symptoms of {disease}")
            st.write(response.choices[0].message.content)
