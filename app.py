import streamlit as st
import openai
import pandas as pd
import PyPDF2

# Secure OpenAI API Key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Page settings
st.set_page_config(page_title="Doctor Assistant AI", layout="wide")

# App title
st.markdown("""
    <div style='text-align:center; padding: 10px'>
        <h1 style='color: #0078D4;'>🩺 Doctor Assistant AI</h1>
        <p style='font-size: 18px;'>Upload lab results, explore disease symptoms, or chat with your AI medical assistant instantly.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar navigation
page = st.sidebar.radio("🧭 Navigation", ["🏥 AI Chat", "📄 Upload Lab Results", "🦠 Disease Symptoms"])

# ------------------------------------------
# AI CHAT PAGE (ENTER KEY SUBMITS AUTOMATICALLY)
# ------------------------------------------
if page == "🏥 AI Chat":
    st.subheader("💬 Ask the Doctor Assistant")
    st.markdown("Type your question and press **Enter**. The AI will answer instantly.")

    # Store question and answer in session
    if "question" not in st.session_state:
        st.session_state.question = ""
    if "answer" not in st.session_state:
        st.session_state.answer = ""

    # Callback function when input changes (Enter key pressed)
    def handle_submit():
        query = st.session_state.question
        if query:
            with st.spinner("Thinking..."):
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful and professional medical assistant."},
                        {"role": "user", "content": query}
                    ],
                    max_tokens=300
                )
                st.session_state.answer = response.choices[0].message.content

    # Input field that auto-submits on Enter
    st.text_input(
        "👨‍⚕️ Your question:",
        key="question",
        on_change=handle_submit,
        placeholder="e.g. What are the symptoms of malaria?"
    )

    # Display the response
    if st.session_state.answer:
        st.success("🤖 AI Response")
        st.write(st.session_state.answer)

# ------------------------------------------
# LAB RESULT INTERPRETER
# ------------------------------------------
elif page == "📄 Upload Lab Results":
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

# ------------------------------------------
# DISEASE SYMPTOMS PAGE
# ------------------------------------------
elif page == "🦠 Disease Symptoms":
    st.subheader("🔍 Disease Symptom Checker")
    st.markdown("Select a disease to view its common symptoms and basic advice.")

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
