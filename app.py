import streamlit as st
import openai
import pandas as pd
import google.generativeai as genai
import PyPDF2

# Secure OpenAI API Key from Streamlit secrets
genai.api_key = st.secrets["GEMINI_API_KEY"]

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
# 🏥 AI CHAT PAGE — Chat History Style
# ------------------------------------------
if page == "🏥 AI Chat":
    st.subheader("💬 Ask the Doctor Assistant")
    st.markdown("Type your medical question and press **Enter**. The AI will respond immediately. All past questions will stay visible.")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Function to handle input and get GPT response
    def handle_chat():
        user_input = st.session_state.user_question.strip()
        if user_input:
            with st.spinner("Thinking..."):
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful and professional medical assistant."},
                        {"role": "user", "content": user_input}
                    ],
                    max_tokens=300
                )
                ai_reply = response.choices[0].message.content.strip()
                st.session_state.chat_history.append((user_input, ai_reply))
            st.session_state.user_question = ""  # Clear input after sending

    # Show previous Q&A
    for i, (q, a) in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**👨‍⚕️ Question {i+1}:** {q}")
            st.markdown(f"**🤖 Answer {i+1}:** {a}")
            st.markdown("---")

    # Input box at bottom of chat
    st.text_input(
        "Ask your next question here:",
        key="user_question",
        on_change=handle_chat,
        placeholder="e.g. What are the signs of severe asthma?"
    )

# ------------------------------------------
# 📄 LAB RESULT INTERPRETER
# ------------------------------------------
elif page == "📄 Upload Lab Results":
    st.subheader("📋 Upload and Interpret Lab Report")
    uploaded_file = st.file_uploader("Upload a PDF or CSV lab report", type=["pdf", "csv"])

    if uploaded_file:
        if uploaded_file.name.endswith(".pdf"):
            st.info("🧾 PDF file uploaded successfully.")
            reader = PyPDF2.PdfReader(uploaded_file)
            pdf_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
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
# 🦠 DISEASE SYMPTOMS PAGE
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
