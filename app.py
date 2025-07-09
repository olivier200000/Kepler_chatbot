import streamlit as st
import pandas as pd
import google.generativeai as genai
import PyPDF2

# Secure Gemini API Key from Streamlit secrets
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error(f"Failed to configure Gemini API. Please ensure 'GEMINI_API_KEY' is set in Streamlit secrets. Error: {e}")
    st.stop()

# Initialize Gemini model safely
try:
    model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
    chat = model.start_chat()
except Exception as e:
    st.error(f"Failed to initialize Gemini model. Check model name or API access. Error: {e}")
    st.stop()

# Page settings with green theme
st.set_page_config(page_title="Doctor Assistant AI", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #F7FFF7;
        }
        .block-container {
            padding-top: 2rem;
        }
        .chat-box {
            background-color: #E8F5E9;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .user-question {
            color: #2E7D32;
            font-weight: bold;
        }
        .ai-answer {
            color: #1B5E20;
        }
        .custom-title {
            text-align: center;
            color: #2E7D32;
        }
        .header-image {
            text-align: center;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# 🖼 Add a centered image of doctor helping patient
st.markdown("""
    <div class="header-image">
        <img src="https://www.pngmart.com/files/22/Doctor-Consulting-Patient-PNG-Isolated-HD.png" width="300">
    </div>
""", unsafe_allow_html=True)

# Updated app title
st.markdown("""
    <h1 class="custom-title">🩺 Doctor Assistant AI</h1>
    <p style="text-align:center; font-size:18px;">
        Upload lab results, explore disease symptoms, or chat with your AI medical assistant instantly.
    </p>
""", unsafe_allow_html=True)

# Sidebar navigation
page = st.sidebar.radio("🧭 Navigation", ["🏥 AI Chat", "📄 Upload Lab Results", "🦠 Disease Symptoms"])

# ------------------------------------------
# 🏥 AI CHAT PAGE — Chat History Style
# ------------------------------------------
if page == "🏥 AI Chat":
    st.subheader("💬 Ask the Doctor Assistant")
    st.markdown("Type your medical question and press **Enter**. The AI will respond immediately. All past questions will stay visible.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    def handle_chat():
        user_input = st.session_state.user_question.strip()
        if user_input:
            with st.spinner("Thinking..."):
                try:
                    response = chat.send_message(user_input)
                    ai_reply = response.text.strip()
                    st.session_state.chat_history.append((user_input, ai_reply))
                except Exception as e:
                    st.error(f"Error communicating with Gemini: {e}")
            st.session_state.user_question = ""

    for i, (q, a) in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**👨‍⚕️ Question {i+1}:** {q}")
            st.markdown(f"**🤖 Answer {i+1}:** {a}")
            st.markdown("---")

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
            try:
                reader = PyPDF2.PdfReader(uploaded_file)
                pdf_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                st.text_area("📃 Extracted Text", pdf_text, height=250)
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
                pdf_text = ""

            if st.button("🧠 Interpret with AI") and pdf_text:
                with st.spinner("Interpreting..."):
                    try:
                        prompt = f"You are a helpful AI doctor. Interpret this lab report:\n\n{pdf_text}"
                        response = chat.send_message(prompt)
                        st.success("🤖 AI Interpretation")
                        st.write(response.text)
                    except Exception as e:
                        st.error(f"Error interpreting with AI: {e}")

        elif uploaded_file.name.endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df)
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                df = None

            if st.button("🧠 Analyze CSV with AI") and df is not None:
                with st.spinner("Analyzing..."):
                    try:
                        csv_text = df.to_string()
                        prompt = f"You're a helpful AI doctor assistant. Analyze this lab data:\n\n{csv_text}"
                        response = chat.send_message(prompt)
                        st.success("🤖 AI Analysis")
                        st.write(response.text)
                    except Exception as e:
                        st.error(f"Error analyzing with AI: {e}")

# ------------------------------------------
# 🦠 DISEASE SYMPTOMS PAGE
# ------------------------------------------
elif page == "🦠 Disease Symptoms":
    st.subheader("🔍 Disease Symptom Checker")
    st.markdown("Select a disease to view its common symptoms and basic advice.")

    disease = st.selectbox("Select disease", ["Malaria", "Asthma", "Typhoid", "Tuberculosis", "COVID-19"])

    if st.button("🔬 Get Symptoms"):
        with st.spinner("Fetching symptom data..."):
            try:
                prompt = f"What are the major symptoms and advice for treating {disease}?"
                response = chat.send_message(prompt)
                st.success(f"🧾 Symptoms of {disease}")
                st.write(response.text)
            except Exception as e:
                st.error(f"Error getting symptoms from Gemini: {e}")
