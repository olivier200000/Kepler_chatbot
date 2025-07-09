import streamlit as st
from openai import OpenAI
import pandas as pd
import PyPDF2

# Initialize OpenAI client with API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Page settings
st.set_page_config(page_title="Doctor Assistant AI", layout="wide")

# App title
st.markdown("""
    <div style='text-align:center; padding: 10px'>
        <h1 style='color: #0078D4;'>🍺 Doctor Assistant AI</h1>
        <p style='font-size: 18px;'>Upload lab results, explore disease symptoms, or chat with your AI medical assistant instantly.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar navigation
page = st.sidebar.radio("🗭 Navigation", ["🏥 AI Chat", "📄 Upload Lab Results", "🥠 Disease Symptoms"])

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
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful and professional medical assistant."},
                        {"role": "user", "content": user_input}
                    ],
                    max_tokens=300
                )
                ai_reply = response.choices[0].message.content.strip()
                st.session_state.chat_history.append((user_input, ai_reply))
            st.session_state.user_question = ""

    for i, (q, a) in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**👨‍⚕️ Question {i+1}:** {q}")
            st.markdown(f"**🤖 Answer {i+1}:** {a}")
            st.markdown("---")

    st.text_input(
            st.write(response.choices[0].message.content)
