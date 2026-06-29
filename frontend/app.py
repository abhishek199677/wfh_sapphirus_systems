import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Enterprise AI Copilot", page_icon="🤖", layout="wide")

st.title("Enterprise AI Copilot 🤖")
st.markdown("Your intelligent assistant for company knowledge and automated workflows.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question or request a task..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/api/ask",
                json={"query": prompt},
                timeout=60,
            )
            response.raise_for_status()
            bot_reply = response.json().get("answer", "No answer provided.")
        except requests.exceptions.Timeout:
            bot_reply = "Error: Request timed out. Please try again."
        except requests.exceptions.ConnectionError:
            bot_reply = f"Error: Cannot connect to backend at {BACKEND_URL}. Ensure the server is running."
        except requests.exceptions.HTTPError as e:
            bot_reply = f"Error: Backend returned {e.response.status_code}"
        except Exception as e:
            bot_reply = f"Error: {e}"

    with st.chat_message("assistant"):
        st.markdown(bot_reply)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
