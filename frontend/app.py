import streamlit as st
import requests

# Configure page
st.set_page_config(page_title="Enterprise AI Copilot", page_icon="🤖", layout="wide")

st.title("Enterprise AI Copilot 🤖")
st.markdown("Your intelligent assistant for company knowledge and automated workflows.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question or request a task..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call FastAPI backend
    API_URL = "http://backend:8000/api/ask"
    
    with st.spinner("Thinking..."):
        try:
            response = requests.post(API_URL, json={"query": prompt})
            if response.status_code == 200:
                data = response.json()
                bot_reply = data.get("answer", "No answer provided.")
            else:
                bot_reply = f"Error: API returned status code {response.status_code}"
        except Exception:
            # Fallback for local testing without docker
            try:
                local_url = "http://localhost:8765/api/ask"
                response = requests.post(local_url, json={"query": prompt})
                if response.status_code == 200:
                    data = response.json()
                    bot_reply = data.get("answer", "No answer provided.")
                else:
                    bot_reply = f"Error: API returned status code {response.status_code}"
            except Exception as e2:
                bot_reply = f"Error connecting to backend: {e2}"

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
