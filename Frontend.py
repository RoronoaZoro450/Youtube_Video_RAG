import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="YouTube RAG",
    page_icon="🎥"
)

st.title("🎥 YouTube RAG Chatbot")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# Video Ingestion
# ---------------------------

st.header("Ingest YouTube Video")

video_url = st.text_input(
    "YouTube URL",
    placeholder="https://www.youtube.com/watch?v=..."
)

if st.button("Ingest Video"):
    if not video_url:
        st.warning("Please enter a URL.")
    else:
        with st.spinner("Processing video..."):
            response = requests.post(
                f"{API_URL}/ingest_video",
                json={"url": video_url}
            )

        if response.status_code == 200:
            st.success("Video ingested successfully!")
        else:
            st.error(response.json()["detail"])

st.divider()

# ---------------------------
# Chat Interface
# ---------------------------

st.header("Ask Questions")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask something about the video...")

if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = requests.post(
                f"{API_URL}/query",
                json={"query": query}
            )

            if response.status_code == 200:

                answer = response.json()["answer"]

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            else:
                st.error(response.json()["detail"])

st.divider()

# ---------------------------
# Controls
# ---------------------------

col1, col2 = st.columns(2)

with col1:
    if st.button("Clear Chat"):
        requests.post(f"{API_URL}/clear_history")
        st.session_state.messages = []
        st.rerun()

with col2:
    if st.button("Reset Session"):
        requests.post(f"{API_URL}/reset")
        st.session_state.messages = []
        st.rerun()