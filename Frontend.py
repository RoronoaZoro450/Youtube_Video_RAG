import streamlit as st
import requests

API_URL = "https://youtube-video-rag-34pu.onrender.com"

st.set_page_config(
    page_title="YouTube RAG",
    page_icon="🎥",
    layout="centered"
)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = False
if "video_url" not in st.session_state:
    st.session_state.video_url = None

st.title("🎥 YouTube RAG Chatbot")
st.caption("Ask questions about any YouTube video's content")

# ---------------------------
# Video Ingestion
# ---------------------------

with st.container(border=True):
    st.subheader("📥 Load a Video")

    col1, col2 = st.columns([4, 1])

    with col1:
        video_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed"
        )

    with col2:
        ingest_clicked = st.button("Ingest", use_container_width=True, type="primary")

    if ingest_clicked:
        if not video_url:
            st.warning("Please enter a URL.")
        else:
            with st.spinner("Fetching transcript and building index..."):
                try:
                    response = requests.post(
                        f"{API_URL}/ingest_video",
                        json={"url": video_url},
                        timeout=120
                    )
                    if response.status_code == 200:
                        st.session_state.video_loaded = True
                        st.session_state.video_url = video_url
                        st.session_state.messages = []
                        st.success("Video loaded. Ask away below.")
                    else:
                        st.error(response.json().get("detail", "Ingestion failed."))
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {e}")

    # Status indicator
    if st.session_state.video_loaded:
        st.markdown(f"✅ **Loaded:** [{st.session_state.video_url}]({st.session_state.video_url})")
    else:
        st.markdown("⚪ No video loaded yet")

st.divider()

# ---------------------------
# Chat Interface
# ---------------------------

st.subheader("💬 Ask Questions")

if not st.session_state.video_loaded:
    st.info("Load a video above to start chatting.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input(
    "Ask something about the video...",
    disabled=not st.session_state.video_loaded
)

if query:

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={"query": query},
                    timeout=120
                )

                if response.status_code == 200:
                    answer = response.json()["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    error_msg = response.json().get("detail", "Something went wrong.")
                    st.error(error_msg)
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {e}")

st.divider()

# ---------------------------
# Controls
# ---------------------------

col1, col2 = st.columns(2)

with col1:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        requests.post(f"{API_URL}/clear_history")
        st.session_state.messages = []
        st.rerun()

with col2:
    if st.button("🔄 Reset Session", use_container_width=True):
        requests.post(f"{API_URL}/reset")
        st.session_state.messages = []
        st.session_state.video_loaded = False
        st.session_state.video_url = None
        st.rerun()