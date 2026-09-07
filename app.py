import uuid
import streamlit as st
import httpx

st.set_page_config(page_title="Biology GraphRAG", page_icon="🧬")
st.title("🧬 Biology GraphRAG UI")

# FastAPI Backend URL
# API_URL = "http://localhost:8000/chat"
import os

# Uses container network if inside Docker, defaults to localhost if running bare-metal
BASE_API_URL = os.environ.get("BACKEND_API_URL", "http://localhost:8000")
API_URL = f"{BASE_API_URL.rstrip('/')}/chat"

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a biology question..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Connecting to backend API..."):
            try:
                # Make HTTP Request to decoupled FastAPI server
                payload = {"thread_id": st.session_state["thread_id"], "query": prompt}
                response = httpx.post(API_URL, json=payload, timeout=60.0)
                response.raise_for_status()
                
                data = response.json()
                answer = data["answer"]
                
                st.markdown(answer)
                with st.expander("Backend Metrics"):
                    st.json(data["metrics"])
                    
            except Exception as e:
                answer = f"⚠️ API Error: Ensure FastAPI server is running. ({e})"
                st.error(answer)
                
    st.session_state["messages"].append({"role": "assistant", "content": answer})