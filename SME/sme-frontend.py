import streamlit as st
import requests
import json
import uuid
from pathlib import Path
import os
import time
from typing import List, Dict

API_BASE_URL = "http://localhost:8000"
DOCS_FOLDER = Path("./Docs")
GENERATED_FILES_FOLDER = Path("./generated_files")

DOCS_FOLDER.mkdir(exist_ok=True)
GENERATED_FILES_FOLDER.mkdir(exist_ok=True)


def get_documents_list() -> List[str]:
    """Get list of documents in the Docs folder"""
    if not DOCS_FOLDER.exists():
        return []
    
    documents = []
    supported_extensions = ['.pdf', '.docx', '.pptx', '.txt', '.md']
    for file_path in DOCS_FOLDER.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            documents.append(file_path.name)
    
    return sorted(documents)

def get_generated_files() -> List[str]:
    """Get list of generated files"""
    if not GENERATED_FILES_FOLDER.exists():
        return []
    
    files = []
    for file_path in GENERATED_FILES_FOLDER.iterdir():
        if file_path.is_file():
            files.append(file_path.name)
    
    return sorted(files, reverse=True)

def delete_document(filename: str) -> bool:
    """Delete a document from the Docs folder"""
    try:
        file_path = DOCS_FOLDER / filename
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    except Exception as e:
        st.error(f"Failed to delete {filename}: {str(e)}")
        return False

st.set_page_config(
    page_title="SME RAG Agent",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    /* Dark theme main container */
    .main {
        background-color: #1a1a2e;
    }
    
    /* Sidebar - Darker purple */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16213e 0%, #0f3460 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #eee !important;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: #16213e;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        border: 1px solid #2a2d3a;
    }
    
    /* All text white by default */
    .main * {
        color: #eee;
    }
    
    /* Info boxes - Cyberpunk blue */
    div[data-baseweb="notification"][kind="info"] {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%) !important;
        border-left: 4px solid #00b8d4 !important;
        color: #000 !important;
    }
    
    div[data-baseweb="notification"][kind="info"] * {
        color: #000 !important;
    }
    
    /* Success boxes - Neon green */
    div[data-baseweb="notification"][kind="success"] {
        background: linear-gradient(135deg, #00ff88 0%, #00cc66 100%) !important;
        border-left: 4px solid #00e676 !important;
        color: #000 !important;
    }
    
    div[data-baseweb="notification"][kind="success"] * {
        color: #000 !important;
    }
    
    /* Error boxes - Hot pink */
    div[data-baseweb="notification"][kind="error"] {
        background: linear-gradient(135deg, #ff006e 0%, #cc0055 100%) !important;
        border-left: 4px solid #f50057 !important;
        color: #fff !important;
    }
    
    div[data-baseweb="notification"][kind="error"] * {
        color: #fff !important;
    }
    
    /* Buttons - Neon accent */
    .stButton button {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: #000;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.6);
    }
    
    /* Download buttons */
    .stDownloadButton button {
        background: linear-gradient(135deg, #00ff88 0%, #00cc66 100%);
        color: #000;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
    }
    
    .stDownloadButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 30px rgba(0, 255, 136, 0.6);
    }
    
    /* Titles - Neon glow */
    h1 {
        color: #00d4ff;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    
    h2, h3 {
        color: #00ff88;
        text-shadow: 0 0 8px rgba(0, 255, 136, 0.3);
    }
    
    /* Input styling */
    .stChatInputContainer {
        border-top: 2px solid #00d4ff;
        background-color: #16213e;
        box-shadow: 0 -4px 20px rgba(0, 212, 255, 0.2);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #16213e;
        border: 1px solid #2a2d3a;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #00d4ff !important;
    }

    /* Custom small button style for file delete */
    .small-delete-btn button {
        background: linear-gradient(135deg, #ff006e 0%, #cc0055 100%);
        color: #fff;
        font-size: 0.75rem;
        padding: 0.1rem 0.5rem;
        height: auto;
        line-height: 1;
        box-shadow: none;
    }

    .small-delete-btn button:hover {
        box-shadow: 0 0 10px rgba(255, 0, 110, 0.6);
    }
</style>
""", unsafe_allow_html=True)

if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'model_name' not in st.session_state:
    st.session_state.model_name = "all-mpnet-base-v2"

with st.sidebar:
    st.title("🤖 SME RAG Agent")
    st.markdown("---")
    
    st.session_state.model_name = st.selectbox(
        "⚙️ Embedding Model",
        ["all-mpnet-base-v2", "BAAI/bge-base-en-v1.5"]
    )
    
    if st.button("🆕 New Chat"):
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.success("New chat created!")
        st.rerun()
    
    st.markdown(f"**🔑 Chat ID:** `{st.session_state.conversation_id[:8]}...`")
    
    st.markdown("---")
    st.subheader("📚 Input Documents (Docs/)")
    
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=['pdf', 'docx', 'pptx', 'txt', 'md'],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("📤 Upload", key="upload_btn_side"):
        for file in uploaded_files:
            path = DOCS_FOLDER / file.name
            with open(path, "wb") as f:
                f.write(file.getbuffer())
        st.success(f"✅ Uploaded {len(uploaded_files)} file(s). Ingestion triggered.")
        st.rerun() 
    
    documents = get_documents_list()
    if documents:
        st.markdown("**Current Corpus Files:**")
        
        for doc in documents:
            col_doc, col_del = st.columns([5, 1], gap="small")
            
            with col_doc:
                st.markdown(f"`{doc}`")
            
            with col_del:
                if st.button("🗑️", key=f"del_{doc}", help=f"Delete {doc}"):
                    if delete_document(doc):
                        st.success(f"Deleted {doc}. Re-ingestion triggered.")
                        time.sleep(0.5)
                        st.rerun()
        st.caption(f"Total: {len(documents)} files.")
    else:
        st.info("No documents found in Docs/")

    st.markdown("---")
    st.subheader("📄 Generated Files")
    
    generated_files = get_generated_files()
    if generated_files:
        st.markdown("**Last 10 Generated Files:**")
        
        for file in generated_files[:10]:
            file_path = GENERATED_FILES_FOLDER / file
            if file_path.exists():
                with open(file_path, "rb") as f:
                    st.download_button(
                        label=f"📥 {file}",
                        data=f,
                        file_name=file,
                        mime="application/octet-stream",
                        key=f"dl_{file}_side"
                    )
        st.caption(f"Total: {len(generated_files)} files in generated_files/")
    else:
        st.info("No generated files found.")

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <p style='font-size: 0.9em; opacity: 0.7;'>🌙 Dark Mode Enabled</p>
    </div>
    """, unsafe_allow_html=True)

st.title("💬 Chat with SME Agent")
st.caption(f"🎯 Model: {st.session_state.model_name}")

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        if msg['role'] == 'user':
            st.write(msg['content'])
        else:
            if 'thought' in msg and msg['thought']:
                st.info(f"💭 **Thought Process:**\n\n{msg['thought']}")
            
            if 'answer' in msg and msg['answer']:
                st.success(f"✅ **Answer:**\n\n{msg['answer']}")
            
            if 'file_path' in msg and msg['file_path']:
                st.info(f"📄 **File Generated:** `{msg['file_path']}`")
                file_path = Path(msg['file_path'])
                if file_path.exists():
                    with open(file_path, "rb") as f:
                        st.download_button(
                            f"📥 Download {file_path.name}",
                            f,
                            file_name=file_path.name,
                            key=f"dl_chat_{file_path.name}"
                        )
            
            if 'error' in msg and msg['error']:
                st.error(f"⚠️ **Error:** {msg['error']}")
            
            if 'final_answer' in msg and msg['final_answer']:
                if not msg.get('answer'):
                    st.write(msg['final_answer'])

if prompt := st.chat_input("✨ Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        placeholder = st.empty()

        assistant_message = {
            "role": "assistant",
            "plan": None, "thought": "", "answer": "", "file_path": None,
            "error": None, "final_answer": None
        }

        with st.spinner("🔮 Processing..."):
            try:
                url = f"{API_BASE_URL}/agent/invoke_stream"
                payload = {
                    "query": prompt,
                    "model_name": st.session_state.model_name,
                    "conversation_id": st.session_state.conversation_id
                }
                
                response = requests.post(url, json=payload, stream=True, timeout=300)
                response.raise_for_status()
                
                for line in response.iter_lines(decode_unicode=True):
                    if line and line.startswith('data: '):
                        json_str = line[6:].strip()
                        
                        if json_str.startswith('data: '):
                            json_str = json_str[6:].strip()

                        try:
                            event = json.loads(json_str)
                            event_type = event.get('type')
                            content = event.get('content', '')
                            
                            if event_type == 'thought' and content:
                                assistant_message['thought'] = content
                            elif event_type == 'answer' and content:
                                assistant_message['answer'] = content
                            elif event_type == 'file_generated' and content:
                                assistant_message['file_path'] = content
                            elif event_type == 'error' and content:
                                assistant_message['error'] = content
                            elif event_type == 'final_answer' and content and not assistant_message['answer']:
                                assistant_message['final_answer'] = content

                            with placeholder.container():
                                if assistant_message.get('thought'):
                                    st.info(f"💭 **Thought Process:**\n\n{assistant_message['thought']}")
                                if assistant_message.get('answer'):
                                    st.success(f"✅ **Answer:**\n\n{assistant_message['answer']}")
                                if assistant_message.get('file_path'):
                                    st.info(f"📄 **File Generated:** `{assistant_message['file_path']}`")
                                    file_path = Path(assistant_message['file_path'])
                                    if file_path.exists():
                                        with open(file_path, "rb") as f:
                                            dynamic_key = f"dl_live_{file_path.name}_{hash(content)}" 
                                            st.download_button(
                                                f"📥 Download {file_path.name}", f, file_name=file_path.name,
                                                key=dynamic_key
                                            )
                                if assistant_message.get('error'):
                                    st.error(f"⚠️ **Error:** {assistant_message['error']}")
                                elif assistant_message.get('final_answer') and not assistant_message.get('answer'):
                                    st.write(assistant_message['final_answer'])

                        except json.JSONDecodeError:
                            continue

                placeholder.empty()

            except requests.exceptions.ConnectionError:
                assistant_message['error'] = "❌ Cannot connect to backend. Make sure it's running on http://localhost:8000"
            except Exception as e:
                assistant_message['error'] = f"❌ Error during response: {str(e)}"

        with placeholder.container():
            if assistant_message.get('thought'):
                st.info(f"💭 **Thought Process:**\n\n{assistant_message['thought']}")
            
            if assistant_message.get('answer'):
                st.success(f"✅ **Answer:**\n\n{assistant_message['answer']}")
            
            if assistant_message.get('file_path'):
                st.info(f"📄 **File Generated:** `{assistant_message['file_path']}`")
                file_path = Path(assistant_message['file_path'])
                if file_path.exists():
                    with open(file_path, "rb") as f:
                        st.download_button(
                            f"📥 Download {file_path.name}", f, file_name=file_path.name,
                            key=f"dl_final_{file_path.name}"
                        )
            
            if assistant_message.get('error'):
                st.error(f"⚠️ **Error:** {assistant_message['error']}")
            
            if assistant_message.get('final_answer') and not assistant_message.get('answer'):
                st.write(assistant_message['final_answer'])

        st.session_state.messages.append({"role": "assistant", **assistant_message})


st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 1rem; opacity: 0.6;'>
    <p>Powered by LangGraph × Pinecone × Gemini | Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)