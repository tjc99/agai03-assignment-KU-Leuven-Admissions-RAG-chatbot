# app.py
import os
import sys
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Core Path alignment logic for standard repo deployment
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "src"))

from chatbot import generate_final_response

# 1. Load environment configurations
load_dotenv(os.path.join(script_dir, ".env"))
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# 2. Target standardized relative paths
db_dir = os.path.join(script_dir, "data", "chroma_db")
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(persist_directory=db_dir, embedding_function=embeddings_model)

def local_rag_brain(user_query: str):
    """Orchestrates standard modular lookup and guardrail refusals."""
    retrieved_docs = vector_store.similarity_search(user_query, k=4)
    context_blocks = []
    sources = set()
    
    for doc in retrieved_docs:
        context_blocks.append(doc.page_content)
        source_id = doc.metadata.get("source", "Official Portal")
        sources.add(os.path.basename(source_id))
        
    unified_context = "\n\n---\n\n".join(context_blocks)
    
    # Invoke our modular chatbot logic from src/
    final_answer = generate_final_response(user_query, unified_context, api_key)
        
    if "cannot find" in final_answer.lower() or "sorry" in final_answer.lower():
        return {"answer": final_answer, "sources": []}
        
    return {"answer": final_answer, "sources": list(sources) if sources else []}

# ==============================================================================
# STREAMLIT WINDOW RENDERING
# ==============================================================================
st.set_page_config(page_title="KU Leuven AI Advisor", page_icon="🎓", layout="centered")
st.title("🎓 KU Leuven Admissions AI Advisor")
st.caption("Standardized RAG Architecture Deployment | Powered by Gemini 2.5 & ChromaDB")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask about application deadlines, tuition fees, etc..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        with st.spinner("Searching official university database..."):
            brain_output = local_rag_brain(user_input)
            bot_reply = brain_output["answer"]
            referenced_sources = brain_output["sources"]
            
            ui_display_text = bot_reply
            if referenced_sources:
                ui_display_text += "\n\n**📍 Verified Reference Sources:**"
                for source in referenced_sources:
                    ui_display_text += f"\n- `{source}`"
                    
            st.markdown(ui_display_text)
            st.session_state.messages.append({"role": "assistant", "content": ui_display_text})