import os
import streamlit as st
import tempfile
from pathlib import Path

# Import backend functions
from text import (
    summarize_notes,
    answer_question,
    split_into_sentences,
    _load_env_file
)
from textPDF import extract_text_from_pdf
from ocrimage import extract_text_from_image

# Ensure env file is loaded
_load_env_file()

# Configure Streamlit page config
st.set_page_config(
    page_title="Agentic AI Lecture Notes Summarizer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium stylesheet
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Base typography */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header titles */
    .app-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sidebar-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e293b;
    }
    
    /* Metrics cards styling */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
    }
    
    /* Provider status indicator */
    .provider-badge {
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1d4ed8;
        font-weight: 600;
        font-size: 0.8rem;
        padding: 5px 12px;
        border-radius: 9999px;
        display: inline-flex;
        align-items: center;
        margin-top: 8px;
    }
    
    /* Summary container border highlight */
    .summary-card {
        border-left: 5px solid #2563eb !important;
        background-color: rgba(37, 99, 235, 0.02);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
    }
    
    .stButton>button {
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to get text metrics
def get_metrics(text):
    words = len(text.split())
    sentences = len(split_into_sentences(text))
    chars = len(text)
    return words, sentences, chars

# Display Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-title">📚 Agentic AI Summarizer</div>', unsafe_allow_html=True)
    st.write("Upload lecture notes, PDFs, or handwritten notes and generate AI-powered summaries and answers.")
    
    st.divider()
    
    st.markdown("### ⚙️ Configuration")
    
    # Show active LLM Provider
    provider = os.getenv("LLM_PROVIDER", "openai").strip().upper()
    st.markdown(f"**Active LLM Provider:**")
    st.markdown(f'<div class="provider-badge">🤖 {provider} Active</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Input Mode Selector
    st.markdown("### 📥 Select Input Mode")
    input_mode = st.radio(
        label="Choose how to upload your notes:",
        options=["📝 Text Notes", "📄 PDF Notes", "📷 Handwritten Notes"],
        label_visibility="collapsed"
    )

# Clean/normalize mode key
mode_key = "text"
if "PDF" in input_mode:
    mode_key = "pdf"
elif "Handwritten" in input_mode:
    mode_key = "image"

# Initialize Session State
if f"{mode_key}_notes" not in st.session_state:
    st.session_state[f"{mode_key}_notes"] = ""
if f"{mode_key}_summary" not in st.session_state:
    st.session_state[f"{mode_key}_summary"] = ""
if f"{mode_key}_qa_history" not in st.session_state:
    st.session_state[f"{mode_key}_qa_history"] = []

# Main Container
st.markdown('<div class="app-title">Agentic AI Lecture Notes Summarizer</div>', unsafe_allow_html=True)
st.write("A clean academic workspace powered by advanced Agentic AI model integrations.")

st.divider()

# Mode specific layouts
if mode_key == "text":
    st.subheader("📝 Enter Text Notes")
    text_input = st.text_area(
        label="Lecture notes content",
        placeholder="Paste your lecture notes here...",
        height=300,
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        submit = st.button("Generate Summary", use_container_width=True, type="primary")
        
    if submit:
        if not text_input.strip():
            st.error("Please enter some lecture notes first.")
        else:
            with st.spinner("Generating summary..."):
                st.session_state["text_notes"] = text_input.strip()
                st.session_state["text_summary"] = summarize_notes(st.session_state["text_notes"])
                # Clear previous Q&A history for fresh run
                st.session_state["text_qa_history"] = []

elif mode_key == "pdf":
    st.subheader("📄 Upload PDF Notes")
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        file_size_kb = len(uploaded_file.getvalue()) / 1024
        st.info(f"**Filename:** {uploaded_file.name} | **Size:** {file_size_kb:.2f} KB")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submit = st.button("Analyze PDF", use_container_width=True, type="primary")
            
        if submit:
            with st.spinner("Analyzing document..."):
                # Write to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = Path(tmp_file.name)
                
                try:
                    notes = extract_text_from_pdf(tmp_path)
                    st.session_state["pdf_notes"] = notes
                    st.session_state["pdf_summary"] = summarize_notes(notes)
                    st.session_state["pdf_qa_history"] = []
                except Exception as e:
                    st.error(f"Error reading PDF: {e}")
                finally:
                    # Clean up temp file
                    if tmp_path.exists():
                        tmp_path.unlink()

elif mode_key == "image":
    st.subheader("📷 Upload Handwritten Notes Image")
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Handwritten Notes Preview", width=600)
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submit = st.button("Extract & Analyze", use_container_width=True, type="primary")
            
        if submit:
            with st.spinner("Extracting notes..."):
                suffix = Path(uploaded_file.name).suffix
                # Write to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = Path(tmp_file.name)
                
                try:
                    notes = extract_text_from_image(tmp_path)
                    st.session_state["image_notes"] = notes
                    st.session_state["image_summary"] = summarize_notes(notes)
                    st.session_state["image_qa_history"] = []
                except Exception as e:
                    st.error(f"Error during OCR extraction: {e}")
                finally:
                    # Clean up temp file
                    if tmp_path.exists():
                        tmp_path.unlink()


# Display Results if we have data
if st.session_state[f"{mode_key}_notes"]:
    st.divider()
    
    # Display Stats/Metrics
    words, sentences, chars = get_metrics(st.session_state[f"{mode_key}_notes"])
    st.markdown("### 📊 Document Metrics")
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric(label="Words", value=f"{words:,}")
    m_col2.metric(label="Sentences", value=f"{sentences:,}")
    m_col3.metric(label="Characters", value=f"{chars:,}")
    
    st.divider()
    
    # Extracted notes (only for PDF & Image, or if explicitly needed)
    if mode_key in ["pdf", "image"]:
        with st.expander("🔍 Extracted Notes Content", expanded=False):
            st.text_area(
                label="extracted_raw_content",
                value=st.session_state[f"{mode_key}_notes"],
                height=250,
                disabled=True,
                label_visibility="collapsed"
            )
            st.download_button(
                label="📥 Download Extracted Notes",
                data=st.session_state[f"{mode_key}_notes"],
                file_name=f"{mode_key}_extracted_notes.txt",
                mime="text/plain"
            )
            
    # AI Summary
    st.markdown("### ✨ AI Summary")
    with st.container(border=True):
        st.markdown(st.session_state[f"{mode_key}_summary"])
        st.download_button(
            label="📥 Download Summary",
            data=st.session_state[f"{mode_key}_summary"],
            file_name=f"{mode_key}_summary.txt",
            mime="text/plain"
        )
        
    st.divider()
    
    # Question Answering
    st.markdown("### 💬 Ask Questions About the Notes")
    
    q_col1, q_col2 = st.columns([4, 1])
    with q_col1:
        question_input = st.text_input(
            label="Enter your question...",
            placeholder="What is the main argument? / What does X refer to?",
            label_visibility="collapsed",
            key=f"{mode_key}_q_in"
        )
    with q_col2:
        ask_btn = st.button("Ask", use_container_width=True)
        
    if ask_btn and question_input.strip():
        with st.spinner("Analyzing..."):
            ans = answer_question(question_input.strip(), st.session_state[f"{mode_key}_notes"])
            st.session_state[f"{mode_key}_qa_history"].append((question_input.strip(), ans))
            
    # Display Q&A History
    if st.session_state[f"{mode_key}_qa_history"]:
        st.markdown("**Conversation:**")
        for q, a in reversed(st.session_state[f"{mode_key}_qa_history"]):
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(a)
