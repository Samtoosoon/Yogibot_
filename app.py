import streamlit as st
from logic.rag import get_rag_engine
from logic.llm import YogiLLM

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Yogi AI Suite",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- UI & CSS OVERHAUL ---
# This CSS hides the password 'eye' and styles the containers
st.markdown("""
    <style>
    /* 1. HIDE THE API KEY 'EYE' ICON */
    button[aria-label="Show password"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* 2. GENERAL UI CLEANUP */
    .main .block-container {
        padding-top: 2rem;
    }
    
    /* 3. CARD-LIKE STYLING FOR COLUMNS */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 10px;
    }
    
    /* 4. BUTTON STYLING */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        height: 3em;
    }
    
    /* 5. TEXT AREA FONT */
    .stTextArea textarea {
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.title("🧘 Yogi AI Suite")
    st.caption("Engineering Productivity System")
    st.markdown("---")
    
    # API KEY INPUT (SECURE)
    st.subheader("🔐 Access Control")
    api_key = st.text_input(
        "Google Gemini API Key", 
        type="password", 
        placeholder="Paste key here (hidden)",
        help="The 'Show Password' button has been disabled for security."
    )
    
    st.markdown("---")
    
    # NAVIGATION
    nav_selection = st.radio(
        "System Module", 
        ["Yogi Coding Assistant", "Article Summarizer"],
        captions=["Generate & Debug", "Digest Content"]
    )
    
    st.markdown("---")
    st.info("System Status: **Online**\nRAG Engine: **Active**")

# --- INITIALIZE LOGIC ---
if api_key:
    try:
        llm_engine = YogiLLM(api_key)
        rag_engine = get_rag_engine()
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.stop()
else:
    # LANDING STATE (When no key is entered)
    st.markdown("### 🛑 System Locked")
    st.warning("Please enter your Google Gemini API Key in the sidebar to activate the Neural Engine.")
    st.stop()

# ==========================================
# FEATURE A: YOGI CODING ASSISTANT
# ==========================================
if nav_selection == "Yogi Coding Assistant":
    
    # Title Section
    st.markdown("## 👨‍💻 **Coding Architect**")
    st.caption("Mode: Engineering & Systems Design")
    
    # Layout: 40% Control Panel | 60% Output
    col1, col2 = st.columns([2, 3], gap="medium")
    
    with col1:
        with st.container(border=True):
            st.markdown("### 🛠️ Input Parameters")
            
            c1, c2 = st.columns(2)
            with c1:
                mode = st.selectbox("Operation Mode", ["Generate Code", "Debug Code", "Explain Concept"])
            with c2:
                language = st.selectbox("Target Language", ["Python", "JavaScript", "Go", "SQL", "HTML/CSS"])
            
            user_query = st.text_area("Primary Query", height=120, placeholder="Describe the engineering problem...")
            
            with st.expander("📝 Additional Context (Code/Errors)"):
                code_snippet = st.text_area("Code Snippet", height=150, placeholder="Paste existing code here...")
                error_msg = st.text_input("Error Logs", placeholder="Paste error output...")

            st.markdown("<br>", unsafe_allow_html=True)
            submit_btn = st.button("🚀 Execute Protocol", type="primary")

    with col2:
        with st.container(border=True):
            st.markdown("### ⚡ System Output")
            
            if submit_btn:
                if not user_query:
                    st.error("Input Friction: Query field is empty.")
                else:
                    with st.status("Processing...", expanded=True) as status:
                        st.write("🔍 Retrieving Knowledge Base vectors...")
                        retrieved_docs = rag_engine.search(user_query + " " + language)
                        
                        st.write("🧠 Generating Engineering Solution...")
                        try:
                            response = llm_engine.generate_code_response(
                                mode, user_query, code_snippet, error_msg, retrieved_docs
                            )
                            status.update(label="Complete", state="complete", expanded=False)
                            
                            # RENDER RESPONSE
                            st.markdown(response)
                            
                            # RAG SOURCES FOOTER
                            if retrieved_docs:
                                st.divider()
                                st.caption(f"📚 Context referenced from {len(retrieved_docs)} internal documents.")
                                with st.expander("View Source Context"):
                                    for doc in retrieved_docs:
                                        st.markdown(f"**File:** `{doc['source']}`")
                                        st.code(doc['content'][:300] + "...", language="text")
                                        
                        except Exception as e:
                            status.update(label="Error", state="error")
                            st.error(f"LLM Error: {e}")
            else:
                st.info("Awaiting Input... Configure parameters on the left.")

# ==========================================
# FEATURE B: ARTICLE SUMMARIZER
# ==========================================
elif nav_selection == "Article Summarizer":
    st.markdown("## 📄 **Information Compressor**")
    st.caption("Mode: High-Signal Extraction")
    
    with st.container(border=True):
        input_text = st.text_area("Source Material", height=250, placeholder="Paste article or documentation text here...")
        
        col_act, col_info = st.columns([1, 4])
        with col_act:
            sum_btn = st.button("📉 Summarize", type="primary")
        
        if sum_btn:
            if input_text:
                with st.spinner("Extracting insights..."):
                    try:
                        summary = llm_engine.summarize_article(input_text)
                        st.markdown("### 💡 Key Insights")
                        st.markdown(summary)
                    except Exception as e:
                        st.error(f"Processing Error: {e}")
            else:
                st.warning("Input Friction: No text provided to summarize.")