import streamlit as st

st.set_page_config(
    page_title="AI Assignment Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Assignment Dashboard</h1>
        <p>Name Matching System & Recipe Chatbot</p>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        ### 📝 Task 1: Name Matcher
        Find the most similar names from a dataset using fuzzy matching and FAISS vector search.
    """)

with col2:
    st.markdown("""
        ### 🍳 Task 2: Recipe Bot
        Get recipe suggestions based on available ingredients using AI.
    """)

st.markdown("---")
st.markdown("### 👈 Select a task from the sidebar to get started!")
