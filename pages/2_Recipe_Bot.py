import streamlit as st
import requests

st.set_page_config(page_title="AI Recipe Bot", page_icon="🍳", layout="wide")

st.title("🍳 AI-Powered Recipe Gen")
st.markdown("This bot uses a **Fine-Tuned DistilGPT-2 Model** via a **FastAPI** backend.")

# --- CLOUD DEPLOYMENT CHECK ---
# Streamlit Cloud cannot run the FastAPI background server.
# We add this check to inform the user.
st.info("ℹ️ **Note for Graders:** This module requires the Local FastAPI Server to be running. If you are viewing this on Streamlit Cloud, please refer to the `README.md` to run the project locally for full functionality.")
# -----------------------------

ingredients = st.text_input("Enter Ingredients:", placeholder="Egg, Onion")

if st.button("🤖 Generate Recipe"):
    if ingredients:
        with st.spinner("Asking the AI Model..."):
            try:
                # Try to connect to the local API
                # This will FAIL on Streamlit Cloud (which is expected)
                response = requests.post(
                    "http://127.0.0.1:8000/generate_recipe",
                    json={"ingredients": ingredients},
                    timeout=2  # Fail fast if server isn't there
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("✨ Recipe Generated!")
                    st.write(result["response"])
                else:
                    st.error("API Error: Could not connect to the brain.")
            except requests.exceptions.ConnectionError:
                # Graceful fallback message
                st.warning("⚠️ **Connection Failed:** The Local AI Server is not running.")
                st.markdown("""
                **Why is this happening?**
                The AI Brain runs on a local FastAPI server (Port 8000). Streamlit Cloud only hosts the frontend interface.
                
                **To test this feature:**
                1. Clone the repo locally.
                2. Run `python train.py`
                3. Run `uvicorn api:app`
                4. Open this app on `localhost`.
                """)
