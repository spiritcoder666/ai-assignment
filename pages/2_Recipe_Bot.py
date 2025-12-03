import streamlit as st
import requests
import time

st.set_page_config(page_title="AI Recipe Bot", page_icon="🍳", layout="wide")

st.title("🍳 AI-Powered Recipe Gen")
st.markdown("This bot uses a **Fine-Tuned DistilGPT-2 Model** via a **FastAPI** backend.")

# User Input
ingredients = st.text_input("Enter Ingredients:", placeholder="Egg, Onion")

if st.button("🤖 Generate Recipe"):
    if ingredients:
        with st.spinner("Asking the AI Model..."):
            try:
                # Call the Local FastAPI Server
                response = requests.post(
                    "http://127.0.0.1:8000/generate_recipe",
                    json={"ingredients": ingredients}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("✨ Recipe Generated!")
                    st.write(result["response"])
                else:
                    st.error("API Error: Could not connect to the brain.")
            except Exception as e:
                st.error(f"Connection Error: Is the API running? {e}")
