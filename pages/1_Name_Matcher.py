import streamlit as st
import pandas as pd
from fuzzywuzzy import fuzz
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Name Matcher", page_icon="📝", layout="wide")

NAMES_DATABASE = [
    "Geetha", "Gita", "Gitu", "Geetu", "Geetha Kumar",
    "Rajesh", "Rajesh Kumar", "Rajeev", "Raju", "Raja",
    "Priya", "Priyanka", "Priyam", "Preeti", "Prema",
    "Amit", "Amith", "Amitabh", "Aman", "Amar",
    "Sneha", "Snehal", "Snehil", "Snehi", "Sneh",
    "Vikram", "Vikrant", "Vikas", "Vinay", "Vishal",
    "Anjali", "Anjalee", "Anjana", "Anjum", "Anita",
    "Ramesh", "Ramesh Kumar", "Ramya", "Raman", "Ramu",
    "Kavya", "Kavita", "Kavi", "Kavitha", "Kavin"
]

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def create_faiss_index(_names):
    model = load_model()
    embeddings = model.encode(_names)
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))
    return index, model

class NameMatcher:
    def __init__(self, names):
        self.names = names
        self.faiss_index, self.model = create_faiss_index(names)
    
    def fuzzy_match(self, query, top_n=5):
        results = []
        for name in self.names:
            score = fuzz.ratio(query.lower(), name.lower())
            results.append((name, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]
    
    def faiss_match(self, query, top_n=5):
        query_emb = self.model.encode([query])
        faiss.normalize_L2(query_emb)
        distances, indices = self.faiss_index.search(query_emb.astype('float32'), top_n)
        results = [(self.names[idx], dist * 100) for idx, dist in zip(indices[0], distances[0])]
        return results

if 'matcher' not in st.session_state:
    st.session_state.matcher = NameMatcher(NAMES_DATABASE)

st.title("📝 Name Matching System")
st.markdown("Find similar names using **FuzzyWuzzy** and **FAISS**")

query = st.text_input("🔍 Enter a name:", placeholder="e.g., Geeta")
top_n = st.slider("Number of matches:", 3, 10, 5)

if query:
    st.markdown("---")
    fuzzy_results = st.session_state.matcher.fuzzy_match(query, top_n)
    
    st.markdown("### 🎯 Best Match")
    st.success(f"**{fuzzy_results[0][0]}** - Score: {fuzzy_results[0][1]}")
    
    st.markdown("### 🔤 Fuzzy Matching Results")
    df = pd.DataFrame(fuzzy_results, columns=["Name", "Score"])
    st.dataframe(df, use_container_width=True)
    
    st.markdown("### 🧠 FAISS Vector Results")
    faiss_results = st.session_state.matcher.faiss_match(query, top_n)
    df2 = pd.DataFrame(faiss_results, columns=["Name", "Score"])
    st.dataframe(df2, use_container_width=True)
