import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb

# Load model and Chroma
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.get_or_create_collection(name="semantic_data")

st.title("🔍 MyGP Semantic Search")

query = st.text_input("Enter your question or keyword:")

if st.button("Search") and query:
    embedding = model.encode(query).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=5)

    st.markdown("## 🔎 Results")
    for doc in results["documents"][0]:
        st.markdown(f"- **{doc}**")
