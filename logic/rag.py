import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

class RAGEngine:
    def __init__(self, knowledge_dir="knowledge"):
        self.knowledge_dir = knowledge_dir
        # Load a lightweight, free embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = None
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        """Loads all .txt files and computes embeddings."""
        loaded_docs = []
        
        if not os.path.exists(self.knowledge_dir):
            os.makedirs(self.knowledge_dir)
            return

        for filename in os.listdir(self.knowledge_dir):
            if filename.endswith(".txt"):
                path = os.path.join(self.knowledge_dir, filename)
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                    # Simple chunking by paragraphs or max chars could be added here
                    # For now, we treat the whole file or large chunks as documents
                    if text.strip():
                        loaded_docs.append({"source": filename, "content": text})
        
        if loaded_docs:
            self.documents = loaded_docs
            texts = [doc["content"] for doc in loaded_docs]
            self.embeddings = self.embedder.encode(texts)

    def search(self, query, top_k=2):
        """Retrieves top-k relevant context."""
        if not self.documents:
            return []

        query_embedding = self.embedder.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append(self.documents[idx])
            
        return results

@st.cache_resource
def get_rag_engine():
    """Singleton pattern for Streamlit caching"""
    return RAGEngine()  