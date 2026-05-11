import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

model.save('./models/all-MiniLM-L6-v2')  # commit this to git
INPUT_JSON = "../data/shl_catalog.json"
OUTPUT_INDEX = "../data/faiss_index.bin"

def build_vector_index():
    print("Loading catalog data...")
    if not os.path.exists(INPUT_JSON):
        print(f"Error: {INPUT_JSON} not found. Run scrape_catalog.py first.")
        return

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        catalog = json.load(f)

    # We will embed the combination of the test name and its description
    texts_to_embed = [f"{item['name']}: {item['description']}" for item in catalog]

    print("Loading embedding model (this may take a minute on first run)...")
    # all-MiniLM-L6-v2 is a great, fast, open-source model for local embeddings
    

    print(f"Generating embeddings for {len(texts_to_embed)} items...")
    embeddings = model.encode(texts_to_embed, convert_to_numpy=True)
    
    # Normalize embeddings to use Cosine Similarity with FAISS Inner Product
    faiss.normalize_L2(embeddings)

    # Create a FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) # Inner Product for cosine similarity
    
    print("Adding vectors to FAISS index...")
    index.add(embeddings)

    # Save the index to disk
    faiss.write_index(index, OUTPUT_INDEX)
    print(f"FAISS index saved to {OUTPUT_INDEX}")

if __name__ == "__main__":
    build_vector_index()