
import json
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

CATALOG_PATH = "../data/shl_catalog.json"
OUT_VEC      = "../data/tfidf_vectorizer.pkl"
OUT_MAT      = "../data/tfidf_matrix.pkl"

with open(CATALOG_PATH, "r", encoding="utf-8") as f:
    catalog = json.load(f)

# Build one text blob per catalog item for indexing
docs = [
    f"{item['name']} {item['test_type']} {item['description']}"
    for item in catalog
]

print(f"Building TF-IDF index over {len(docs)} items...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
matrix = vectorizer.fit_transform(docs)

with open(OUT_VEC, "wb") as f:
    pickle.dump(vectorizer, f)
with open(OUT_MAT, "wb") as f:
    pickle.dump(matrix, f)

print(f"Saved vectorizer → {OUT_VEC}")
print(f"Saved matrix     → {OUT_MAT}  shape={matrix.shape}")
print("Done. Now commit data/tfidf_vectorizer.pkl and data/tfidf_matrix.pkl")