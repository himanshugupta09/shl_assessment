import os, sys, json
import numpy as np
import dotenv
dotenv.load_dotenv()
print("=" * 50)
print("SHL Deployment Pre-flight Check")
print("=" * 50)

errors = []
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
# 1. Check data files
for path in ["../data/shl_catalog.json", "../data/tfidf_vectorizer.pkl", "../data/tfidf_matrix.pkl"]:
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024
        print(f"  ✅ {path} ({size:.0f} KB)")
    else:
        print(f"  ❌ MISSING: {path}  →  run: python scripts/build_tfidf_index.py")
        errors.append(path)

# 2. Test retrieval if files exist
if not errors:
    import pickle
    from sklearn.metrics.pairwise import cosine_similarity

    with open("../data/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("../data/tfidf_matrix.pkl", "rb") as f:
        matrix = pickle.load(f)
    with open("../data/shl_catalog.json") as f:
        catalog = json.load(f)

    q = vectorizer.transform(["Java developer test"])
    scores = cosine_similarity(q, matrix).flatten()
    top = np.argsort(scores)[::-1][0]
    print(f"  ✅ TF-IDF search works — top result: '{catalog[top]['name']}'")

# 3. Check env
print("\n  Env vars:")
for var in ["GEMINI_API_KEY"]:
    val = os.getenv(var)
    print(f"    {'✅' if val else '❌'} {var}: {'set' if val else 'MISSING — add to .env and Railway Variables'}")
    if not val:
        errors.append(var)

print("\n" + "=" * 50)
if errors:
    print(f"❌ {len(errors)} issue(s) found — fix before deploying!")
    sys.exit(1)
else:
    print("✅ All checks passed — safe to push to Railway.")