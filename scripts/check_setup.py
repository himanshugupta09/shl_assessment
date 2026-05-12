import os, sys, json
import numpy as np
import dotenv
print("=" * 50)
print("SHL Deployment Pre-flight Check")
print("=" * 50)
dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
errors = []

# 1. Check data files
for path in ["../data/shl_catalog.json", "../data/faiss_index.bin"]:
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024
        print(f"  ✅ {path} ({size:.0f} KB)")
    else:
        print(f"  ❌ MISSING: {path}")
        errors.append(path)

# 2. Check model
model_paths = [
    "../models/paraphrase-MiniLM-L3-v2",
    "../models/all-MiniLM-L6-v2",
]
model_found = None
for mp in model_paths:
    if os.path.exists(mp):
        print(f"  ✅ Model found: {mp}")
        model_found = mp
        break
if not model_found:
    print("  ❌ MISSING: No model found in ./models/")
    errors.append("model")

# 3. Check FAISS + model dimension match
if not errors:
    import faiss
    from sentence_transformers import SentenceTransformer

    index = faiss.read_index("../data/faiss_index.bin")
    print(f"\n  FAISS index: {index.ntotal} vectors, dim={index.d}")

    model = SentenceTransformer(model_found)
    test_vec = model.encode(["test query"], convert_to_numpy=True)
    print(f"  Model output dim: {test_vec.shape[1]}")

    if index.d != test_vec.shape[1]:
        print(f"\n  ❌ DIMENSION MISMATCH: index={index.d}, model={test_vec.shape[1]}")
        print("     You must rebuild faiss_index.bin with the current model.")
        print("     Run: python scripts/build_index.py")
        errors.append("dimension_mismatch")
    else:
        print("  ✅ Dimensions match — FAISS index is compatible with model")

    # 4. Quick search test
    test_vec = test_vec.astype(np.float32)
    faiss.normalize_L2(test_vec)
    D, I = index.search(test_vec, 3)
    print(f"  ✅ FAISS search works — top result index: {I[0][0]}")

    # 5. Catalog load
    with open("../data/shl_catalog.json") as f:
        catalog = json.load(f)
    print(f"  ✅ Catalog loaded: {len(catalog)} items")

# 6. Check env
print("\n  Env vars:")
for var in ["GEMINI_API_KEY", "MODEL_PATH", "CATALOG_PATH", "INDEX_PATH"]:
    val = os.getenv(var)
    if var == "GEMINI_API_KEY":
        print(f"    {'✅' if val else '❌'} {var}: {'set' if val else 'MISSING'}")
    else:
        print(f"    {'✅' if val else '⚠️ '} {var}: {val or 'not set (using default)'}")

print("\n" + "=" * 50)
if errors:
    print(f"❌ {len(errors)} issue(s) found — fix before deploying!")
    sys.exit(1)
else:
    print("✅ All checks passed — safe to push to Railway.")