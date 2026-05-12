
from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = "paraphrase-MiniLM-L3-v2"
SAVE_PATH  = f"./models/{MODEL_NAME}"

os.makedirs(SAVE_PATH, exist_ok=True)

print(f"Downloading {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
model.save(SAVE_PATH)
print(f"Saved to {SAVE_PATH}")
print("Now run: python scripts/build_index.py")