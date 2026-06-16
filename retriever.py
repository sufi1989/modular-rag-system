import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE_PATH = r"D:\RAG_Project - testing\embeddings_gpu_v1"

INDEX_FILE = f"{BASE_PATH}/faiss.index"
META_FILE = f"{BASE_PATH}/metadata.npy"

# ---------------- LOAD ----------------
index = faiss.read_index(INDEX_FILE)
texts = np.load(META_FILE, allow_pickle=True).tolist()

model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")

# ---------------- SEARCH FUNCTION ----------------
def search(query, k=5):
    query_vec = model.encode([query]).astype("float32")

    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, k)

    results = []
    for i, score in zip(indices[0], scores[0]):
        results.append((texts[i], float(score)))

    return results

# ---------------- TEST ----------------
query = "What is fine-tuning?"
results = search(query)

for i, (text, score) in enumerate(results):
    print(f"\nRank {i+1} | Score: {score:.4f}")
    print(text[:300])