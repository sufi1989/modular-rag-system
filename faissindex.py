import os
import numpy as np
import faiss

# ---------------- PATH ----------------
BASE_PATH = r"D:\RAG_Project - testing\embeddings_gpu_v1"

EMB_FILE = os.path.join(BASE_PATH, "embeddings.npy")
TEXT_FILE = os.path.join(BASE_PATH, "emb_texts.npy")

INDEX_FILE = os.path.join(BASE_PATH, "faiss.index")
META_FILE = os.path.join(BASE_PATH, "metadata.npy")

# ---------------- LOAD DATA ----------------
print("Loading embeddings...")

embeddings = np.load(EMB_FILE).astype("float32")
texts = np.load(TEXT_FILE, allow_pickle=True).tolist()

print("Embeddings:", embeddings.shape)
print("Texts:", len(texts))

# ---------------- NORMALIZE (important for cosine similarity) ----------------
faiss.normalize_L2(embeddings)

# ---------------- BUILD INDEX ----------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)  # Inner Product = cosine after normalization
index.add(embeddings)

print("FAISS index built. Total vectors:", index.ntotal)

# ---------------- SAVE ----------------
faiss.write_index(index, INDEX_FILE)
np.save(META_FILE, np.array(texts, dtype=object))

print("Saved FAISS index + metadata")