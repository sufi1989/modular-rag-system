import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# ---------------- PATHS ----------------
BASE_PATH = r"D:\RAG_Project - testing"

INPUT_FILE = os.path.join(BASE_PATH, "clean_chunks.npy")

OUTPUT_DIR = os.path.join(BASE_PATH, "embeddings_gpu_v1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "embeddings.npy")
TEXTS_FILE = os.path.join(OUTPUT_DIR, "emb_texts.npy")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "embed_checkpoint.npy")
BAD_LOG_FILE = os.path.join(OUTPUT_DIR, "bad_batches.txt")

# ---------------- LOAD DATA ----------------
texts = np.load(INPUT_FILE, allow_pickle=True).tolist()
print("Loaded chunks:", len(texts))

# ---------------- DEVICE ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ---------------- MODEL ----------------
model = SentenceTransformer(
    "BAAI/bge-large-en-v1.5",
    device=device
)

# ---------------- SETTINGS ----------------
batch_size = 32

embeddings = []
aligned_texts = []
failed_batches = []

# ---------------- CLEAN FUNCTION ----------------
def safe_text(t):
    if not isinstance(t, str):
        return None
    t = t.replace("\x00", " ").strip()
    if len(t) < 20:
        return None
    return t

# ---------------- RESUME ----------------
start_idx = 0

if os.path.exists(CHECKPOINT_FILE):
    data = np.load(CHECKPOINT_FILE, allow_pickle=True).item()
    start_idx = data.get("index", 0)
    embeddings = list(data.get("embeddings", []))
    aligned_texts = list(data.get("texts", []))
    print("Resuming from:", start_idx)

# ---------------- EMBEDDING LOOP ----------------
with torch.no_grad():

    for i in range(start_idx, len(texts), batch_size):

        batch = texts[i:i + batch_size]

        clean_batch = [safe_text(t) for t in batch]
        clean_batch = [t for t in clean_batch if t]

        if not clean_batch:
            continue

        success = False

        for attempt in range(3):
            try:
                emb = model.encode(
                    clean_batch,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                success = True
                break
            except Exception as e:
                if attempt == 2:
                    failed_batches.append(i)
                    with open(BAD_LOG_FILE, "a", encoding="utf-8") as f:
                        f.write(f"\nFAILED BATCH {i}\n")
                        for t in clean_batch:
                            f.write(repr(t) + "\n")

        if success:
            embeddings.extend(emb)
            aligned_texts.extend(clean_batch)
        else:
            failed_batches.append(i)

        if i % 2000 == 0:
            np.save(
                CHECKPOINT_FILE,
                {
                    "index": i,
                    "embeddings": embeddings,
                    "texts": aligned_texts
                }
            )
            print("Checkpoint saved at:", i)

# ---------------- FINAL SAVE ----------------
np.save(EMBEDDINGS_FILE, np.array(embeddings))
np.save(TEXTS_FILE, np.array(aligned_texts, dtype=object))

print("Final Stats")
print("Chunks:", len(texts))
print("Embeddings:", len(embeddings))
print("Failed batches:", len(failed_batches))

print("Done")