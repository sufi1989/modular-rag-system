import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# =========================
# STEP 1: Load chunks
# =========================
with open("chunked_docs.pkl", "rb") as f:
    chunked_docs = pickle.load(f)

print("Chunks loaded:", len(chunked_docs))

# =========================
# STEP 2: FORCE CLEAN STRINGS
# =========================
print("Cleaning + validating text...")

texts = []

for i, doc in enumerate(chunked_docs):

    try:
        # handle LangChain Document
        if hasattr(doc, "page_content"):
            text = doc.page_content
        else:
            text = doc

        # FORCE string conversion
        if text is None:
            continue

        text = str(text).strip()

        # skip garbage
        if len(text) < 10:
            continue

        # remove weird null characters
        text = text.replace("\x00", " ")

        texts.append(text)

    except Exception:
        continue

print("Valid texts:", len(texts))

if len(texts) == 0:
    raise ValueError("No valid texts found")

# DEBUG CHECK (IMPORTANT)
print("Sample:", texts[0][:300])

# =========================
# STEP 3: Embeddings (CPU SAFE)
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# =========================
# STEP 4: BUILD FAISS
# =========================
print("Building FAISS index...")

db = FAISS.from_texts(texts, embeddings)

# =========================
# STEP 5: SAVE
# =========================
db.save_local("faiss_db")

print("FAISS saved!")

# =========================
# STEP 6: TEST RETRIEVER
# =========================
retriever = db.as_retriever(search_kwargs={"k": 5})

query = "What is overfitting in machine learning?"

results = retriever.get_relevant_documents(query)

for i, r in enumerate(results):
    print(f"\n--- RESULT {i} ---")
    print(r.page_content[:400])