import os
import numpy as np

#  Better PDF loader (more stable than PyPDF)
from langchain_community.document_loaders import PyMuPDFLoader

#  Text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- PATH CONFIG ----------------
#  Source PDFs live here
PDF_SOURCE_PATH = r"D:\RAG_Project - testing\data\pdfs"

#  Output folder (new clean pipeline project)
OUTPUT_BASE_PATH = r"D:\RAG_Project - testing"

#  Output file
OUTPUT_FILE = os.path.join(OUTPUT_BASE_PATH, "clean_chunks.npy")

# ---------------- CREATE OUTPUT FOLDER IF NEEDED ----------------
os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)

print("Loading PDFs from:", PDF_SOURCE_PATH)

# ---------------- LOAD PDFs ----------------
docs = []

for file in os.listdir(PDF_SOURCE_PATH):
    if file.endswith(".pdf"):
        file_path = os.path.join(PDF_SOURCE_PATH, file)

        loader = PyMuPDFLoader(file_path)
        docs.extend(loader.load())

print(f" Total PDF pages loaded: {len(docs)}")

# ---------------- CLEAN FUNCTION ----------------
def clean_text(t):
    """
    Clean PDF extracted text:
    - remove null bytes
    - remove newlines
    - trim spaces
    - filter garbage text
    """

    if not isinstance(t, str):
        return None

    t = t.replace("\x00", " ")
    t = t.replace("\n", " ")
    t = t.strip()

    # remove useless text
    if len(t) < 20:
        return None

    return t

# ---------------- APPLY CLEANING ----------------
for d in docs:
    d.page_content = clean_text(d.page_content)

# remove empty pages
docs = [d for d in docs if d.page_content]

print(f"Clean pages after filtering: {len(docs)}")

# ---------------- CHUNKING ----------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=120
)

chunks = splitter.split_documents(docs)

# ---------------- EXTRACT TEXT ----------------
texts = []

for c in chunks:
    t = c.page_content
    if isinstance(t, str) and len(t) > 20:
        texts.append(t)

print(f"Total final chunks: {len(texts)}")

# ---------------- SAVE OUTPUT ----------------
np.save(OUTPUT_FILE, np.array(texts, dtype=object))

print(f"Saved clean chunks to: {OUTPUT_FILE}")