import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "data/pdfs"

docs = []

for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_PATH, file))
        docs.extend(loader.load())

print(f"Loaded {len(docs)} pages")

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=120
)

chunked_docs = splitter.split_documents(docs)

print(f"Created {len(chunked_docs)} chunks")