from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pickle

# LOAD PDFs (change folder name if needed)
loader = PyPDFDirectoryLoader("data/pdfs")
docs = loader.load()

print("Pages loaded:", len(docs))

# SPLITTER
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=120
)

chunked_docs = splitter.split_documents(docs)

print("Chunks created:", len(chunked_docs))

# SAVE CHUNKS
with open("chunked_docs.pkl", "wb") as f:
    pickle.dump(chunked_docs, f)

print("Saved successfully!")