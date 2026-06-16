# Modular RAG System (GPU + FAISS)

## Overview

This project is a modular Retrieval-Augmented Generation (RAG) system built from scratch to explore scalable document question-answering using local embeddings, vector search, and large language models.

The system transforms raw documents into searchable knowledge using embedding models and FAISS vector indexing, then uses a local LLM to generate context-aware answers.



## Pipeline Architecture

The system follows a standard RAG flow:


Documents → Chunking → Embeddings → FAISS Index → Retrieval → LLM Generation


Each stage is modular and can be improved independently without breaking the overall pipeline.



## Key Components

### 1. Document Chunking

Raw text documents are split into smaller, semantically meaningful chunks for better retrieval accuracy.



### 2. Embedding Generation (GPU-accelerated)

Chunks are converted into dense vector representations using:

- BAAI/bge-large-en-v1.5  
- CUDA-enabled PyTorch pipeline  

This allows high-quality semantic encoding of text.



### 3. Vector Database (FAISS)

Embeddings are stored in a FAISS index for fast similarity search.

- Supports cosine similarity search  
- Optimized for large-scale retrieval  
- Enables fast top-k document lookup  



### 4. Retriever

Given a user query:

- Query is embedded  
- FAISS retrieves top-k most relevant chunks  
- Returns ranked context passages  



### 5. LLM Answer Generation

A local instruction-tuned model (Zephyr 7B) is used to generate answers:

- Uses retrieved context as grounding  
- Reduces hallucination by constraining responses  
- Produces structured natural language answers  



## Features

- GPU-accelerated embedding pipeline  
- FAISS-based semantic search  
- Modular architecture (each component independent)  
- Resume-safe embedding pipeline with checkpoints  
- Local LLM inference (no API dependency)  
- Clean separation of retriever and generator  



## Tech Stack

- Python  
- PyTorch (CUDA)  
- SentenceTransformers  
- FAISS  
- HuggingFace Transformers  
- Zephyr 7B (LLM)  



## Current Status

- Chunking pipeline: Complete  
- Embedding pipeline: GPU-enabled and checkpointed  
- Vector database: FAISS integrated  
- Retriever: Functional semantic search  
- RAG pipeline: End-to-end working locally  



## Goal

To build a production-style, modular RAG system that can be extended with:

- Reranking models  
- Hybrid search (BM25 + vectors)  
- LangSmith tracing  
- Multi-document reasoning  
- API deployment  
```
