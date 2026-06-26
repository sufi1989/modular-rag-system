# Modular RAG System (GPU + FAISS)

## Overview

This project is a modular Retrieval-Augmented Generation (RAG) system built from scratch to explore how production-style document question-answering systems work under the hood.

The system converts raw documents into searchable vector representations using GPU-accelerated embeddings, stores them in a FAISS index, retrieves relevant context based on semantic similarity, and generates answers using a local large language model.

The goal of this project is not just to build a working RAG pipeline, but to iteratively improve it toward a production-grade retrieval system with evaluation, reranking, and observability.

---

## Pipeline Architecture

The system follows a standard RAG flow:

Documents → Chunking → Embeddings → FAISS Index → Retrieval → LLM Generation



Each component is modular and can be independently improved without affecting the full pipeline.

---

## Key Components

### 1. Document Chunking
Raw documents are split into fixed-size chunks to improve retrieval granularity.

- Chunk size: 700 tokens  
- Overlap: 100 tokens  

This ensures semantic continuity while maintaining retrieval efficiency.

---

### 2. Embedding Generation (GPU-Accelerated)

Text chunks are converted into dense vector representations using:

- Model: `BAAI/bge-large-en-v1.5`
- Embedding dimension: 1024
- Execution: CUDA-enabled PyTorch pipeline

This enables high-quality semantic understanding of text.

---

### 3. Vector Database (FAISS)

All embeddings are stored in a FAISS index for fast similarity search.

Features:
- Cosine similarity-based retrieval
- Efficient top-k nearest neighbor search
- Scalable vector indexing

---

### 4. Retriever

Given a user query:

- The query is embedded using the same model
- FAISS retrieves top-k most similar chunks
- Results are ranked by similarity score

---

### 5. LLM Answer Generation

A local instruction-tuned model is used for response generation:

- Model: `Zephyr-7B-beta`
- Runs locally using Hugging Face Transformers
- Uses retrieved context as grounding input

The model is prompted to generate answers based only on retrieved information to reduce hallucination.

---

## Features

- GPU-accelerated embedding pipeline
- FAISS-based semantic search
- Fully modular architecture (each component independent)
- Local LLM inference (no external API dependency)
- Clean separation of retrieval and generation logic
- End-to-end working RAG pipeline

---

## Tech Stack

- Python
- PyTorch (CUDA)
- SentenceTransformers
- FAISS
- Hugging Face Transformers
- Zephyr 7B (LLM)

---

## Current Status

This project is under active development.

### Completed
- Chunking pipeline implemented
- GPU-based embedding generation
- FAISS vector index integration
- Functional retriever system
- End-to-end RAG pipeline working locally

---

## Current Limitations

While the system is fully functional, it is not yet production-optimized:

- No reranking layer (FAISS top-k only retrieval)
- No formal evaluation framework (retrieval or generation metrics)
- Limited observability for debugging retrieval behavior
- Prompt grounding still being optimized to reduce hallucinations

---

## Active Development

This project is actively evolving into a production-style RAG system.

Current work focuses on:

- Improving retrieval quality using reranking models (e.g., bge-reranker)
- Building a structured evaluation framework (Recall@k, LLM-as-a-judge)
- Optimizing prompt grounding to reduce hallucinations
- Experimenting with hybrid search (dense + sparse retrieval)
- Adding logging and tracing for retrieval debugging and system analysis

---

## Planned Improvements

- Add cross-encoder reranking for improved retrieval precision
- Implement structured evaluation metrics (Recall@k, groundedness, accuracy)
- Introduce hybrid search (FAISS + BM25)
- Add observability and logging for query-level debugging
- Explore API deployment or lightweight UI interface

---

## Goal

The goal of this project is to build a production-grade, modular RAG system that goes beyond a basic demo and demonstrates:

- Strong retrieval quality
- Measurable system performance
- Reduced hallucination through grounding
- Extensible architecture for real-world applications
