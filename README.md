# Retrieval-Augmented Generation (RAG) System

A lightweight **Retrieval-Augmented Generation (RAG)** pipeline that answers questions using external documents through **semantic search and a local LLM**.

This project demonstrates the core architecture behind modern **AI document QA systems**.

---

## Demo

**Question 1**

![Question and answer 1](demo_screenshots/qa1.png)

**Question 2**

![Question and answer 2](demo_screenshots/qa2.png)

---

## Key Features

- Document ingestion from text files
- Text chunking with overlap for better context retention
- Semantic embeddings using SentenceTransformers
- Vector similarity search with FAISS
- Context retrieval using Top-K nearest neighbors
- Answer generation using a local LLM (Ollama)

---

## Tech Stack

- **Python**
- **SentenceTransformers** – text embeddings
- **FAISS** – vector similarity search
- **NumPy** – vector operations
- **Ollama** – local LLM inference

---

## System Architecture
```
Documents
↓
Text Chunking
↓
Embedding Generation
↓
FAISS Vector Index
↓
User Query → Query Embedding
↓
Top-K Semantic Retrieval
↓
Context + Prompt
↓
LLM Answer
```
## How It Works

1. Load documents from a local data directory.
2. Split documents into overlapping text chunks.
3. Convert chunks into dense embeddings.
4. Store embeddings in a **FAISS vector index**.
5. Convert the user query into an embedding.
6. Retrieve the **top-K most similar chunks**.
7. Provide retrieved context to the LLM to generate an answer.