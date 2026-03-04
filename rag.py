import os
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
import faiss


# Load Documents
def load_documents(folder_path):
    documents = []
    for file in os.listdir(folder_path):
        with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
            documents.append(f.read().strip())
    return documents


# Chunk Documents
def chunk_text(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))

    return chunks


# Create Embeddings
def create_embeddings(chunks, model):
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    return embeddings

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Retrieve Top K Chunks
def retrieve(query, chunks, index, model, top_k=3):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, top_k)

    results = [chunks[i] for i in indices[0]]

    return results


# Generate Answer using LLM
def generate_answer(query, context):

    prompt = f"""
You are an AI assistant answering questions using provided context.

If the answer is not contained in the context, say you do not know.

Context:
{context}

Question:
{query}

Answer clearly using only the provided context.
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response['message']['content']


# Main Pipeline
def main():

    print("Loading documents...")
    docs = load_documents("data")

    if not docs:
        print("No documents found in the data folder.")
        return

    print("Chunking documents...")
    chunks = []
    for doc in docs:
        chunks.extend(chunk_text(doc))

    if not chunks:
        print("No text chunks generated from documents.")
        return

    print("Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Creating embeddings...")
    embeddings = create_embeddings(chunks, embedding_model)
    
    print("Building FAISS index...")
    index = build_faiss_index(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors")

    while True:

        query = input("\nAsk a question (or type 'exit'): ").strip()

        if not query:
            continue

        if query.lower() == "exit":
            break

        retrieved_chunks = retrieve(query, chunks, index, embedding_model, top_k=3)

        context = "\n".join(retrieved_chunks)

        answer = generate_answer(query, context)

        print("\nRetrieved Context:")
        print(context)

        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()
