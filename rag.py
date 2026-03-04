import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ollama


# Load Documents
def load_documents(folder_path):
    documents = []
    for file in os.listdir(folder_path):
        with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
            documents.append(f.read())
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
    embeddings = model.encode(chunks)
    return embeddings


# Retrieve Top K Chunks
def retrieve(query, chunks, embeddings, model, top_k=3):
    query_embedding = model.encode([query])

    scores = cosine_similarity(query_embedding, embeddings)[0]

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = [chunks[i] for i in top_indices]

    return results


# Generate Answer using LLM
def generate_answer(query, context):

    prompt = f"""
You are a helpful assistant.

Use the following context to answer the question.

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

    print("Chunking documents...")
    chunks = []
    for doc in docs:
        chunks.extend(chunk_text(doc))

    print("Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Creating embeddings...")
    embeddings = create_embeddings(chunks, embedding_model)

    while True:

        query = input("\nAsk a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        retrieved_chunks = retrieve(query, chunks, embeddings, embedding_model)

        context = "\n".join(retrieved_chunks)

        answer = generate_answer(query, context)

        print("\nRetrieved Context:")
        print(context)

        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()
