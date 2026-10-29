import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
import chromadb

load_dotenv()
client = genai.Client(api_key=os.getenv("Gemini_API_Key"))

db = chromadb.PersistentClient(path="chroma_db")
collection = db.get_or_create_collection(
    name="data_collection",
    metadata={"hnsw:space": "cosine"}   # important for similarity
)

# ----------------------------
# 1. Load documents
# ----------------------------
def load_documents(data_dir="data"):
    documents = []
    data_path = Path(data_dir)

    for file_path in data_path.rglob("*.txt"):
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        documents.append({
            "id": str(file_path.relative_to(data_path)).replace("\\", "/"),
            "text": text,
            "metadata": {
                "source": str(file_path),
                "type": "journal" if "journals" in str(file_path) else "form"
            }
        })
    return documents


# ----------------------------
# 2. Embed texts
# ----------------------------
def embed_texts(texts: list[str], task_type="RETRIEVAL_DOCUMENT", model="gemini-embedding-001"):
    response = client.models.embed_content(
        model=model,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type
        )
    )
    return [emb.values for emb in response.embeddings]


# ----------------------------
# 3. Build / update the vector store (run once or when data changes)
# ----------------------------
def build_vector_store(data_dir="data", model="gemini-embedding-001"):
    docs = load_documents(data_dir)
    print(f"Loaded {len(docs)} documents")

    # Optional: clear old data so we don't get duplicates
    # collection.delete(where={})   # careful – deletes everything

    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]
    metadatas = [d["metadata"] for d in docs]

    vectors = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT", model=model)

    collection.add(
        documents=texts,          # ← list of strings (required)
        embeddings=vectors,       # ← list of vectors
        ids=ids,                  # ← unique ids
        metadatas=metadatas       # ← optional but very useful
    )

    print(f"Stored {len(docs)} documents in Chroma → chroma_db/")
    return docs


# ----------------------------
# 4. Correct retrieval from Chroma
# ----------------------------
def search(query: str, top_k=4, model="gemini-embedding-001"):
    # Embed the query with RETRIEVAL_QUERY
    query_vec = embed_texts(
        [query],
        task_type="RETRIEVAL_QUERY",
        model=model
    )[0]

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # results is a dict with lists
    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],   # lower = more similar (cosine)
        })
    return retrieved


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    # Run this only when you add/change files
    build_vector_store("data")

    # Actual retrieval
    results = search(
        "I feel unmotivated at my internship and need a plan for tomorrow",
        top_k=3
    )

    print("\n=== Top retrieved documents ===")
    for r in results:
        print(f"\nID: {r['id']} | distance: {r['distance']:.4f}")
        print(r["text"][:300] + "...")