import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
import chromadb

load_dotenv()
client = genai.Client(api_key=os.getenv("Gemini_API_Key"))

CATEGORIES = [
    "job_career",
    "personal_life",
    "mental_health",
    "relationships",
    "physical_health",
    "hobbies_interests",
    "others",
]

SYSTEM_INSTRUCTION = """
You are a text chunker for a personal AI assistant (RAG system).

Your job:
- Split the given text into small, self-contained chunks.
- Keep the original meaning of each idea intact.
- Assign exactly one category to each chunk.

Allowed categories (use these exact names):
1. job_career
2. personal_life
3. mental_health
4. relationships
5. physical_health
6. hobbies_interests
7. others

Rules:
- Each chunk should be 1–4 sentences and focus on one main idea.
- Do not invent facts. Only use what is in the text.
- If a sentence mixes topics, split it into separate chunks when possible.
- Prefer more precise categories over "others".
- Keep the user's original voice and details (names, feelings, plans).
- Return ONLY valid JSON. No markdown, no explanation.

Output format (strict JSON array):
[
  {
    "category": "job_career",
    "text": "chunk text here"
  },
  {
    "category": "mental_health",
    "text": "another chunk"
  }
]

Examples:

Input:
"Internship was draining again. No feedback on my work. In the evening I coded my LLM project and felt better. Missed my ex a little while working."

Output:
[
  {
    "category": "job_career",
    "text": "Internship was draining again. No feedback on my work."
  },
  {
    "category": "hobbies_interests",
    "text": "In the evening I coded my LLM project and felt better."
  },
  {
    "category": "relationships",
    "text": "Missed my ex a little while working."
  }
]

Input:
"Mom called and asked if I was eating properly. I slept only 6 hours and felt anxious about graduation."

Output:
[
  {
    "category": "relationships",
    "text": "Mom called and asked if I was eating properly."
  },
  {
    "category": "physical_health",
    "text": "I slept only 6 hours."
  },
  {
    "category": "mental_health",
    "text": "I felt anxious about graduation."
  }
]
"""

db = chromadb.PersistentClient(path="chroma_db")
collection = db.get_or_create_collection(
    name="data_collection",
    metadata={"hnsw:space": "cosine"}
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


def extract_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"Could not parse JSON:\n{text[:400]}")


def chuck_text(files=None):
    if files is None:
        files = load_documents()

    chat = client.chats.create(
        model="gemini-3.6-flash",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
        )
    )

    all_chunks = []

    for file in files:
        text = file["text"]
        response = chat.send_message(
            f"Chunk the following text into categorized pieces:\n\n{text}"
        )

        raw_chunks = extract_json(response.text)
        chunks = add_chuck_metadata(raw_chunks, file)
        all_chunks.extend(chunks)

        print(f"Chunked {file['id']} → {len(chunks)} chunks")

    return all_chunks


def add_chuck_metadata(chunks, file):
    cleaned = []

    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue

        category = str(chunk.get("category", "others")).strip().lower()
        text = str(chunk.get("text", "")).strip()

        if not text:
            continue

        if category not in CATEGORIES:
            category = "others"

        cleaned.append({
            "id": f"{file['id']}::chunk_{i+1}",
            "text": text,
            "metadata": {
                "source": file["metadata"]["source"],
                "type": file["metadata"]["type"],
                "category": category,
                "source_file": file["id"],
            }
        })

    return cleaned


# ----------------------------
# 2. Embed texts
# ----------------------------
def embed_texts(chunks: list[str], task_type="RETRIEVAL_DOCUMENT", model="gemini-embedding-001"):
    response = client.models.embed_content(
        model=model,
        contents=chunks,
        config=types.EmbedContentConfig(
            task_type=task_type
        )
    )
    return [emb.values for emb in response.embeddings]


# ----------------------------
# 3. Build / update the vector store
# ----------------------------
def build_vector_store(data_dir="data", model="gemini-embedding-001"):
    docs = load_documents(data_dir)
    print(f"Loaded {len(docs)} documents")

    # Chunk first, then embed chunks
    chunks = chuck_text(docs)
    print(f"Total chunks: {len(chunks)}")

    # Optional: clear old data so you don't get duplicates
    # collection.delete(where={})

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    vectors = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT", model=model)

    collection.add(
        documents=texts,
        embeddings=vectors,
        ids=ids,
        metadatas=metadatas
    )

    print(f"Stored {len(chunks)} chunks in Chroma → chroma_db/")
    return chunks


# ----------------------------
# 4. Retrieval from Chroma
# ----------------------------
def search(query: str, top_k=4, model="gemini-embedding-001"):
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

    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return retrieved


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    # Build store with chunking → embedding → chroma
    chunks = build_vector_store("data")

    # Test retrieval
    results = search(
        "I feel unmotivated at my internship and need a plan for tomorrow",
        top_k=3
    )

    print("\n=== Top retrieved chunks ===")
    for r in results:
        print(f"\nID: {r['id']} | category: {r['metadata'].get('category')} | distance: {r['distance']:.4f}")
        print(r["text"][:250] + "...")