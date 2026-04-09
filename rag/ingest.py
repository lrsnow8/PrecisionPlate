"""
One-time (idempotent) ingestion script.

Run from the project root:
    python rag/ingest.py

Loads every .txt file from rag/docs/, splits into chunks, embeds with
sentence-transformers/all-MiniLM-L6-v2, and upserts into the
"nutrition_knowledge" ChromaDB collection.

Safe to re-run: uses collection.upsert() with stable chunk IDs derived
from the filename and chunk index, so no duplicates are created.
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = "rag/docs"
CHROMA_PATH = "rag/chroma_db"
COLLECTION_NAME = "nutrition_knowledge"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def ingest() -> None:
    # Load model
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    txt_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".txt")]
    if not txt_files:
        print(f"No .txt files found in {DOCS_DIR}. Nothing to ingest.")
        return

    for filename in sorted(txt_files):
        filepath = os.path.join(DOCS_DIR, filename)
        print(f"Processing: {filepath}")

        with open(filepath, "r", encoding="utf-8") as fh:
            text = fh.read()

        chunks = splitter.split_text(text)
        print(f"  → {len(chunks)} chunks")

        # Stable IDs: "<filename>_chunk_<index>" — guarantees upsert idempotency
        ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
        embeddings = model.encode(chunks).tolist()

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
        )
        print(f"  → Upserted {len(chunks)} chunks into '{COLLECTION_NAME}'")

    total = collection.count()
    print(f"Ingestion complete. Total documents in collection: {total}")


if __name__ == "__main__":
    ingest()
