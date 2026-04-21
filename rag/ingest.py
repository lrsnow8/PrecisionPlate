"""
One-time (idempotent) ingestion script.

Run from the project root:
    python rag/ingest.py

Loads every .txt and .json file from rag/docs/, splits into chunks, embeds with
sentence-transformers/all-MiniLM-L6-v2, and upserts into the
"nutrition_knowledge" ChromaDB collection.

JSON files must follow the USDA FoodData Central format (top-level
"FoundationFoods" array). Each food item is converted to a plain-text string
before chunking so retrieval quality matches the .txt pipeline.

Safe to re-run: uses collection.upsert() with stable chunk IDs derived
from the filename and chunk index, so no duplicates are created.
"""

import json
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


def _load_txt(filepath: str) -> list[str]:
    with open(filepath, "r", encoding="utf-8") as fh:
        return [fh.read()]


def _load_json(filepath: str) -> list[str]:
    """Convert USDA FoodData Central foundation food JSON to per-food text strings."""
    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    texts = []
    for food in data.get("FoundationFoods", []):
        name = food.get("description", "Unknown food")
        parts = []
        for n in food.get("foodNutrients", []):
            nutrient_name = n.get("nutrient", {}).get("name", "")
            unit = n.get("nutrient", {}).get("unitName", "")
            amount = n.get("amount")
            if nutrient_name and amount is not None:
                parts.append(f"{nutrient_name}: {amount} {unit}")
        nutrients_str = "; ".join(parts) if parts else "no nutrient data"
        texts.append(f"Food: {name}. Nutrients per 100g: {nutrients_str}.")
    return texts


def ingest() -> None:
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    doc_files = [
        f for f in os.listdir(DOCS_DIR)
        if f.endswith(".txt") or f.endswith(".json")
    ]
    if not doc_files:
        print(f"No .txt or .json files found in {DOCS_DIR}. Nothing to ingest.")
        return

    for filename in sorted(doc_files):
        filepath = os.path.join(DOCS_DIR, filename)
        print(f"Processing: {filepath}")

        if filename.endswith(".json"):
            raw_texts = _load_json(filepath)
        else:
            raw_texts = _load_txt(filepath)

        chunks = []
        for text in raw_texts:
            chunks.extend(splitter.split_text(text))
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
