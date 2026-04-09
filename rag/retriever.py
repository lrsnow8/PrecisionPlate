import chromadb
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_CHROMA_PATH = "rag/chroma_db"
_COLLECTION_NAME = "nutrition_knowledge"

# Lazy singletons — initialised on first call to query()
_model: SentenceTransformer | None = None
_collection = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=_CHROMA_PATH)
        _collection = client.get_or_create_collection(_COLLECTION_NAME)
    return _collection


def query(text: str, top_k: int = 5) -> str:
    """
    Embed *text*, run a cosine similarity search against the nutrition_knowledge
    ChromaDB collection, and return the top-k chunks concatenated as a single string.
    """
    model = _get_model()
    collection = _get_collection()

    embedding = model.encode(text).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents"],
    )

    documents = results.get("documents", [[]])[0]
    return "\n\n".join(documents)
