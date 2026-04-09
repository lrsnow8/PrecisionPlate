from langchain_core.tools import tool

from rag.retriever import query as _rag_query


@tool
def get_nutrition_info(query: str) -> str:
    """
    Look up nutrition information from the knowledge base using semantic search.
    Use this for questions about dietary guidelines, nutrients, food groups,
    recommended intakes, and general nutrition science.

    Args:
        query: A free-text nutrition question or topic.

    Returns:
        Relevant excerpts from the nutrition knowledge base as a single string.
    """
    try:
        result = _rag_query(query, top_k=5)
        if not result.strip():
            return "No relevant nutrition information found for that query."
        return result
    except Exception as e:
        return f"Error retrieving nutrition information: {e}"
