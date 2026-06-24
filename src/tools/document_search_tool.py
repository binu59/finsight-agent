"""
document_search_tool.py
------------------------
TOOL 3: Annual report PDF search via LlamaIndex.

Loads a PDF, chunks and embeds it with a local, free HuggingFace
embedding model, and builds a vector index
that's persisted to disk so it's only built once.

Design choice: this tool does RETRIEVAL ONLY — it returns the raw most
relevant chunks of text, with no LLM call inside the tool itself. The
agent's own LLM (Gemini/Claude) reads those chunks and reasons over
them in the ReAct loop, the same pattern as Tools 1 and 2.

REFACTORED: Index is now injectable via create_document_search_tool(),
rather than a global singleton. This allows Streamlit sessions to use
different uploaded PDFs without cache collisions.
"""

import os

from langchain_core.tools import Tool
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.config import ANNUAL_REPORT_PATH


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None

_PERSIST_DIR = "data/index_storage"
_default_index_cache = None  # Module-level cache for default index only


def get_default_index():
    """
    Return the default vector index from persisted storage (Apple 10-K),
    building it from ANNUAL_REPORT_PATH on first call and caching it
    thereafter. This is the fallback index used when no user-uploaded
    PDF is active in the session.
    """
    global _default_index_cache
    if _default_index_cache is not None:
        return _default_index_cache

    if os.path.exists(_PERSIST_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=_PERSIST_DIR)
        _default_index_cache = load_index_from_storage(storage_context)
    else:
        if not os.path.exists(ANNUAL_REPORT_PATH):
            raise FileNotFoundError(
                f"Annual report PDF not found at '{ANNUAL_REPORT_PATH}'. "
                f"Place a PDF there or update ANNUAL_REPORT_PATH in .env."
            )
        documents = SimpleDirectoryReader(input_files=[ANNUAL_REPORT_PATH]).load_data()
        _default_index_cache = VectorStoreIndex.from_documents(documents)
        _default_index_cache.storage_context.persist(persist_dir=_PERSIST_DIR)

    return _default_index_cache


def build_index_from_file(filepath: str) -> VectorStoreIndex:
    """
    Build a new VectorStoreIndex from a single PDF file (e.g., user-uploaded).
    Does NOT persist to disk — intended for session-scoped use.
    Raises FileNotFoundError if the file doesn't exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PDF file not found: {filepath}")

    documents = SimpleDirectoryReader(input_files=[filepath]).load_data()
    if not documents:
        raise ValueError(f"No documents found in {filepath}. Is it a valid PDF?")

    index = VectorStoreIndex.from_documents(documents)
    return index


def _search_with_index(index: VectorStoreIndex, query: str) -> str:
    """
    Helper: retrieve the most relevant chunks from a given index for a query,
    formatted as plain readable text with similarity scores attached.
    """
    retriever = index.as_retriever(similarity_top_k=3)
    nodes = retriever.retrieve(query)

    if not nodes:
        return f"No relevant information found in the document for '{query}'."

    formatted = []
    for i, node in enumerate(nodes, 1):
        text = node.get_content().strip().replace("\n", " ")
        score = node.score if node.score is not None else 0.0
        formatted.append(f"{i}. (relevance: {score:.2f}) {text[:500]}...")

    return "\n\n".join(formatted)


def create_document_search_tool(index: VectorStoreIndex | None = None) -> Tool:
    """
    Factory function: create and return a document_search Tool bound to a
    specific index. If index is None, uses get_default_index().

    This allows each Streamlit session (or test) to have its own index
    without global state collisions.
    """
    if index is None:
        index = get_default_index()

    def search_func(query: str) -> str:
        """Closure that captures the injected index."""
        try:
            return _search_with_index(index, query)
        except Exception as e:
            return f"Error searching document: {str(e)}"

    return Tool(
        name="document_search",
        func=search_func,
        description=(
            "Use this to find information FROM THE COMPANY'S OWN ANNUAL REPORT "
            "(10-K filing) — things like management discussion, risk factors, "
            "segment performance, or other specifics mentioned in the filing. "
            "Input should be a natural-language question, e.g. "
            "'what are the main risk factors' or 'how did the services segment perform'. "
            "Do NOT use this for live/current market data like market cap or "
            "stock price — use the financial_data tool for that instead. "
            "Do NOT use this for recent news or events — use web_search_news "
            "for that instead."
        ),
    )


def search_annual_report(query: str) -> str:
    """
    Search the default annual report (Apple 10-K).
    For backward compatibility and CLI/test usage.
    """
    index = get_default_index()
    return _search_with_index(index, query)


# Default tool instance for backward compatibility and non-Streamlit usage
document_search_tool = create_document_search_tool()


if __name__ == "__main__":
    
    test_query = "What are the main risk factors mentioned in the report?"
    print(f"Testing document search with query: '{test_query}'\n")
    print(search_annual_report(test_query))