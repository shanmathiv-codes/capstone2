"""
LlamaIndex Document RAG for Telecom AI Operations Center.

Responsibilities:
1. Load policy TXT files from data/documents/
2. Create embeddings using HuggingFace all-MiniLM-L6-v2
3. Build a LlamaIndex VectorStoreIndex
4. Persist the index to data/vector_index/
5. Load the persisted index on later runs
6. Expose a single function: answer_policy_question(question)

This module intentionally uses LlamaIndex's query engine directly.
It does NOT use LlamaIndex agents, workflows, or ReAct patterns.
"""

from pathlib import Path
import os
from dotenv import load_dotenv
import httpx
import traceback
from huggingface_hub import set_client_factory

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()


# ---------------------------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------------------------

# document_rag.py is inside:
# capstone-project/llamaindex_rag/
#
# Therefore:
# Path(__file__).resolve().parent.parent
# points to:
# capstone-project/

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
VECTOR_INDEX_DIR = BASE_DIR / "data" / "vector_index"


# ---------------------------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------------------------

# The project specification recommends:
# HuggingFace all-MiniLM-L6-v2
#
# This model is used locally to convert document chunks and user questions
# into embeddings for semantic retrieval.

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def configure_models() -> None:
    """
    Configure the embedding model and LLM used by LlamaIndex.

    The embedding model runs locally.

    The LLM is used only after relevant document chunks have been retrieved,
    so that those chunks can be synthesized into a natural-language answer.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY was not found in the environment.")

    # Hugging Face Hub and OpenRouter use separate HTTP clients.
    set_client_factory(lambda: httpx.Client(verify=False))
    http_client = httpx.Client(verify=False)

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME
    )

    Settings.llm = OpenAI(
        model=os.getenv("OPENROUTER_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        http_client=http_client
    )


# ---------------------------------------------------------------------------
# DIRECTORY VALIDATION
# ---------------------------------------------------------------------------

def validate_document_directory() -> None:
    """
    Verify that the policy document directory exists and contains TXT files.
    """

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Policy document directory does not exist: {DOCUMENTS_DIR}"
        )

    txt_files = list(DOCUMENTS_DIR.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(
            f"No TXT policy documents were found in: {DOCUMENTS_DIR}"
        )


# ---------------------------------------------------------------------------
# INDEX CREATION
# ---------------------------------------------------------------------------

def build_vector_index() -> VectorStoreIndex:
    """
    Load all policy TXT files, create embeddings, build the VectorStoreIndex,
    and persist it to data/vector_index/.

    Returns:
        VectorStoreIndex: The newly created index.
    """

    validate_document_directory()

    print("Loading policy documents...")
    print(f"Source directory: {DOCUMENTS_DIR}")

    documents = SimpleDirectoryReader(
        input_dir=str(DOCUMENTS_DIR),
        required_exts=[".txt"],
        recursive=False,
    ).load_data()

    if not documents:
        raise ValueError(
            "No policy documents were loaded from data/documents/."
        )

    print(f"Loaded {len(documents)} document(s).")

    print("Building vector index...")

    index = VectorStoreIndex.from_documents(
        documents
    )

    # Create the output directory if it does not exist.
    VECTOR_INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Persist the index so that embeddings do not need to be rebuilt
    # every time the application starts.
    index.storage_context.persist(
        persist_dir=str(VECTOR_INDEX_DIR)
    )

    print(f"Vector index persisted to: {VECTOR_INDEX_DIR}")

    return index


# ---------------------------------------------------------------------------
# INDEX LOADING
# ---------------------------------------------------------------------------

def load_vector_index() -> VectorStoreIndex:
    """
    Load the existing persisted vector index.

    Returns:
        VectorStoreIndex: Previously persisted LlamaIndex index.
    """

    if not VECTOR_INDEX_DIR.exists():
        raise FileNotFoundError(
            f"Vector index does not exist: {VECTOR_INDEX_DIR}"
        )

    print("Loading existing vector index...")
    print(f"Index directory: {VECTOR_INDEX_DIR}")

    storage_context = StorageContext.from_defaults(
        persist_dir=str(VECTOR_INDEX_DIR)
    )

    index = load_index_from_storage(
        storage_context
    )

    print("Vector index loaded successfully.")

    return index


# ---------------------------------------------------------------------------
# GET INDEX
# ---------------------------------------------------------------------------

def get_vector_index() -> VectorStoreIndex:
    """
    Return an existing vector index if available.

    If the index has not been created yet, build it automatically.

    Returns:
        VectorStoreIndex: Ready-to-query vector index.
    """

    configure_models()

    # A persisted index normally contains files such as:
    # docstore.json
    # index_store.json
    # graph_store.json
    # default__vector_store.json
    #
    # Checking whether the directory contains files is more reliable
    # than checking only whether the directory exists.

    if VECTOR_INDEX_DIR.exists() and any(VECTOR_INDEX_DIR.iterdir()):
        return load_vector_index()

    return build_vector_index()


# ---------------------------------------------------------------------------
# QUERY ENGINE
# ---------------------------------------------------------------------------

def get_query_engine():
    """
    Create a LlamaIndex query engine over the policy vector index.

    The specification specifically requires using as_query_engine rather
    than LlamaIndex agents, workflows, or ReAct patterns.
    """

    index = get_vector_index()

    query_engine = index.as_query_engine(
        similarity_top_k=3
    )

    return query_engine


# ---------------------------------------------------------------------------
# PUBLIC RAG FUNCTION
# ---------------------------------------------------------------------------

def answer_policy_question(question: str) -> str:
    """
    Answer a natural-language question using the policy documents.

    Args:
        question: User's policy/FAQ question.

    Returns:
        A synthesized natural-language answer.

    Example:
        answer_policy_question(
            "What is the roaming policy for Western Europe?"
        )
    """

    if not isinstance(question, str):
        raise TypeError("question must be a string.")

    question = question.strip()

    if not question:
        raise ValueError("question cannot be empty.")

    query_engine = get_query_engine()

    response = query_engine.query(question)

    return str(response)


# ---------------------------------------------------------------------------
# COMMAND-LINE TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("Telecom AI Operations Center - Policy RAG")
    print("=" * 60)

    question = input(
        "\nEnter a policy question: "
    ).strip()

    try:
        answer = answer_policy_question(question)

        print("\nAnswer:")
        print("-" * 60)
        print(answer)
        print("-" * 60)

    except Exception:
        print("\nRAG execution failed:")
        traceback.print_exc()

