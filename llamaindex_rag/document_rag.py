"""
LlamaIndex Document RAG for Prodapt AI Operations Center.

Responsibilities:
1. Load policy TXT files from data/documents/
2. Generate embeddings through the Hugging Face Inference API
3. Build a LlamaIndex VectorStoreIndex
4. Persist the index to data/vector_index/
5. Load the persisted index on later runs
6. Use OpenRouter's dynamic free-model router as the LLM
7. Expose one public function:
       answer_policy_question(question)

Important:
- LlamaIndex is used only for document retrieval and query-engine functionality.
- No LlamaIndex agents, workflows, or ReAct are used.
- Operational/business data is not stored here; this module only works with
  the supplied policy/FAQ TXT documents.
"""

from pathlib import Path
import logging
import os
import sys

# os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

from dotenv import load_dotenv
import httpx
from huggingface_hub import (
    set_client_factory,
)

# TEMPORARY DEVELOPMENT WORKAROUND:
# Disable SSL certificate verification for Hugging Face requests.
set_client_factory(lambda: httpx.Client(verify=False))

# TEMPORARY DEVELOPMENT WORKAROUND:
# Disable SSL certificate verification for OpenRouter requests.
openrouter_http_client = httpx.Client(
    verify=False,
    timeout=120.0,
)

openrouter_async_http_client = httpx.AsyncClient(
    verify=False,
    timeout=120.0,
)

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)

from llama_index.embeddings.huggingface_api import (
    HuggingFaceInferenceAPIEmbedding,
)
from llama_index.utils.huggingface import format_query, format_text

from llama_index.llms.openrouter import OpenRouter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

# Project root:
# capstone-project/
#     data/
#     llamaindex_rag/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
VECTOR_INDEX_DIR = PROJECT_ROOT / "data" / "vector_index"

# Hugging Face embedding model.
# This is the API version of the model rather than downloading it locally.
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# OpenRouter dynamically selects an available free model.
OPENROUTER_MODEL = "openrouter/free"


class SynchronousHuggingFaceInferenceAPIEmbedding(
    HuggingFaceInferenceAPIEmbedding
):
    """Use the sync Hub client so sync LlamaIndex calls do not cross event loops."""

    def _embed_sync(self, text: str) -> list[float]:
        embedding = self._sync_client.feature_extraction(text)
        if len(embedding.shape) == 1:
            return embedding.tolist()

        embedding = embedding.squeeze(axis=0)
        if len(embedding.shape) == 1:
            return embedding.tolist()

        return self.pooling(embedding).tolist()

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed_sync(
            format_query(query, self.model_name, self.query_instruction)
        )

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed_sync(
            format_text(text, self.model_name, self.text_instruction)
        )

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [self._get_text_embedding(text) for text in texts]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------

def _validate_environment() -> None:
    """
    Validate that the required API keys are available.
    """

    huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    if not huggingface_api_key:
        raise RuntimeError(
            "HUGGINGFACE_API_KEY is not set in the .env file."
        )

    if not openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set in the .env file."
        )


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

def _configure_models() -> None:
    """
    Configure the embedding model and LLM used by LlamaIndex.

    Embeddings:
        Hugging Face Inference API

    LLM:
        OpenRouter dynamic free-model router
    """

    _validate_environment()

    huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    # ---------------------------------------------------------------
    # Hugging Face embedding model
    # ---------------------------------------------------------------
    #
    # The model runs remotely through Hugging Face's inference service.
    # No sentence-transformers model weights are downloaded locally.
    #
    Settings.embed_model = SynchronousHuggingFaceInferenceAPIEmbedding(
        model_name=HF_EMBEDDING_MODEL,
        token=huggingface_api_key,
    )

    # ---------------------------------------------------------------
    # OpenRouter LLM
    # ---------------------------------------------------------------
    #
    # openrouter/free dynamically chooses an available free model.
    #
    Settings.llm = OpenRouter(
        api_key=openrouter_api_key,
        model=OPENROUTER_MODEL,
        temperature=0.2,
        max_tokens=512,
        http_client=openrouter_http_client,
        async_http_client=openrouter_async_http_client,
    )

    

    logger.info(
        "Embedding model configured: %s",
        HF_EMBEDDING_MODEL,
    )

    logger.info(
        "LLM configured through OpenRouter: %s",
        OPENROUTER_MODEL,
    )


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def _load_documents():
    """
    Load all documents from data/documents/.

    SimpleDirectoryReader is intentionally used because the project
    specification explicitly requires it.
    """

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Document directory not found: {DOCUMENTS_DIR}"
        )

    txt_files = list(DOCUMENTS_DIR.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(
            f"No TXT files found in: {DOCUMENTS_DIR}"
        )

    logger.info(
        "Found %d policy documents in %s",
        len(txt_files),
        DOCUMENTS_DIR,
    )

    reader = SimpleDirectoryReader(
        input_dir=str(DOCUMENTS_DIR),
        required_exts=[".txt"],
        recursive=False,
    )

    documents = reader.load_data()

    if not documents:
        raise RuntimeError(
            f"SimpleDirectoryReader loaded no documents from {DOCUMENTS_DIR}"
        )

    logger.info(
        "Loaded %d document objects",
        len(documents),
    )

    return documents


# ---------------------------------------------------------------------------
# Vector index creation / loading
# ---------------------------------------------------------------------------

def _build_index():
    """
    Build a new VectorStoreIndex from the policy documents.

    The index is persisted to data/vector_index/.
    """

    logger.info("Building new VectorStoreIndex...")

    documents = _load_documents()

    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
    )

    VECTOR_INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    index.storage_context.persist(
        persist_dir=str(VECTOR_INDEX_DIR)
    )

    logger.info(
        "Vector index persisted to %s",
        VECTOR_INDEX_DIR,
    )

    return index


def _load_existing_index():
    """
    Load an already-persisted VectorStoreIndex.
    """

    logger.info(
        "Loading existing vector index from %s",
        VECTOR_INDEX_DIR,
    )

    storage_context = StorageContext.from_defaults(
        persist_dir=str(VECTOR_INDEX_DIR)
    )

    index = load_index_from_storage(
        storage_context
    )

    logger.info("Existing vector index loaded successfully.")

    return index


def _get_index():
    """
    Return the existing persisted index if available.

    Otherwise build and persist a new index.
    """

    if VECTOR_INDEX_DIR.exists():
        # Check whether the directory actually contains files.
        index_files = list(VECTOR_INDEX_DIR.iterdir())

        if index_files:
            try:
                return _load_existing_index()

            except Exception as exc:
                logger.warning(
                    "Could not load existing vector index: %s",
                    exc,
                )

                logger.warning(
                    "Rebuilding vector index from source documents..."
                )

    return _build_index()


# ---------------------------------------------------------------------------
# Query engine
# ---------------------------------------------------------------------------

def _create_query_engine(index):
    """
    Create the LlamaIndex query engine.

    This intentionally uses as_query_engine().
    No LlamaIndex agents, workflows, or ReAct patterns are used.
    """

    query_engine = index.as_query_engine(
        similarity_top_k=3,
        response_mode="compact",
    )

    return query_engine


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def answer_policy_question(question: str) -> str:
    """
    Answer a natural-language question using the policy/FAQ documents.

    Parameters
    ----------
    question : str
        Natural-language policy or FAQ question.

    Returns
    -------
    str
        LLM-generated answer grounded in the retrieved policy documents.
    """

    if not isinstance(question, str):
        raise TypeError(
            "question must be a string."
        )

    question = question.strip()

    if not question:
        raise ValueError(
            "question cannot be empty."
        )

    # Configure embedding model and LLM.
    _configure_models()

    # Load existing index or build it once.
    index = _get_index()

    # Create the LlamaIndex query engine.
    query_engine = _create_query_engine(index)

    logger.info(
        "Running policy RAG query: %s",
        question,
    )

    try:
        response = query_engine.query(question)

    except Exception as exc:
        logger.exception(
            "Policy RAG query failed."
        )

        raise RuntimeError(
            f"Failed to answer policy question: {exc}"
        ) from exc

    answer = str(response).strip()

    if not answer:
        return (
            "I could not find enough information in the provided "
            "policy documents to answer the question."
        )

    return answer


# ---------------------------------------------------------------------------
# Direct execution test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # test_question = (
    #     "What is the roaming policy for Western Europe?"
    # )

    question=input("Enter a question: ")

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(answer_policy_question(question))