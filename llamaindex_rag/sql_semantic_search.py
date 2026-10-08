"""
LlamaIndex Semantic SQL for Prodapt AI Operations Center.

Responsibilities:
1. Connect to data/telecom_ops.db through LlamaIndex SQLDatabase.
2. Create SQLTableNodeMapping for the database tables.
3. Define SQLTableSchema objects with semantic context descriptions.
4. Build an ObjectIndex over the table schemas.
5. Retrieve the most relevant table(s) for a natural-language question.
6. Use SQLTableRetrieverQueryEngine to generate and execute SQL.
7. Return a natural-language answer based on the SQL result.

Important:
- Operational/business data comes only from SQLite.
- No operational data is stored in Python dictionaries.
- ObjectIndex is used only for semantic table selection.
- SQLTableRetrieverQueryEngine handles text-to-SQL and response synthesis.
- This module does not use LlamaIndex agents, workflows, or ReAct.
"""

from pathlib import Path
import logging
import os
import sys

from dotenv import load_dotenv
from sqlalchemy import create_engine

from llama_index.core import (
    
    Settings,
    SQLDatabase,
    VectorStoreIndex,
)

from llama_index.core.objects import (
    ObjectIndex,
    SQLTableNodeMapping,
    SQLTableSchema,
)

from llama_index.core.query_engine import (
    SQLTableRetrieverQueryEngine,
)

from llama_index.llms.openrouter import OpenRouter

from llama_index.embeddings.huggingface_api import (
    HuggingFaceInferenceAPIEmbedding,
)

from llama_index.utils.huggingface import (
    format_query,
    format_text,
)

from llama_index.core.prompts import PromptTemplate

import httpx
from huggingface_hub import set_client_factory


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATABASE_PATH = PROJECT_ROOT / "data" / "telecom_ops.db"


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

OPENROUTER_MODEL = "openrouter/free"


# ---------------------------------------------------------------------------
# Temporary SSL workaround
# ---------------------------------------------------------------------------
#
# Your environment previously produced:
#
#     CERTIFICATE_VERIFY_FAILED:
#     self-signed certificate in certificate chain
#
# Hugging Face and OpenRouter therefore use temporary HTTP clients
# with certificate verification disabled.
#
# This is for development only.
# ---------------------------------------------------------------------------

set_client_factory(
    lambda: httpx.Client(
        verify=False,
        timeout=120.0,
    )
)


openrouter_http_client = httpx.Client(
    verify=False,
    timeout=120.0,
)


openrouter_async_http_client = httpx.AsyncClient(
    verify=False,
    timeout=120.0,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synchronous Hugging Face embedding implementation
# ---------------------------------------------------------------------------

class SynchronousHuggingFaceInferenceAPIEmbedding(
    HuggingFaceInferenceAPIEmbedding
):
    """
    Use the synchronous Hugging Face client for LlamaIndex
    synchronous query execution.
    """

    def _embed_sync(self, text: str) -> list[float]:
        embedding = self._sync_client.feature_extraction(text)

        if len(embedding.shape) == 1:
            return embedding.tolist()

        embedding = embedding.squeeze(axis=0)

        if len(embedding.shape) == 1:
            return embedding.tolist()

        return self.pooling(embedding).tolist()

    def _get_query_embedding(
        self,
        query: str,
    ) -> list[float]:

        return self._embed_sync(
            format_query(
                query,
                self.model_name,
                self.query_instruction,
            )
        )

    def _get_text_embedding(
        self,
        text: str,
    ) -> list[float]:

        return self._embed_sync(
            format_text(
                text,
                self.model_name,
                self.text_instruction,
            )
        )

    def _get_text_embeddings(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        return [
            self._get_text_embedding(text)
            for text in texts
        ]


# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------

def _validate_environment() -> None:
    """
    Validate API keys and database availability.
    """

    huggingface_api_key = os.getenv(
        "HUGGINGFACE_API_KEY"
    )

    openrouter_api_key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    if not huggingface_api_key:
        raise RuntimeError(
            "HUGGINGFACE_API_KEY is not set in the .env file."
        )

    if not openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set in the .env file."
        )

    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"SQLite database not found: {DATABASE_PATH}"
        )


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

def _configure_models() -> None:
    """
    Configure the embedding model and LLM.

    Embeddings:
        Hugging Face Inference API

    LLM:
        OpenRouter dynamic free-model router
    """

    _validate_environment()

    huggingface_api_key = os.getenv(
        "HUGGINGFACE_API_KEY"
    )

    openrouter_api_key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    # ---------------------------------------------------------------
    # Hugging Face embedding model
    # ---------------------------------------------------------------

    Settings.embed_model = (
        SynchronousHuggingFaceInferenceAPIEmbedding(
            model_name=HF_EMBEDDING_MODEL,
            token=huggingface_api_key,
        )
    )

    # ---------------------------------------------------------------
    # OpenRouter LLM
    # ---------------------------------------------------------------

    Settings.llm = OpenRouter(
        api_key=openrouter_api_key,
        model=OPENROUTER_MODEL,
        temperature=0.2,
        max_tokens=512,
        http_client=openrouter_http_client,
        async_http_client=openrouter_async_http_client,
    )

    # test_prompt = """
    # You are a SQLite SQL expert.

    # Generate ONLY the SQL query.

    # Table:
    # network_outages(
    #     outage_id TEXT,
    #     region TEXT,
    #     severity TEXT,
    #     start_time TEXT,
    #     end_time TEXT,
    #     duration_hours REAL,
    #     affected_customers INTEGER,
    #     root_cause TEXT,
    #     status TEXT,
    #     description TEXT
    # )

    # Question:
    # Which region had the most CRITICAL network outages recently?

    # Return only executable SQLite SQL.
    # """

    # response = Settings.llm.complete(test_prompt)

    # print("\n===== RAW LLM RESPONSE =====")
    # print(response)
    # print("===== END RAW RESPONSE =====\n")

    logger.info(
        "Embedding model configured: %s",
        HF_EMBEDDING_MODEL,
    )

    logger.info(
        "LLM configured through OpenRouter: %s",
        OPENROUTER_MODEL,
    )


# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

def _create_sql_database() -> SQLDatabase:
    """
    Create a SQLAlchemy engine and wrap it with LlamaIndex SQLDatabase.
    """

    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"SQLite database not found: {DATABASE_PATH}"
        )

    logger.info(
        "Connecting to SQLite database: %s",
        DATABASE_PATH,
    )

    engine = create_engine(
        f"sqlite:///{DATABASE_PATH}",
    )

    sql_database = SQLDatabase(
        engine,
    )

    logger.info(
        "SQLite database connected successfully."
    )

    logger.info(
        "Usable tables: %s",
        sql_database.get_usable_table_names(),
    )

    return sql_database


# ---------------------------------------------------------------------------
# Semantic table descriptions
# ---------------------------------------------------------------------------
#
# These descriptions are intentionally written around USER INTENT.
#
# The ObjectIndex embeds these descriptions and compares them with
# the natural-language question.
# ---------------------------------------------------------------------------

TABLE_CONTEXTS = {

    "network_towers": (
        "Contains telecom network tower inventory and tower "
        "operational information. Use this table for questions "
        "about tower locations, regions, cities, technology types, "
        "tower status, active or inactive towers, and network "
        "infrastructure."
    ),

    "network_outages": (
        "Contains historical network outage records. Use this "
        "table for questions about network outages, outage severity, "
        "CRITICAL incidents, affected customers, outage duration, "
        "incident IDs, outage dates, regions, and root causes."
    ),

    "tower_performance": (
        "Contains network tower performance measurements. Use this "
        "table for questions about latency, packet loss, throughput, "
        "signal strength, performance degradation, and tower-level "
        "network quality metrics."
    ),

    "open_incidents": (
        "Contains currently open NOC incidents associated with "
        "network infrastructure. Use this table for questions "
        "about active incidents, incident status, incident priority, "
        "open network issues, and incidents linked to towers."
    ),

    "customer_subscriptions": (
        "Contains customer subscription information. Use this "
        "table for questions about subscription plans, plan names, "
        "subscription fees, account types, customers by region, "
        "and customer subscription patterns."
    ),

    "billing_accounts": (
        "Contains telecom customer billing account information. "
        "Use this table for questions about customer billing "
        "accounts, account balances, billing status, and current "
        "amounts owed by customers."
    ),

    "billing_charges": (
        "Contains individual billing charge line items. Use this "
        "table for questions about bill charges, charge descriptions, "
        "billing periods, disputed charges, duplicate charges, and "
        "charge amounts."
    ),

    "billing_credits": (
        "Contains billing credits applied to customer accounts. "
        "Use this table for questions about credits, credit amounts, "
        "credit status, approved credits, pending credits, and "
        "billing adjustments."
    ),

    "billing_disputes": (
        "Contains customer billing dispute records. Use this "
        "table for questions about billing disputes, dispute status, "
        "disputed amounts, dispute reasons, opened disputes, and "
        "resolved billing disputes."
    ),
}


# ---------------------------------------------------------------------------
# Validate expected tables
# ---------------------------------------------------------------------------

def _validate_expected_tables(
    sql_database: SQLDatabase,
) -> None:
    """
    Ensure the tables required by the project are present.
    """

    available_tables = set(
        sql_database.get_usable_table_names()
    )

    expected_tables = set(
        TABLE_CONTEXTS.keys()
    )

    missing_tables = expected_tables - available_tables

    if missing_tables:
        raise RuntimeError(
            "The SQLite database is missing expected tables: "
            + ", ".join(sorted(missing_tables))
        )

    logger.info(
        "All expected semantic SQL tables are available."
    )


# ---------------------------------------------------------------------------
# ObjectIndex construction
# ---------------------------------------------------------------------------

def _build_table_retriever(
    sql_database: SQLDatabase,
):
    """
    Build an ObjectIndex containing SQLTableSchema objects.

    Flow:

        SQLTableSchema
              ↓
        SQLTableNodeMapping
              ↓
          ObjectIndex
              ↓
       VectorStoreIndex
              ↓
        semantic retriever
    """

    logger.info(
        "Creating SQLTableNodeMapping..."
    )

    table_node_mapping = SQLTableNodeMapping(
        sql_database
    )

    table_schema_objs = []

    for table_name, context_str in TABLE_CONTEXTS.items():

        table_schema = SQLTableSchema(
            table_name=table_name,
            context_str=context_str,
        )

        table_schema_objs.append(
            table_schema
        )

        logger.info(
            "Added semantic schema for table: %s",
            table_name,
        )

    logger.info(
        "Building ObjectIndex over table schemas..."
    )

    obj_index = ObjectIndex.from_objects(
        table_schema_objs,
        table_node_mapping,
        VectorStoreIndex,
    )

    # Small semantic retrieval depth as required
    table_retriever = obj_index.as_retriever(
        similarity_top_k=1,
    )

    logger.info(
        "ObjectIndex created with similarity_top_k=1."
    )

    return table_retriever


# ---------------------------------------------------------------------------
# Text-to-SQL prompt
# ---------------------------------------------------------------------------

# TEXT_TO_SQL_PROMPT = PromptTemplate(
#     """
# You are a SQLite SQL expert.

# Given a natural-language question and the relevant database table
# information, generate a valid SQLite SQL query that answers the question.

# Rules:
# 1. Use only the tables and columns provided in the table information.
# 2. Do not invent table names or columns.
# 3. Return ONLY the SQL query.
# 4. Do not return explanations.
# 5. Do not use markdown code fences.
# 6. Do not prefix the answer with "SQLQuery:".
# 7. The generated SQL must be executable SQLite SQL.
# 8. For ranking questions, use ORDER BY and LIMIT when appropriate.
# 9. For questions asking for the most or highest value, aggregate and
#    sort in descending order.
# 10. Use the actual column names from the schema.

# Table information:
# ---------------------
# {schema}
# ---------------------

# Question:
# {query_str}

# SQL:
# """
# )

TEXT_TO_SQL_PROMPT = PromptTemplate(
    """
You are an expert SQLite SQL generator.

Your task is to convert the user's question into ONE valid SQLite SQL query.

Rules:
1. Use only the tables and columns provided in the schema.
2. Do not invent tables or columns.
3. Return exactly one SQL query.
4. Do not explain the query.
5. Do not use Markdown code fences.
6. The query must be valid SQLite syntax.
7. For "most", "highest", or "largest" questions, use GROUP BY,
   ORDER BY ... DESC, and LIMIT when appropriate.
8. For "CRITICAL" outages, filter severity using:
   severity = 'CRITICAL'
9. For "recently", use the available date/time column when appropriate.
10. Start the response with SQLQuery:.

Schema:
---------------------
{schema}
---------------------

Question:
{query_str}

SQLQuery:
"""
)

# ---------------------------------------------------------------------------
# Query engine
# ---------------------------------------------------------------------------


def _create_query_engine(
    sql_database: SQLDatabase,
    table_retriever,
):
    """
    Create the SQLTableRetrieverQueryEngine.

    The table retriever first selects the semantically relevant table.
    The SQL query engine then generates SQLite SQL and executes it.
    """

    query_engine = SQLTableRetrieverQueryEngine(
        sql_database,
        table_retriever,
        llm=Settings.llm,
        text_to_sql_prompt=TEXT_TO_SQL_PROMPT,
        synthesize_response=True,
        verbose=True,
    )

    return query_engine


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def answer_sql_question(
    question: str,
) -> str:
    """
    Answer a natural-language question about operational data.

    Parameters
    ----------
    question : str
        Natural-language question about the telecom operational
        SQLite database.

    Returns
    -------
    str
        Natural-language answer generated from SQL results.
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

    logger.info(
        "Running semantic SQL query: %s",
        question,
    )

    try:

        # -----------------------------------------------------------
        # 1. Configure embedding model and LLM
        # -----------------------------------------------------------

        _configure_models()

        # -----------------------------------------------------------
        # 2. Connect to SQLite
        # -----------------------------------------------------------

        sql_database = _create_sql_database()

        # -----------------------------------------------------------
        # 3. Validate expected schema
        # -----------------------------------------------------------

        _validate_expected_tables(
            sql_database
        )

        # -----------------------------------------------------------
        # 4. Build semantic table retriever
        # -----------------------------------------------------------

        table_retriever = _build_table_retriever(
            sql_database
        )

        # -----------------------------------------------------------
        # 5. Create SQL query engine
        # -----------------------------------------------------------

        query_engine = _create_query_engine(
            sql_database,
            table_retriever,
        )

        # -----------------------------------------------------------
        # 6. Natural language → semantic table → SQL → answer
        # -----------------------------------------------------------

        response = query_engine.query(
            question
        )

        answer = str(response).strip()

        if not answer:
            return (
                "I could not generate an answer from the "
                "operational database."
            )

        return answer

    except Exception as exc:

        logger.exception(
            "Semantic SQL query failed."
        )

        raise RuntimeError(
            f"Failed to answer SQL question: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Direct execution test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(
            encoding="utf-8",
            errors="replace",
        )

    # test_question = (
    #     "Which region had the most CRITICAL "
    #     "network outages?"
    # )

    question=input("Enter question: ")

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")

    answer = answer_sql_question(
        question
    )

    print(answer)