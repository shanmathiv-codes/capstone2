"""
LangGraph Orchestration for Prodapt AI Operations Center.

Implements:
1. Supervisor router node that determines the next worker or FINISH.
2. Specialist worker nodes:
   - PolicyRAG (LlamaIndex Document RAG over policy documents)
   - NetworkAnalytics (LlamaIndex Semantic SQL over telecom_ops.db)
   - NetworkDiagnosticsADK (Remote Google ADK agent via A2A on port 8001)
   - BillingResolutionADK (Remote Google ADK agent via A2A on port 8002)
   - CustomerCommsCrew (CrewAI sequential crew for customer-ready communication)
3. Compiled StateGraph with loop-back to supervisor until FINISH.
4. Public entry function:
       run_telecom_assistant(user_query: str) -> dict
"""

import logging
import os
import re
from typing import Literal

from dotenv import load_dotenv
import httpx
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestration.state import AgentState

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Routing Model & Supervisor LLM
# ---------------------------------------------------------------------------

WORKER_NAMES = [
    "PolicyRAG",
    "NetworkAnalytics",
    "NetworkDiagnosticsADK",
    "BillingResolutionADK",
    "CustomerCommsCrew",
    "FINISH",
]


class RouteDecision(BaseModel):
    """Structured decision output from the supervisor LLM."""

    next: Literal[
        "PolicyRAG",
        "NetworkAnalytics",
        "NetworkDiagnosticsADK",
        "BillingResolutionADK",
        "CustomerCommsCrew",
        "FINISH",
    ] = Field(
        description="The next worker to invoke or FINISH if all tasks are complete."
    )
    reason: str = Field(description="Reasoning explaining the routing choice.")


def _get_supervisor_llm():
    """Create the LLM instance for structured supervisor decisions."""
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        return None

    try:
        http_client = httpx.Client(verify=False, timeout=60.0)
        return ChatOpenAI(
            model=os.getenv("SUPERVISOR_MODEL", "openrouter/free"),
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            http_client=http_client,
            temperature=0.0,
        )
    except Exception as exc:
        logger.warning("Could not initialize supervisor LLM: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Deterministic Routing Helper (Specification Section 9.6)
# ---------------------------------------------------------------------------

def classify_routing_decision(user_query: str, executed_workers: list[str]) -> str:
    """
    Apply the official Supervisor Routing Decision Matrix from the specification:
    - Policy/FAQ: policy, roaming, SLA rules, 5G FAQ, upgrade -> PolicyRAG
    - Analytics: outage trends, packet loss ranking, critical outages -> NetworkAnalytics
    - Connectivity: signal drops, tower ID, diagnose, not working -> NetworkDiagnosticsADK
    - Billing: charged twice, credit, dispute, CUST- prefix -> BillingResolutionADK
    - Combined: outage + SLA credit policy -> NetworkAnalytics first, then PolicyRAG
    """
    q_lower = user_query.lower()

    # Rule 1: If CustomerCommsCrew has already executed, flow is complete.
    if "CustomerCommsCrew" in executed_workers:
        return "FINISH"

    # Rule 2: Check for Multi-Worker Combined Flow (Scenario 5)
    # Signals: outage/analytics AND policy/credit eligibility
    is_outage_query = bool(
        re.search(r"\b(outage|outages|down|downtime|cut|incident)\b", q_lower)
    )
    is_policy_credit = bool(
        re.search(r"\b(sla|credit|policy|eligible|eligibility|threshold)\b", q_lower)
    )

    if is_outage_query and is_policy_credit:
        # Multi-worker sequence: NetworkAnalytics -> PolicyRAG -> CustomerCommsCrew
        if "NetworkAnalytics" not in executed_workers:
            return "NetworkAnalytics"
        if "PolicyRAG" not in executed_workers:
            return "PolicyRAG"
        return "CustomerCommsCrew"

    # Rule 3: Single-domain flows: If any specialist has completed, route to comms crew
    specialists = [
        "PolicyRAG",
        "NetworkAnalytics",
        "NetworkDiagnosticsADK",
        "BillingResolutionADK",
    ]
    has_specialist_run = any(s in executed_workers for s in specialists)
    if has_specialist_run:
        return "CustomerCommsCrew"

    # Rule 4: First routing step based on intent keywords

    # Billing Dispute (Scenario 4)
    if (
        "cust-" in q_lower
        or "charged twice" in q_lower
        or "double charge" in q_lower
        or "dispute" in q_lower
        or "billing" in q_lower
        or "credit" in q_lower
    ):
        return "BillingResolutionADK"

    # Live Network Diagnostics (Scenario 3)
    if (
        re.search(r"\b(tx-\d+|fl-\d+|il-\d+|ca-\d+|ny-\d+|tower)\b", q_lower)
        or "drops" in q_lower
        or "diagnose" in q_lower
        or "troubleshoot" in q_lower
        or "signal drops" in q_lower
        or "connectivity" in q_lower
    ):
        return "NetworkDiagnosticsADK"

    # Network Analytics (Scenario 2)
    if (
        re.search(r"\b(critical|packet loss|highest|most|latency|throughput)\b", q_lower)
        or "how many" in q_lower
        or "ranking" in q_lower
        or "outages" in q_lower
    ):
        return "NetworkAnalytics"

    # Policy / FAQ (Scenario 1)
    if (
        "policy" in q_lower
        or "roaming" in q_lower
        or "travel pass" in q_lower
        or "trade-in" in q_lower
        or "trade in" in q_lower
        or "upgrade" in q_lower
        or "indoor" in q_lower
        or "faq" in q_lower
        or "sla" in q_lower
    ):
        return "PolicyRAG"

    # Default fallback
    return "PolicyRAG"


# ---------------------------------------------------------------------------
# Node 1: Supervisor Router
# ---------------------------------------------------------------------------

def supervisor_node(state: AgentState) -> dict:
    """
    Supervisor router that reads user_query, execution trace, and prior context
    to select the next worker node or FINISH.
    """
    user_query = state.get("user_query", "").strip()
    execution_trace = state.get("execution_trace", [])
    executed_workers = [step["worker"] for step in execution_trace]

    logger.info("Supervisor evaluating state. Executed workers so far: %s", executed_workers)

    # 1. Check if CustomerCommsCrew already ran
    if "CustomerCommsCrew" in executed_workers:
        logger.info("Supervisor decided: FINISH (Comms crew already ran)")
        return {"next": "FINISH"}

    # 2. Try structured LLM routing
    decision = None
    llm = _get_supervisor_llm()
    if llm:
        try:
            structured_llm = llm.with_structured_output(RouteDecision)
            prompt = f"""
You are the Supervisor Router for the Prodapt Telecom AI Operations Center.
Your role is to route the inquiry to the appropriate specialist worker according to these rules:

AVAILABLE WORKERS:
1. PolicyRAG: For policy, roaming, device trade-in/upgrade, SLA terms, and 5G FAQ questions.
2. NetworkAnalytics: For operational analytics, outage statistics, regional outage queries, and tower packet loss rankings from SQL.
3. NetworkDiagnosticsADK: For live tower diagnostics (e.g. TX-512), signal drops, or connectivity troubleshooting.
4. BillingResolutionADK: For customer billing disputes, duplicate charges (e.g. CUST-10002), and credit adjustments.
5. CustomerCommsCrew: Customer communications crew that formats the final answer once specialist data is gathered.
6. FINISH: Only when CustomerCommsCrew has finished and the final customer response is ready.

WORKERS ALREADY EXECUTED:
{executed_workers}

CURRENT AGENT CONTEXT:
{state.get('agent_context', '(none)')}

CUSTOMER INQUIRY:
{user_query}

ROUTING MATRIX:
- If a specialist has run and all required data is collected, route to CustomerCommsCrew.
- If CustomerCommsCrew has run, route to FINISH.
- For combined queries (outage data + SLA credit policy), run NetworkAnalytics, then PolicyRAG, then CustomerCommsCrew.

Pick the next worker.
"""
            result = structured_llm.invoke(prompt)
            if result and result.next in WORKER_NAMES:
                decision = result.next
                # Safeguard: Do not finish before comms crew runs
                if decision == "FINISH" and "CustomerCommsCrew" not in executed_workers:
                    decision = "CustomerCommsCrew"
                logger.info("Supervisor LLM selected: %s (reason: %s)", decision, result.reason)
        except Exception as exc:
            logger.warning("Supervisor LLM structured call failed or timed out: %s", exc)
            decision = None

    # 3. Fallback to exact specification matrix if LLM unavailable or invalid
    if not decision or decision not in WORKER_NAMES:
        decision = classify_routing_decision(user_query, executed_workers)
        logger.info("Supervisor matrix selected: %s", decision)

    return {"next": decision}


# ---------------------------------------------------------------------------
# Node 2: PolicyRAG
# ---------------------------------------------------------------------------

def policy_rag_node(state: AgentState) -> dict:
    """Invokes LlamaIndex Document RAG over policy documents."""
    user_query = state.get("user_query", "")
    logger.info("PolicyRAG executing query: %s", user_query)

    try:
        from llamaindex_rag.document_rag import answer_policy_question
        output = answer_policy_question(user_query)
    except Exception as exc:
        logger.exception("PolicyRAG execution failed")
        output = f"PolicyRAG Error: {exc}"

    output_str = str(output).strip()
    truncated_output = output_str[:500] + "..." if len(output_str) > 500 else output_str

    trace_entry = {
        "worker": "PolicyRAG",
        "output": truncated_output,
    }

    prior_context = state.get("agent_context", "").strip()
    new_context = (
        f"{prior_context}\n\n[PolicyRAG Findings]:\n{output_str}"
        if prior_context
        else f"[PolicyRAG Findings]:\n{output_str}"
    )

    return {
        "agent_context": new_context,
        "execution_trace": state.get("execution_trace", []) + [trace_entry],
        "messages": [AIMessage(content=f"[PolicyRAG]: {truncated_output}")],
    }


# ---------------------------------------------------------------------------
# Node 3: NetworkAnalytics
# ---------------------------------------------------------------------------

def network_analytics_node(state: AgentState) -> dict:
    """Invokes LlamaIndex Semantic SQL over telecom_ops.db."""
    user_query = state.get("user_query", "")
    logger.info("NetworkAnalytics executing query: %s", user_query)

    try:
        from llamaindex_rag.sql_semantic_search import answer_sql_question
        output = answer_sql_question(user_query)
    except Exception as exc:
        logger.exception("NetworkAnalytics execution failed")
        output = f"NetworkAnalytics Error: {exc}"

    output_str = str(output).strip()
    truncated_output = output_str[:500] + "..." if len(output_str) > 500 else output_str

    trace_entry = {
        "worker": "NetworkAnalytics",
        "output": truncated_output,
    }

    prior_context = state.get("agent_context", "").strip()
    new_context = (
        f"{prior_context}\n\n[NetworkAnalytics Findings]:\n{output_str}"
        if prior_context
        else f"[NetworkAnalytics Findings]:\n{output_str}"
    )

    return {
        "agent_context": new_context,
        "execution_trace": state.get("execution_trace", []) + [trace_entry],
        "messages": [AIMessage(content=f"[NetworkAnalytics]: {truncated_output}")],
    }


# ---------------------------------------------------------------------------
# Node 4: NetworkDiagnosticsADK
# ---------------------------------------------------------------------------

def network_diagnostics_adk_node(state: AgentState) -> dict:
    """Invokes remote Google ADK Network Diagnostics agent via A2A on port 8001."""
    user_query = state.get("user_query", "")
    logger.info("NetworkDiagnosticsADK executing query: %s", user_query)

    try:
        from orchestration.adk_remote_client import ask_network_diagnostics
        res = ask_network_diagnostics(user_query)
        if res.get("success"):
            output = res.get("response", "")
        else:
            err = res.get("error", "Remote service unreachable")
            output = (
                f"Network Diagnostics ADK service status: {err}. "
                "Ensure adk-services/network_diagnostics/agent.py is running on port 8001."
            )
    except Exception as exc:
        logger.exception("NetworkDiagnosticsADK invocation failed")
        output = f"NetworkDiagnosticsADK Error: {exc}"

    output_str = str(output).strip()
    truncated_output = output_str[:500] + "..." if len(output_str) > 500 else output_str

    trace_entry = {
        "worker": "NetworkDiagnosticsADK",
        "output": truncated_output,
    }

    prior_context = state.get("agent_context", "").strip()
    new_context = (
        f"{prior_context}\n\n[NetworkDiagnosticsADK Findings]:\n{output_str}"
        if prior_context
        else f"[NetworkDiagnosticsADK Findings]:\n{output_str}"
    )

    return {
        "agent_context": new_context,
        "execution_trace": state.get("execution_trace", []) + [trace_entry],
        "messages": [AIMessage(content=f"[NetworkDiagnosticsADK]: {truncated_output}")],
    }


# ---------------------------------------------------------------------------
# Node 5: BillingResolutionADK
# ---------------------------------------------------------------------------

def billing_resolution_adk_node(state: AgentState) -> dict:
    """Invokes remote Google ADK Billing Resolution agent via A2A on port 8002."""
    user_query = state.get("user_query", "")
    logger.info("BillingResolutionADK executing query: %s", user_query)

    try:
        from orchestration.adk_remote_client import ask_billing_resolution
        res = ask_billing_resolution(user_query)
        if res.get("success"):
            output = res.get("response", "")
        else:
            err = res.get("error", "Remote service unreachable")
            output = (
                f"Billing Resolution ADK service status: {err}. "
                "Ensure adk-services/billing_resolution/agent.py is running on port 8002."
            )
    except Exception as exc:
        logger.exception("BillingResolutionADK invocation failed")
        output = f"BillingResolutionADK Error: {exc}"

    output_str = str(output).strip()
    truncated_output = output_str[:500] + "..." if len(output_str) > 500 else output_str

    trace_entry = {
        "worker": "BillingResolutionADK",
        "output": truncated_output,
    }

    prior_context = state.get("agent_context", "").strip()
    new_context = (
        f"{prior_context}\n\n[BillingResolutionADK Findings]:\n{output_str}"
        if prior_context
        else f"[BillingResolutionADK Findings]:\n{output_str}"
    )

    return {
        "agent_context": new_context,
        "execution_trace": state.get("execution_trace", []) + [trace_entry],
        "messages": [AIMessage(content=f"[BillingResolutionADK]: {truncated_output}")],
    }


# ---------------------------------------------------------------------------
# Node 6: CustomerCommsCrew
# ---------------------------------------------------------------------------

def customer_comms_crew_node(state: AgentState) -> dict:
    """Invokes CrewAI communications crew to polish specialist findings for customer."""
    user_query = state.get("user_query", "")
    agent_context = state.get("agent_context", "")
    logger.info("CustomerCommsCrew synthesizing customer response.")

    try:
        from orchestration.crew_nodes import run_customer_communications
        final_response = run_customer_communications(user_query, agent_context)
    except Exception as exc:
        logger.exception("CustomerCommsCrew failed")
        final_response = (
            f"Customer Communications Error: {exc}\n\n"
            f"Specialist findings:\n{agent_context}"
        )

    output_str = str(final_response).strip()
    truncated_output = output_str[:500] + "..." if len(output_str) > 500 else output_str

    trace_entry = {
        "worker": "CustomerCommsCrew",
        "output": truncated_output,
    }

    return {
        "final_response": output_str,
        "execution_trace": state.get("execution_trace", []) + [trace_entry],
        "messages": [AIMessage(content=f"[CustomerCommsCrew]: {truncated_output}")],
    }


# ---------------------------------------------------------------------------
# Graph Compilation
# ---------------------------------------------------------------------------

def build_telecom_graph():
    """Build and compile the LangGraph StateGraph workflow."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("PolicyRAG", policy_rag_node)
    workflow.add_node("NetworkAnalytics", network_analytics_node)
    workflow.add_node("NetworkDiagnosticsADK", network_diagnostics_adk_node)
    workflow.add_node("BillingResolutionADK", billing_resolution_adk_node)
    workflow.add_node("CustomerCommsCrew", customer_comms_crew_node)

    # Entry point
    workflow.set_entry_point("supervisor")

    # Conditional routing from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "PolicyRAG": "PolicyRAG",
            "NetworkAnalytics": "NetworkAnalytics",
            "NetworkDiagnosticsADK": "NetworkDiagnosticsADK",
            "BillingResolutionADK": "BillingResolutionADK",
            "CustomerCommsCrew": "CustomerCommsCrew",
            "FINISH": END,
        },
    )

    # Every worker returns to supervisor until FINISH
    workflow.add_edge("PolicyRAG", "supervisor")
    workflow.add_edge("NetworkAnalytics", "supervisor")
    workflow.add_edge("NetworkDiagnosticsADK", "supervisor")
    workflow.add_edge("BillingResolutionADK", "supervisor")
    workflow.add_edge("CustomerCommsCrew", "supervisor")

    return workflow.compile()


# Singleton compiled graph instance
telecom_graph = build_telecom_graph()


# ---------------------------------------------------------------------------
# Public Graph Entry Function (Specification Section 6.8.4)
# ---------------------------------------------------------------------------

def run_telecom_assistant(user_query: str) -> dict:
    """
    Public entry point for Streamlit UI and external callers.

    Contract:
        Returns:
            dict with:
                - 'final_response': str (customer-ready answer from CustomerCommsCrew)
                - 'execution_trace': list of dicts [{"worker": str, "output": str}]
                - 'agent_context': str (accumulated raw specialist context)
    """
    if not user_query or not user_query.strip():
        return {
            "final_response": "Please provide a valid question or inquiry.",
            "execution_trace": [],
            "agent_context": "",
        }

    initial_state = {
        "messages": [HumanMessage(content=user_query.strip())],
        "next": "",
        "user_query": user_query.strip(),
        "agent_context": "",
        "execution_trace": [],
        "final_response": "",
    }

    try:
        final_state = telecom_graph.invoke(initial_state)

        # Fallback if final_response was not populated
        final_resp = final_state.get("final_response", "").strip()
        if not final_resp:
            final_resp = final_state.get("agent_context", "Inquiry processed successfully.")

        return {
            "final_response": final_resp,
            "execution_trace": final_state.get("execution_trace", []),
            "agent_context": final_state.get("agent_context", ""),
        }
    except Exception as exc:
        logger.exception("run_telecom_assistant failed")
        return {
            "final_response": f"An error occurred while processing your request: {exc}",
            "execution_trace": initial_state.get("execution_trace", []),
            "agent_context": str(exc),
        }


# ---------------------------------------------------------------------------
# CLI Direct Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=" * 70)
    print("PRODAPT TELECOM ORCHESTRATION GRAPH TEST")
    print("=" * 70)

    test_queries = [
        "What is Prodapt's roaming policy for Western Europe?",
        "Which region had the most CRITICAL network outages recently?",
        "My 5G keeps dropping near tower TX-512 in Austin. Please diagnose.",
        "Customer CUST-10002 was charged twice for Unlimited Plus. Investigate and apply credit.",
        "We had a 6-hour outage in the Midwest. Am I eligible for an SLA credit and what does policy say?",
    ]

    for idx, q in enumerate(test_queries, 1):
        print(f"\nScenario {idx}: {q}")
        decision = classify_routing_decision(q, [])
        print(f" -> First step routed: {decision}")
