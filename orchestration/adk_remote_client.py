import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.runners import InMemoryRunner
from google.genai import types


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NETWORK_DIAGNOSTICS_AGENT_CARD = (
    "http://localhost:8001/.well-known/agent-card.json"
)

BILLING_RESOLUTION_AGENT_CARD = (
    "http://localhost:8002/.well-known/agent-card.json"
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Remote A2A agents
# ---------------------------------------------------------------------------

network_diagnostics_remote = RemoteA2aAgent(
    name="network_diagnostics_remote",
    description=(
        "Remote telecom network diagnostics agent. "
        "Handles tower status, connectivity diagnostics, "
        "and regional network summaries."
    ),
    agent_card=NETWORK_DIAGNOSTICS_AGENT_CARD,
    use_legacy=False,
)


billing_resolution_remote = RemoteA2aAgent(
    name="billing_resolution_remote",
    description=(
        "Remote telecom billing resolution agent. "
        "Investigates billing disputes, duplicate charges, "
        "and billing-credit actions."
    ),
    agent_card=BILLING_RESOLUTION_AGENT_CARD,
    use_legacy=False,
)


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

network_runner = InMemoryRunner(
    agent=network_diagnostics_remote,
    app_name="network_diagnostics_remote_client",
)


billing_runner = InMemoryRunner(
    agent=billing_resolution_remote,
    app_name="billing_resolution_remote_client",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _run_remote_agent(
    runner: InMemoryRunner,
    query: str,
    user_id: str,
    session_id: str,
) -> dict[str, Any]:
    """
    Send a text query to a remote A2A agent through an ADK Runner.

    The Runner executes the RemoteA2aAgent, which communicates with
    the remote A2A service using the agent card supplied when the
    RemoteA2aAgent was created.

    Args:
        runner: ADK InMemoryRunner containing a RemoteA2aAgent.
        query: Natural-language request.
        user_id: User/session owner identifier.
        session_id: Unique session identifier.

    Returns:
        Dictionary containing success status and remote response.
    """

    try:
        # ---------------------------------------------------------------
        # Create the ADK user message.
        # ---------------------------------------------------------------

        message = types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=query)
            ],
        )

        # ---------------------------------------------------------------
        # Make sure the session exists.
        #
        # InMemoryRunner does not automatically create a missing session
        # for run_async(), so create it explicitly.
        # ---------------------------------------------------------------

        existing_session = await runner.session_service.get_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )

        if existing_session is None:
            await runner.session_service.create_session(
                app_name=runner.app_name,
                user_id=user_id,
                session_id=session_id,
            )

        # ---------------------------------------------------------------
        # Execute the remote agent.
        #
        # Runner.run_async() yields ADK Event objects.
        # ---------------------------------------------------------------

        response_parts = []
        events = []

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=message,
        ):
            events.append(event)

            # -----------------------------------------------------------
            # Collect textual content from events.
            # -----------------------------------------------------------

            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        response_parts.append(part.text)

        response_text = "\n".join(response_parts).strip()

        if not response_text:
            return {
                "success": False,
                "error": "Remote agent returned no textual response.",
                "events_received": len(events),
            }

        return {
            "success": True,
            "response": response_text,
            "events_received": len(events),
        }

    except Exception as exc:
        logger.exception(
            "Remote A2A call failed: %s",
            exc,
        )

        return {
            "success": False,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Async wrappers
# ---------------------------------------------------------------------------

async def ask_network_diagnostics_async(
    query: str,
    user_id: str = "langgraph_user",
    session_id: str = "network_diagnostics_session",
) -> dict[str, Any]:
    """
    Asynchronously send a query to the Network Diagnostics ADK agent.

    The Network Diagnostics service must be running on port 8001.
    """

    logger.info(
        "Sending request to Network Diagnostics A2A service: %s",
        query,
    )

    result = await _run_remote_agent(
        runner=network_runner,
        query=query,
        user_id=user_id,
        session_id=session_id,
    )

    if not result["success"]:
        result["service"] = "network_diagnostics"
        result["service_url"] = NETWORK_DIAGNOSTICS_AGENT_CARD

    return result


async def ask_billing_resolution_async(
    query: str,
    user_id: str = "langgraph_user",
    session_id: str = "billing_resolution_session",
) -> dict[str, Any]:
    """
    Asynchronously send a query to the Billing Resolution ADK agent.

    The Billing Resolution service must be running on port 8002.
    """

    logger.info(
        "Sending request to Billing Resolution A2A service: %s",
        query,
    )

    result = await _run_remote_agent(
        runner=billing_runner,
        query=query,
        user_id=user_id,
        session_id=session_id,
    )

    if not result["success"]:
        result["service"] = "billing_resolution"
        result["service_url"] = BILLING_RESOLUTION_AGENT_CARD

    return result


# ---------------------------------------------------------------------------
# Synchronous execution helper
# ---------------------------------------------------------------------------

def _run_coroutine_sync(coroutine):
    """
    Execute an async coroutine from synchronous LangGraph code.

    Normal case:
        asyncio.run() is sufficient.

    If the caller already has an active event loop in the current
    thread, execute the coroutine in a separate thread instead.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No event loop is running in this thread.
        return asyncio.run(coroutine)

    # An event loop is already running.
    #
    # asyncio.run() cannot be called inside an active event loop,
    # so execute the coroutine in a separate worker thread.
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coroutine)
        return future.result()


# ---------------------------------------------------------------------------
# Synchronous wrappers for LangGraph
# ---------------------------------------------------------------------------

def ask_network_diagnostics(
    query: str,
    user_id: str = "langgraph_user",
    session_id: str = "network_diagnostics_session",
) -> dict[str, Any]:
    """
    Synchronous wrapper for LangGraph.

    Example:

        result = ask_network_diagnostics(
            "My 5G keeps dropping in Austin near tower TX-512."
        )
    """

    return _run_coroutine_sync(
        ask_network_diagnostics_async(
            query=query,
            user_id=user_id,
            session_id=session_id,
        )
    )


def ask_billing_resolution(
    query: str,
    user_id: str = "langgraph_user",
    session_id: str = "billing_resolution_session",
) -> dict[str, Any]:
    """
    Synchronous wrapper for LangGraph.

    Example:

        result = ask_billing_resolution(
            "Customer CUST-10002 was charged twice "
            "for Unlimited Plus. Investigate and apply credit."
        )
    """

    return _run_coroutine_sync(
        ask_billing_resolution_async(
            query=query,
            user_id=user_id,
            session_id=session_id,
        )
    )


# ---------------------------------------------------------------------------
# Direct A2A client test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("A2A REMOTE CLIENT TEST")
    print("=" * 70)

    print("\n--- Network Diagnostics A2A ---")

    network_result = ask_network_diagnostics(
        "My 5G keeps dropping in Austin near tower TX-512. "
        "Please diagnose the problem."
    )

    print(network_result)

    print("\n--- Billing Resolution A2A ---")

    billing_result = ask_billing_resolution(
        "Customer CUST-10002 was charged twice for Unlimited Plus. "
        "Investigate the duplicate charge."
    )

    print(billing_result)

    print("\n" + "=" * 70)
    print("A2A REMOTE CLIENT TEST COMPLETE")
    print("=" * 70)