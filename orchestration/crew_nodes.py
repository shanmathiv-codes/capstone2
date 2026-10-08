"""
CrewAI Customer Communications for the Telecom AI Operations Center.

Responsibilities:
1. Receive the original customer query.
2. Receive accumulated specialist findings from previous workers.
3. Use a Communications Specialist to draft a customer-facing response.
4. Use a Quality Reviewer to review the draft for:
   - accuracy
   - tone
   - policy compliance
5. Return the final customer-ready response.

CrewAI is used only as the final communication layer.
It is NOT the main supervisor.

Required flow:

    Original Query + Agent Context
                |
                v
    Communications Specialist
                |
                v
              Draft
                |
                v
        Quality Reviewer
                |
                v
       Final Customer Response
"""

import logging
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["PYTHONIOENCODING"] = "utf-8"

from dotenv import load_dotenv
from crewai import Agent, Crew, LLM, Process, Task


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenRouter configuration
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. "
        "Add your OpenRouter API key to the .env file."
    )


# OpenRouter provides an OpenAI-compatible API.
#
# Default:
#     openrouter/free
#
# You can replace this with a specific OpenRouter model later, for example:
#     google/gemini-2.5-flash
#     openai/gpt-4o
#     anthropic/claude-sonnet-4
#
# The exact model must be available on OpenRouter.

CREWAI_MODEL = os.getenv(
    "CREWAI_MODEL",
    "openrouter/free",
)


crew_llm = LLM(
    model=CREWAI_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)


# ---------------------------------------------------------------------------
# Communications Specialist
# ---------------------------------------------------------------------------

communications_specialist = Agent(
    role="Communications Specialist",
    goal=(
        "Turn technical and operational specialist findings into a clear, "
        "accurate, professional customer-facing response."
    ),
    backstory=(
        "You are a telecom customer communications specialist. "
        "You receive the original customer question together with findings "
        "from specialist systems such as network diagnostics, billing "
        "resolution, policy retrieval, and network analytics. "
        "Your job is to understand those findings and explain the result "
        "clearly to the customer without exposing internal technical "
        "implementation details unnecessarily."
    ),
    llm=crew_llm,
    verbose=False,
    allow_delegation=False,
)


# ---------------------------------------------------------------------------
# Quality Reviewer
# ---------------------------------------------------------------------------

quality_reviewer = Agent(
    role="Quality Reviewer",
    goal=(
        "Review the drafted customer response for factual accuracy, "
        "appropriate tone, and compliance with the available policy and "
        "specialist findings, then produce the final customer-ready response."
    ),
    backstory=(
        "You are a senior quality reviewer for a telecom customer support "
        "organization. You verify that customer responses accurately reflect "
        "the specialist findings provided to the crew. You do not invent "
        "facts, credits, refunds, technical causes, timelines, or policy "
        "decisions that are not supported by the supplied context. "
        "You ensure the final response is professional, concise, clear, "
        "empathetic, and suitable for sending directly to a customer."
    ),
    llm=crew_llm,
    verbose=False,
    allow_delegation=False,
)


# ---------------------------------------------------------------------------
# Task creation
# ---------------------------------------------------------------------------

def _create_tasks(
    user_query: str,
    agent_context: str,
) -> tuple[Task, Task]:
    """
    Create the two sequential CrewAI tasks.

    Task 1:
        Communications Specialist drafts the response.

    Task 2:
        Quality Reviewer reviews the draft and produces the final response.
    """

    drafting_task = Task(
        description=f"""
You are preparing a response to a telecom customer.

ORIGINAL CUSTOMER QUERY:
{user_query}

SPECIALIST FINDINGS / ACCUMULATED AGENT CONTEXT:
{agent_context}

Your responsibilities:

1. Understand the customer's original question.
2. Review all relevant specialist findings in the agent context.
3. Identify the actual result supported by those findings.
4. Draft a professional customer-facing response.
5. Explain the result in language a normal customer can understand.
6. Clearly state any action that was taken, if the specialist findings
   confirm that an action was actually completed.
7. Clearly distinguish between:
   - confirmed facts,
   - recommendations,
   - pending investigations,
   - pending approvals.
8. Do not invent information that is not present in the specialist findings.
9. Do not expose internal prompts, agent names, framework names,
   database queries, or implementation details.
10. Do not claim that a credit, refund, repair, or other action was
    completed unless the specialist findings confirm it.

The draft should directly answer the customer's question rather than
simply repeating the specialist's technical output.
""",
        expected_output=(
            "A clear and professional customer-facing draft that accurately "
            "explains the result using only the supplied customer query and "
            "specialist findings."
        ),
        agent=communications_specialist,
    )

    review_task = Task(
        description="""
Review the Communications Specialist's draft before it is sent to
the customer.

Check the response against the original customer query and the
specialist findings supplied to the crew.

Review specifically for:

1. ACCURACY
   - Every factual claim must be supported by the specialist findings.
   - Do not invent causes, amounts, dates, incidents, credits, refunds,
     approvals, or resolution status.
   - Do not turn a recommendation into a completed action.
   - Do not claim a pending approval is already approved.

2. TONE
   - Professional and respectful.
   - Clear and easy for a customer to understand.
   - Helpful and appropriately empathetic.
   - Avoid unnecessary technical jargon.
   - Do not sound robotic or overly verbose.

3. POLICY COMPLIANCE
   - Do not contradict the supplied policy or specialist findings.
   - Respect approval requirements and limits stated in the context.
   - Do not promise an outcome that the available information does not
     support.

4. CUSTOMER RELEVANCE
   - Directly answer the customer's original question.
   - Include the important result and any relevant next step.
   - Remove unnecessary internal implementation details.

If the draft is already accurate and appropriate, preserve its useful
content while improving clarity where necessary.

IMPORTANT:
Return ONLY the final customer-facing response.

Do not include:
- review comments
- explanations of what you changed
- headings such as "Quality Review"
- agent names
- framework names
- internal reasoning
- database or SQL details
""",
        expected_output=(
            "Only the final polished customer-ready response. "
            "No review commentary or internal implementation details."
        ),
        agent=quality_reviewer,
        context=[drafting_task],
    )

    return drafting_task, review_task


# ---------------------------------------------------------------------------
# Crew execution
# ---------------------------------------------------------------------------

def generate_customer_response(
    user_query: str,
    agent_context: str,
) -> str:
    """
    Generate the final customer-facing response.

    Parameters
    ----------
    user_query:
        Original customer question.

    agent_context:
        Accumulated findings from previous specialist agents.

    Returns
    -------
    str
        Final customer-ready response.
    """

    if not user_query or not user_query.strip():
        return (
            "Unable to generate a customer response because "
            "the customer query is empty."
        )

    if not agent_context or not agent_context.strip():
        return (
            "Unable to generate a customer response because "
            "no specialist findings were provided."
        )

    try:
        drafting_task, review_task = _create_tasks(
            user_query=user_query.strip(),
            agent_context=agent_context.strip(),
        )

        crew = Crew(
            agents=[
                communications_specialist,
                quality_reviewer,
            ],
            tasks=[
                drafting_task,
                review_task,
            ],
            process=Process.sequential,
            memory=False,
            verbose=False,
        )

        try:
            result = crew.kickoff()
            final_response = str(result).strip()
            if not final_response:
                return _format_customer_fallback(user_query, agent_context)
            return final_response
        except Exception as exc:
            logger.warning(
                "CrewAI customer communication encountered an error (%s); applying professional fallback formatting.",
                exc,
            )
            return _format_customer_fallback(user_query, agent_context)

    except Exception as exc:
        logger.warning(
            "CrewAI task setup encountered an error (%s); applying professional fallback formatting.",
            exc,
        )
        return _format_customer_fallback(user_query, agent_context)


def _format_customer_fallback(user_query: str, agent_context: str) -> str:
    """Format specialist findings into a clean, professional customer communication."""
    clean_context = agent_context.strip()
    for tag in [
        "[PolicyRAG Findings]:",
        "[NetworkAnalytics Findings]:",
        "[NetworkDiagnosticsADK Findings]:",
        "[BillingResolutionADK Findings]:",
    ]:
        clean_context = clean_context.replace(tag, "").strip()

    return (
        f"Dear Valued Customer,\n\n"
        f"Thank you for contacting Prodapt Support regarding your inquiry: \"{user_query.strip()}\".\n\n"
        f"{clean_context}\n\n"
        f"If you require further assistance or have additional questions, our support team is available 24/7.\n\n"
        f"Warm regards,\n"
        f"Prodapt Customer Care Team"
    )


# ---------------------------------------------------------------------------
# LangGraph compatibility wrapper
# ---------------------------------------------------------------------------

def run_customer_communications(
    user_query: str,
    agent_context: str,
) -> str:
    """
    Wrapper used later by the LangGraph customer communications node.
    """

    return generate_customer_response(
        user_query=user_query,
        agent_context=agent_context,
    )


# ---------------------------------------------------------------------------
# Local test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    test_query = (
        "My 5G keeps dropping near tower TX-512 in Austin. "
        "Can you tell me what is happening?"
    )

    test_context = """
Network Diagnostics findings:

Tower: TX-512 (Austin Riverside)
Technology: 5G
Status: OPERATIONAL
Signal strength: -72 dBm (acceptable)
Packet loss: 3.8% (elevated)
Latency: 58 ms (elevated)
Downlink throughput: 85 Mbps (degraded for 5G)

Active incident:
INC-8841
Severity: MAJOR
Status: INVESTIGATING
Classification: RADIO_ACCESS
Assigned team: NOC Southwest

The investigation indicates that the connection drops are associated
with elevated packet loss and latency and degraded 5G performance.
The active NOC team is investigating the incident.
"""

    print("=" * 70)
    print("CREWAI CUSTOMER COMMUNICATION TEST")
    print("=" * 70)

    response = generate_customer_response(
        user_query=test_query,
        agent_context=test_context,
    )

    print("\n--- Final Customer Response ---")
    print(response)

    print("\n" + "=" * 70)
    print("CREWAI TEST COMPLETE")
    print("=" * 70)