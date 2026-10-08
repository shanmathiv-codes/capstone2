import logging
import sqlite3
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from google.genai import types
from google.adk.models import Gemini


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "telecom_ops.db"

load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database helper
# ---------------------------------------------------------------------------

def _get_connection() -> sqlite3.Connection:
    """
    Create a SQLite connection to the telecom operations database.
    """
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


# ---------------------------------------------------------------------------
# Tool 1: Check tower status
# ---------------------------------------------------------------------------

async def check_tower_status(tower_id: str) -> dict[str, Any]:
    """
    Retrieve the current tower status, technology, latest performance
    metrics, and active incidents for a tower.

    Args:
        tower_id: Tower identifier such as TX-512.

    Returns:
        Dictionary containing tower information, latest performance,
        and open incidents.
    """

    logger.info("Checking tower status for tower_id=%s", tower_id)

    try:
        with _get_connection() as conn:

            # ---------------------------------------------------------------
            # Tower inventory
            # ---------------------------------------------------------------

            tower = conn.execute(
                """
                SELECT
                    tower_id,
                    tower_name,
                    region,
                    city,
                    state,
                    technology,
                    status,
                    latitude,
                    longitude,
                    commissioned_date
                FROM network_towers
                WHERE tower_id = ?
                """,
                (tower_id,),
            ).fetchone()

            if tower is None:
                return {
                    "success": False,
                    "tower_id": tower_id,
                    "error": f"Tower '{tower_id}' was not found.",
                }

            # ---------------------------------------------------------------
            # Latest performance record
            #
            # The schema explicitly requires diagnostics to use the
            # latest row per tower.
            # ---------------------------------------------------------------

            performance = conn.execute(
                """
                SELECT
                    performance_id,
                    tower_id,
                    recorded_at,
                    latency_ms,
                    packet_loss_pct,
                    downlink_throughput_mbps,
                    uplink_throughput_mbps,
                    signal_strength_dbm,
                    active_connections
                FROM tower_performance
                WHERE tower_id = ?
                ORDER BY recorded_at DESC
                LIMIT 1
                """,
                (tower_id,),
            ).fetchone()

            # ---------------------------------------------------------------
            # Active incidents
            # ---------------------------------------------------------------

            incidents = conn.execute(
                """
                SELECT
                    incident_id,
                    severity,
                    status,
                    title,
                    description,
                    opened_at,
                    classification,
                    assigned_team
                FROM open_incidents
                WHERE tower_id = ?
                ORDER BY opened_at DESC
                """,
                (tower_id,),
            ).fetchall()

            result = {
                "success": True,
                "tower": dict(tower),
                "latest_performance": (
                    dict(performance) if performance else None
                ),
                "open_incidents": [dict(row) for row in incidents],
            }

            logger.info(
                "Tower status retrieved successfully for tower_id=%s",
                tower_id,
            )

            return result

    except sqlite3.Error as exc:
        logger.exception(
            "Database error while checking tower %s",
            tower_id,
        )

        return {
            "success": False,
            "tower_id": tower_id,
            "error": f"Database error: {exc}",
        }


# ---------------------------------------------------------------------------
# Tool 2: Run connectivity diagnostics
# ---------------------------------------------------------------------------

async def run_connectivity_diagnostics(
    tower_id: str,
    symptom: str,
) -> dict[str, Any]:
    """
    Diagnose a connectivity problem using the tower's current status
    and latest performance metrics.

    Args:
        tower_id: Tower identifier such as TX-512.
        symptom: Customer-reported symptom.

    Returns:
        Diagnostic findings and recommendations.
    """

    logger.info(
        "Running connectivity diagnostics: tower=%s symptom=%s",
        tower_id,
        symptom,
    )

    try:
        with _get_connection() as conn:

            # ---------------------------------------------------------------
            # Retrieve tower
            # ---------------------------------------------------------------

            tower = conn.execute(
                """
                SELECT
                    tower_id,
                    tower_name,
                    region,
                    city,
                    state,
                    technology,
                    status
                FROM network_towers
                WHERE tower_id = ?
                """,
                (tower_id,),
            ).fetchone()

            if tower is None:
                return {
                    "success": False,
                    "tower_id": tower_id,
                    "symptom": symptom,
                    "error": f"Tower '{tower_id}' was not found.",
                }

            # ---------------------------------------------------------------
            # Retrieve latest performance record
            # ---------------------------------------------------------------

            performance = conn.execute(
                """
                SELECT
                    recorded_at,
                    latency_ms,
                    packet_loss_pct,
                    downlink_throughput_mbps,
                    uplink_throughput_mbps,
                    signal_strength_dbm,
                    active_connections
                FROM tower_performance
                WHERE tower_id = ?
                ORDER BY recorded_at DESC
                LIMIT 1
                """,
                (tower_id,),
            ).fetchone()

            if performance is None:
                return {
                    "success": False,
                    "tower_id": tower_id,
                    "symptom": symptom,
                    "error": (
                        f"No performance data was found for tower "
                        f"'{tower_id}'."
                    ),
                }

            # ---------------------------------------------------------------
            # Retrieve active incidents
            # ---------------------------------------------------------------

            incidents = conn.execute(
                """
                SELECT
                    incident_id,
                    severity,
                    status,
                    title,
                    description,
                    classification,
                    assigned_team
                FROM open_incidents
                WHERE tower_id = ?
                ORDER BY opened_at DESC
                """,
                (tower_id,),
            ).fetchall()

            # Convert SQLite rows to dictionaries
            tower_data = dict(tower)
            performance_data = dict(performance)
            incident_data = [dict(row) for row in incidents]

            # ---------------------------------------------------------------
            # Diagnostic findings
            # ---------------------------------------------------------------

            findings = []
            recommendations = []

            technology = tower_data["technology"]
            tower_status = tower_data["status"]

            signal = performance_data["signal_strength_dbm"]
            packet_loss = performance_data["packet_loss_pct"]
            latency = performance_data["latency_ms"]
            downlink = performance_data["downlink_throughput_mbps"]

            # ---------------------------------------------------------------
            # Site-down condition
            #
            # From 01_schema.sql:
            # status OFFLINE or packet_loss_pct = 100
            # means the site is down.
            # ---------------------------------------------------------------

            if tower_status == "OFFLINE" or packet_loss == 100:
                findings.append(
                    "The site is down because the tower is OFFLINE "
                    "or packet loss is 100%."
                )

                recommendations.append(
                    "Escalate to the network operations team. "
                    "Do not troubleshoot the customer handset first."
                )

            # ---------------------------------------------------------------
            # Signal strength
            # ---------------------------------------------------------------

            if signal < -110:
                findings.append(
                    f"Signal strength is poor at {signal} dBm."
                )

                recommendations.append(
                    "Investigate radio-access conditions and coverage "
                    "quality around the tower."
                )

            elif signal < -90:
                findings.append(
                    f"Signal strength is marginal at {signal} dBm."
                )

                recommendations.append(
                    "Investigate possible radio-access degradation "
                    "or weak coverage."
                )

            else:
                findings.append(
                    f"Signal strength is acceptable at {signal} dBm."
                )

            # ---------------------------------------------------------------
            # Packet loss
            # ---------------------------------------------------------------

            if packet_loss > 5:
                findings.append(
                    f"Packet loss is severe at {packet_loss}%."
                )

                recommendations.append(
                    "Investigate backhaul or network connectivity "
                    "issues causing severe packet loss."
                )

            elif packet_loss > 2:
                findings.append(
                    f"Packet loss is elevated at {packet_loss}%."
                )

                recommendations.append(
                    "Investigate network connectivity and backhaul "
                    "performance."
                )

            else:
                findings.append(
                    f"Packet loss is within normal range at "
                    f"{packet_loss}%."
                )

            # ---------------------------------------------------------------
            # Latency
            # ---------------------------------------------------------------

            if latency > 50:
                findings.append(
                    f"Latency is elevated at {latency} ms."
                )

                recommendations.append(
                    "Investigate network congestion, backhaul, "
                    "or other transport-related issues."
                )

            elif technology == "5G" and latency <= 40:
                findings.append(
                    f"Latency is normal for 5G at {latency} ms."
                )

            # ---------------------------------------------------------------
            # Downlink throughput
            #
            # Degraded when:
            # technology is 5G
            # status is OPERATIONAL
            # downlink < 100 Mbps
            # ---------------------------------------------------------------

            if (
                technology == "5G"
                and tower_status == "OPERATIONAL"
                and downlink < 100
            ):
                findings.append(
                    f"5G downlink throughput is degraded at "
                    f"{downlink} Mbps."
                )

                recommendations.append(
                    "Investigate possible 5G radio-access or "
                    "capacity degradation."
                )

            # ---------------------------------------------------------------
            # Active incidents
            # ---------------------------------------------------------------

            if incident_data:
                findings.append(
                    f"{len(incident_data)} active incident(s) "
                    f"are associated with this tower."
                )

                for incident in incident_data:
                    recommendations.append(
                        f"Review incident {incident['incident_id']} "
                        f"({incident['severity']}, "
                        f"{incident['classification']}) "
                        f"assigned to {incident['assigned_team']}."
                    )

            # ---------------------------------------------------------------
            # No major findings
            # ---------------------------------------------------------------

            if not recommendations:
                recommendations.append(
                    "No major tower-side network issue was identified "
                    "from the latest available performance data."
                )

            return {
                "success": True,
                "tower_id": tower_id,
                "symptom": symptom,
                "tower": tower_data,
                "latest_performance": performance_data,
                "open_incidents": incident_data,
                "findings": findings,
                "recommendations": recommendations,
            }

    except sqlite3.Error as exc:
        logger.exception(
            "Database error during diagnostics for tower %s",
            tower_id,
        )

        return {
            "success": False,
            "tower_id": tower_id,
            "symptom": symptom,
            "error": f"Database error: {exc}",
        }


# ---------------------------------------------------------------------------
# Tool 3: Regional network summary
# ---------------------------------------------------------------------------

async def get_regional_network_summary(
    region: str,
) -> dict[str, Any]:
    """
    Summarize tower health and status for a network region.

    Valid regions:
        Midwest
        Northeast
        Southeast
        Southwest
        West

    Args:
        region: Network region.

    Returns:
        Regional tower counts and status summary.
    """

    logger.info(
        "Generating regional network summary for region=%s",
        region,
    )

    try:
        with _get_connection() as conn:

            # ---------------------------------------------------------------
            # Get towers in region
            # ---------------------------------------------------------------

            towers = conn.execute(
                """
                SELECT
                    tower_id,
                    tower_name,
                    city,
                    state,
                    technology,
                    status
                FROM network_towers
                WHERE LOWER(region) = LOWER(?)
                ORDER BY tower_id
                """,
                (region,),
            ).fetchall()

            if not towers:
                return {
                    "success": False,
                    "region": region,
                    "error": (
                        f"No network towers were found in region "
                        f"'{region}'."
                    ),
                }

            # ---------------------------------------------------------------
            # Status counts
            # ---------------------------------------------------------------

            status_counts = conn.execute(
                """
                SELECT
                    status,
                    COUNT(*) AS tower_count
                FROM network_towers
                WHERE LOWER(region) = LOWER(?)
                GROUP BY status
                ORDER BY status
                """,
                (region,),
            ).fetchall()

            # ---------------------------------------------------------------
            # Technology counts
            # ---------------------------------------------------------------

            technology_counts = conn.execute(
                """
                SELECT
                    technology,
                    COUNT(*) AS tower_count
                FROM network_towers
                WHERE LOWER(region) = LOWER(?)
                GROUP BY technology
                ORDER BY technology
                """,
                (region,),
            ).fetchall()

            # ---------------------------------------------------------------
            # Active incidents in region
            # ---------------------------------------------------------------

            incident_count = conn.execute(
                """
                SELECT COUNT(*)
                FROM open_incidents oi
                JOIN network_towers nt
                    ON oi.tower_id = nt.tower_id
                WHERE LOWER(nt.region) = LOWER(?)
                """,
                (region,),
            ).fetchone()[0]

            return {
                "success": True,
                "region": region,
                "total_towers": len(towers),
                "status_summary": {
                    row["status"]: row["tower_count"]
                    for row in status_counts
                },
                "technology_summary": {
                    row["technology"]: row["tower_count"]
                    for row in technology_counts
                },
                "active_incident_count": incident_count,
                "towers": [dict(row) for row in towers],
            }

    except sqlite3.Error as exc:
        logger.exception(
            "Database error while generating regional summary "
            "for region %s",
            region,
        )

        return {
            "success": False,
            "region": region,
            "error": f"Database error: {exc}",
        }


# ---------------------------------------------------------------------------
# ADK Agent
# ---------------------------------------------------------------------------

network_diagnostics_agent = Agent(
    name="network_diagnostics_agent",
    model=Gemini(
    model="gemini-3.6-flash",
    client_kwargs={
        "http_options": types.HttpOptions(
            async_client_args={"ssl": False}
        )
    },
),
    description=(
        "Diagnoses telecom network and tower connectivity problems "
        "using live operational data from the SQLite database."
    ),
    instruction="""
You are a telecom network diagnostics agent.

Your job is to diagnose tower and connectivity problems using the
available SQL-backed tools.

Use the tools instead of inventing network information.

For a specific tower:
1. Use check_tower_status to retrieve tower information,
   latest performance metrics, and active incidents.
2. Use run_connectivity_diagnostics when a customer reports
   a connectivity symptom.
3. Explain the likely network-side cause using the returned
   diagnostic findings.
4. Provide clear operational recommendations.

For regional questions:
- Use get_regional_network_summary.
- Valid network regions are Midwest, Northeast, Southeast,
  Southwest, and West.

Do not treat a city such as Austin as a region.
If the user gives a tower ID, use the tower ID to identify
its city and region.

Do not invent measurements, incidents, or tower status.
""",
    tools=[
        check_tower_status,
        run_connectivity_diagnostics,
        get_regional_network_summary,
    ],
)


# ---------------------------------------------------------------------------
# A2A application
# ---------------------------------------------------------------------------

a2a_app = to_a2a(
    network_diagnostics_agent,
    host="localhost",
    port=8001,
)


# ---------------------------------------------------------------------------
# Local test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    import uvicorn

    async def test_tools():
        print("\n--- check_tower_status ---")
        result = await check_tower_status("TX-512")
        print(result)

        print("\n--- run_connectivity_diagnostics ---")
        result = await run_connectivity_diagnostics(
            "TX-512",
            "My 5G keeps dropping",
        )
        print(result)

        print("\n--- get_regional_network_summary ---")
        result = await get_regional_network_summary("Southwest")
        print(result)

    # asyncio.run(test_tools())

    print("\nStarting Network Diagnostics A2A service on port 8001...")

    uvicorn.run(
        a2a_app,
        host="localhost",
        port=8001,
    )