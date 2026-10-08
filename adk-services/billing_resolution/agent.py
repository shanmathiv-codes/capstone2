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
# Billing policy
# ---------------------------------------------------------------------------

AUTO_APPROVAL_LIMIT = 50.00


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

    # Return rows as dictionary-like sqlite3.Row objects.
    connection.row_factory = sqlite3.Row

    # Enable foreign-key enforcement.
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


# ---------------------------------------------------------------------------
# Tool 1: Lookup billing account
# ---------------------------------------------------------------------------

async def lookup_billing_account(
    customer_id: str,
) -> dict[str, Any]:
    """
    Retrieve billing account information, current balance,
    and recent charges for a customer.

    Args:
        customer_id: Customer identifier such as CUST-10002.

    Returns:
        Billing account details and recent charges.
    """

    logger.info(
        "Looking up billing account for customer_id=%s",
        customer_id,
    )

    try:
        with _get_connection() as conn:

            # ---------------------------------------------------------------
            # Billing account
            # ---------------------------------------------------------------

            account = conn.execute(
                """
                SELECT
                    customer_id,
                    customer_name,
                    account_type,
                    current_balance,
                    currency,
                    service_region,
                    city,
                    state,
                    billing_cycle,
                    auto_pay_enabled,
                    account_status,
                    last_updated
                FROM billing_accounts
                WHERE customer_id = ?
                """,
                (customer_id,),
            ).fetchone()

            if account is None:
                return {
                    "success": False,
                    "customer_id": customer_id,
                    "error": (
                        f"Billing account for customer "
                        f"'{customer_id}' was not found."
                    ),
                }

            # ---------------------------------------------------------------
            # Recent charges
            #
            # The schema does not define a separate "recent" flag.
            # We therefore retrieve charges ordered by charge_date.
            # ---------------------------------------------------------------

            charges = conn.execute(
                """
                SELECT
                    charge_id,
                    description,
                    amount,
                    billing_period,
                    charge_date,
                    charge_type,
                    is_duplicate_flag,
                    invoice_status
                FROM billing_charges
                WHERE customer_id = ?
                ORDER BY charge_date DESC, charge_id DESC
                LIMIT 10
                """,
                (customer_id,),
            ).fetchall()

            account_data = dict(account)
            charge_data = [dict(row) for row in charges]

            logger.info(
                "Billing account retrieved successfully for customer_id=%s",
                customer_id,
            )

            return {
                "success": True,
                "customer_id": customer_id,
                "account": account_data,
                "recent_charges": charge_data,
            }

    except sqlite3.Error as exc:
        logger.exception(
            "Database error while looking up billing account %s",
            customer_id,
        )

        return {
            "success": False,
            "customer_id": customer_id,
            "error": f"Database error: {exc}",
        }


# ---------------------------------------------------------------------------
# Tool 2: Check duplicate charges
# ---------------------------------------------------------------------------

async def check_duplicate_charges(
    customer_id: str,
) -> dict[str, Any]:
    """
    Find duplicate or potentially duplicate charges for a customer.

    A charge is explicitly flagged when is_duplicate_flag = 1.

    The tool also identifies repeated charge descriptions for the same
    customer and billing period.

    Args:
        customer_id: Customer identifier.

    Returns:
        Duplicate and potentially duplicate charge information.
    """

    logger.info(
        "Checking duplicate charges for customer_id=%s",
        customer_id,
    )

    try:
        with _get_connection() as conn:

            # ---------------------------------------------------------------
            # Verify customer exists
            # ---------------------------------------------------------------

            account = conn.execute(
                """
                SELECT
                    customer_id,
                    customer_name,
                    current_balance,
                    currency
                FROM billing_accounts
                WHERE customer_id = ?
                """,
                (customer_id,),
            ).fetchone()

            if account is None:
                return {
                    "success": False,
                    "customer_id": customer_id,
                    "error": (
                        f"Billing account for customer "
                        f"'{customer_id}' was not found."
                    ),
                }

            # ---------------------------------------------------------------
            # Explicitly flagged duplicate charges
            # ---------------------------------------------------------------

            flagged_duplicates = conn.execute(
                """
                SELECT
                    charge_id,
                    description,
                    amount,
                    billing_period,
                    charge_date,
                    charge_type,
                    is_duplicate_flag,
                    invoice_status
                FROM billing_charges
                WHERE customer_id = ?
                  AND is_duplicate_flag = 1
                ORDER BY charge_date DESC, charge_id DESC
                """,
                (customer_id,),
            ).fetchall()

            # ---------------------------------------------------------------
            # Repeated descriptions within the same billing period
                #
                # These are relevant duplicate candidates even if the
                # explicit duplicate flag is not set.
                # ---------------------------------------------------------------

            repeated_groups = conn.execute(
                """
                SELECT
                    description,
                    billing_period,
                    COUNT(*) AS charge_count,
                    SUM(amount) AS total_amount
                FROM billing_charges
                WHERE customer_id = ?
                GROUP BY description, billing_period
                HAVING COUNT(*) > 1
                ORDER BY billing_period DESC, description
                """,
                (customer_id,),
            ).fetchall()

            # ---------------------------------------------------------------
            # Retrieve the actual records belonging to repeated groups
            # ---------------------------------------------------------------

            potential_duplicates = []

            for group in repeated_groups:
                records = conn.execute(
                    """
                    SELECT
                        charge_id,
                        description,
                        amount,
                        billing_period,
                        charge_date,
                        charge_type,
                        is_duplicate_flag,
                        invoice_status
                    FROM billing_charges
                    WHERE customer_id = ?
                      AND description = ?
                      AND billing_period = ?
                    ORDER BY charge_date ASC, charge_id ASC
                    """,
                    (
                        customer_id,
                        group["description"],
                        group["billing_period"],
                    ),
                ).fetchall()

                potential_duplicates.append(
                    {
                        "description": group["description"],
                        "billing_period": group["billing_period"],
                        "charge_count": group["charge_count"],
                        "total_amount": group["total_amount"],
                        "charges": [dict(row) for row in records],
                    }
                )

            flagged_data = [dict(row) for row in flagged_duplicates]

            return {
                "success": True,
                "customer_id": customer_id,
                "account": dict(account),
                "flagged_duplicate_count": len(flagged_data),
                "flagged_duplicates": flagged_data,
                "potential_duplicate_groups": potential_duplicates,
            }

    except sqlite3.Error as exc:
        logger.exception(
            "Database error while checking duplicate charges for %s",
            customer_id,
        )

        return {
            "success": False,
            "customer_id": customer_id,
            "error": f"Database error: {exc}",
        }


# ---------------------------------------------------------------------------
# Tool 3: Apply billing credit
# ---------------------------------------------------------------------------

async def apply_billing_credit(
    customer_id: str,
    amount: float,
    reason: str,
) -> dict[str, Any]:
    """
    Apply or request a billing credit according to the billing policy.

    Policy:
        amount <= $50.00
            -> billing_credits.status = APPLIED
            -> billing_accounts.current_balance is reduced

        amount > $50.00
            -> billing_credits.status = PENDING_APPROVAL
            -> current_balance remains unchanged

    This function performs a real SQLite transaction.

    Args:
        customer_id: Customer identifier.
        amount: Credit amount in USD.
        reason: Reason for the credit.

    Returns:
        Result of the billing-credit operation.
    """

    logger.info(
        "Applying billing credit: customer=%s amount=%.2f",
        customer_id,
        amount,
    )

    # -----------------------------------------------------------------------
    # Validate input before opening the transaction
    # -----------------------------------------------------------------------

    if amount <= 0:
        return {
            "success": False,
            "customer_id": customer_id,
            "amount": amount,
            "error": "Credit amount must be greater than 0.",
        }

    if not reason or not reason.strip():
        return {
            "success": False,
            "customer_id": customer_id,
            "amount": amount,
            "error": "Credit reason cannot be empty.",
        }

    # Normalize floating-point value to two decimal places for currency.
    amount = round(float(amount), 2)

    try:
        with _get_connection() as conn:

            # ---------------------------------------------------------------
            # Start an explicit transaction.
            # ---------------------------------------------------------------

            conn.execute("BEGIN")

            # ---------------------------------------------------------------
            # Verify billing account exists and retrieve current balance.
            # ---------------------------------------------------------------

            account = conn.execute(
                """
                SELECT
                    customer_id,
                    customer_name,
                    current_balance,
                    currency,
                    account_status
                FROM billing_accounts
                WHERE customer_id = ?
                """,
                (customer_id,),
            ).fetchone()

            if account is None:
                conn.rollback()

                return {
                    "success": False,
                    "customer_id": customer_id,
                    "amount": amount,
                    "error": (
                        f"Billing account for customer "
                        f"'{customer_id}' was not found."
                    ),
                }

            current_balance = float(account["current_balance"])

            # ---------------------------------------------------------------
            # Determine credit status using the policy limit.
            # ---------------------------------------------------------------

            if amount <= AUTO_APPROVAL_LIMIT:
                credit_status = "APPLIED"

                new_balance = round(
                    current_balance - amount,
                    2,
                )

                # Prevent an invalid negative account balance.
                if new_balance < 0:
                    conn.rollback()

                    return {
                        "success": False,
                        "customer_id": customer_id,
                        "amount": amount,
                        "error": (
                            f"Credit amount ${amount:.2f} exceeds "
                            f"the current balance of "
                            f"${current_balance:.2f}."
                        ),
                    }

            else:
                credit_status = "PENDING_APPROVAL"

                # Balance does not change until approval.
                new_balance = current_balance

            # ---------------------------------------------------------------
            # Insert billing credit
            # ---------------------------------------------------------------

            cursor = conn.execute(
                """
                INSERT INTO billing_credits (
                    customer_id,
                    amount,
                    reason,
                    status
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    customer_id,
                    amount,
                    reason.strip(),
                    credit_status,
                ),
            )

            credit_id = cursor.lastrowid

            # ---------------------------------------------------------------
            # Only APPLIED credits modify current_balance.
            # ---------------------------------------------------------------

            if credit_status == "APPLIED":

                conn.execute(
                    """
                    UPDATE billing_accounts
                    SET
                        current_balance = ?,
                        last_updated = datetime('now')
                    WHERE customer_id = ?
                    """,
                    (
                        new_balance,
                        customer_id,
                    ),
                )

            # ---------------------------------------------------------------
            # Commit the transaction.
            # ---------------------------------------------------------------

            conn.commit()

            logger.info(
                "Billing credit transaction committed: "
                "customer=%s credit_id=%s status=%s",
                customer_id,
                credit_id,
                credit_status,
            )

            return {
                "success": True,
                "customer_id": customer_id,
                "credit_id": credit_id,
                "amount": amount,
                "status": credit_status,
                "reason": reason.strip(),
                "previous_balance": current_balance,
                "new_balance": new_balance,
                "auto_approval_limit": AUTO_APPROVAL_LIMIT,
                "balance_updated": credit_status == "APPLIED",
            }

    except sqlite3.Error as exc:
        logger.exception(
            "Database error while applying billing credit "
            "for customer %s",
            customer_id,
        )

        return {
            "success": False,
            "customer_id": customer_id,
            "amount": amount,
            "error": f"Database error: {exc}",
        }

    except Exception as exc:
        logger.exception(
            "Unexpected error while applying billing credit "
            "for customer %s",
            customer_id,
        )

        return {
            "success": False,
            "customer_id": customer_id,
            "amount": amount,
            "error": f"Unexpected error: {exc}",
        }


# ---------------------------------------------------------------------------
# ADK Agent
# ---------------------------------------------------------------------------

billing_resolution_agent = Agent(
    name="billing_resolution_agent",
    model=Gemini(
    model="gemini-3.6-flash",
    client_kwargs={
        "http_options": types.HttpOptions(
            async_client_args={"ssl": False}
        )
    },
),
    description=(
        "Investigates telecom billing disputes and performs "
        "billing-credit actions using the operational SQLite database."
    ),
    instruction="""
You are a telecom billing resolution agent.

Your job is to investigate customer billing disputes and, when
appropriate, perform billing-credit actions using the available tools.

Always use the database-backed tools instead of inventing account,
charge, duplicate, or balance information.

For a billing dispute:

1. Use lookup_billing_account to retrieve the customer's account,
   current balance, and recent charges.

2. Use check_duplicate_charges when the customer reports duplicate
   or potentially duplicate charges.

3. Determine the appropriate credit amount from the actual billing
   information returned by the tools.

4. When a credit action is requested and justified, use
   apply_billing_credit.

Billing credit policy:

- Credits of $50.00 or less are automatically applied.
- Credits above $50.00 must have status PENDING_APPROVAL.
- An APPLIED credit reduces billing_accounts.current_balance.
- A PENDING_APPROVAL credit does not reduce current_balance.

Do not claim that a credit was applied unless the
apply_billing_credit tool reports success with status APPLIED.

Do not claim that the balance changed unless the tool confirms
the updated balance.

For credits above $50.00, clearly state that the credit is pending
approval and that the current balance was not changed.

Never simulate billing changes in memory.
All billing actions must be performed through the database tool.
""",
    tools=[
        lookup_billing_account,
        check_duplicate_charges,
        apply_billing_credit,
    ],
)


# ---------------------------------------------------------------------------
# A2A application
# ---------------------------------------------------------------------------

a2a_app = to_a2a(
    billing_resolution_agent,
    host="localhost",
    port=8002,
)


# ---------------------------------------------------------------------------
# Service entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    print(
        "Starting Billing Resolution A2A service on port 8002..."
    )

    uvicorn.run(
        a2a_app,
        host="localhost",
        port=8002,
    )