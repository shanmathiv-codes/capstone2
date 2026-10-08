import asyncio

from agent import (
    lookup_billing_account,
    check_duplicate_charges,
    apply_billing_credit,
)


async def main():

    print("\n--- LOOKUP ACCOUNT ---")
    result = await lookup_billing_account("CUST-10002")
    print(result)

    print("\n--- CHECK DUPLICATES ---")
    result = await check_duplicate_charges("CUST-10002")
    print(result)

    # print("\n--- APPLY CREDIT ---")
    # result = await apply_billing_credit(
    #     "CUST-10002",
    #     25.00,
    #     "Duplicate Unlimited Plus charge",
    # )
    # print(result)


if __name__ == "__main__":
    asyncio.run(main())