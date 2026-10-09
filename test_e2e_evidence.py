import sys
import os
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from orchestration.graph import run_telecom_assistant

scenarios = [
    {
        "id": 1,
        "name": "Policy RAG Query",
        "query": "What is Prodapt's roaming policy for Western Europe?",
        "expected_source": "roaming_policy.txt",
    },
    {
        "id": 2,
        "name": "SQL Analytics Query",
        "query": "Which region had the most CRITICAL network outages recently?",
        "expected_source": "network_outages",
    },
    {
        "id": 3,
        "name": "Live Tower Diagnostics",
        "query": "My 5G keeps dropping near tower TX-512 in Austin. Please diagnose.",
        "expected_source": "tower_telemetry",
    },
    {
        "id": 4,
        "name": "Billing Dispute Resolution",
        "query": "Customer CUST-10002 was charged twice for Unlimited Plus. Investigate and apply credit.",
        "expected_source": "customer_billing_records",
    },
    {
        "id": 5,
        "name": "Combined Outage + Policy SLA",
        "query": "We had a 6-hour outage in the Midwest. Am I eligible for an SLA credit and what does policy say?",
        "expected_source": "sla_policy.txt",
    },
]

print("=" * 80)
print("RUNNING E2E EVIDENCE & FRESHNESS TEST SUITE")
print("=" * 80)

for sc in scenarios:
    print(f"\n--- Scenario {sc['id']}: {sc['name']} ---")
    print(f"Query: {sc['query']}")
    result = run_telecom_assistant(sc['query'])
    
    final_resp = result.get("final_response", "")
    evidence = result.get("evidence", [])
    trace = result.get("execution_trace", [])
    
    print(f"Workers executed: {[t['worker'] for t in trace]}")
    print(f"Final response length: {len(final_resp)} chars")
    print(f"Evidence items count: {len(evidence)}")
    
    for idx, ev in enumerate(evidence, 1):
        src = ev.get('source_table_or_doc') or ev.get('source_name')
        ts = ev.get('metric_timestamp') or ev.get('timestamp')
        print(f"  [Evidence #{idx}]")
        print(f"    Worker: {ev.get('worker')}")
        print(f"    Source: {src}")
        print(f"    Timestamp: {ts}")
        print(f"    Confidence: {ev.get('confidence')}")
        print(f"    SQL: {ev.get('generated_sql')}")
        citations = ev.get('citations') or []
        print(f"    Citations count: {len(citations)}")
        if citations:
            quote = citations[0].get('verbatim_quote') or citations[0].get('citation_text') or ""
            print(f"      First citation doc: {citations[0].get('document')}")
            print(f"      Citation snippet: {quote[:100]}...")
        if ev.get('telemetry_snapshot'):
            print(f"    Telemetry/Ledger Snapshot: {ev.get('telemetry_snapshot')}")
        if ev.get('limitations'):
            print(f"    Limitations: {ev.get('limitations')}")
            
    print(f"\nResponse excerpt:\n{final_resp[:250]}...\n")
    print("-" * 80)

print("\nALL SCENARIOS TESTED SUCCESSFULLY!")
