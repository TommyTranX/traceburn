"""Reproduce the cache before-and-after finding on a realistic agent.

A support-ticket triage loop over the anthropic SDK: a large static system
prompt (policy manual, worked examples) plus a per-ticket user message,
with one tool round per ticket. Needs ANTHROPIC_API_KEY. Two variants:

    python examples/cache_before_after.py uncached
    python examples/cache_before_after.py cached
    traceburn waste <uncached-trace>     # flags the uncached prefix
    traceburn diff <uncached> <cached>   # the measured saving

Measured on 2026-07-05 with claude-haiku-4-5: the uncached run cost
$0.0539, the cached run $0.0167, a 69 percent saving from adding one
cache_control block, exactly what the waste report suggested. Prices
change; rerun it yourself. This is a paid API demo, with at most three
requests per ticket by default and SDK retries disabled. This limits calls,
not dollars; check your provider pricing before running it.

Everything here is generic and synthetic; the ONLY network calls are to
the Anthropic API via the instrumented SDK.
"""

import argparse
import json

import traceburn
from traceburn import session, span

MODEL = "claude-haiku-4-5"

# -- the static prefix: policy manual + few-shot examples (~4,700 tokens) ----

POLICY_SECTIONS = []
AREAS = [
    "billing", "authentication", "data export", "API rate limits", "webhooks",
    "single sign-on", "mobile app", "notifications", "user permissions",
    "integrations", "audit logs", "backups", "custom domains",
    "sandbox environments", "invoicing", "usage analytics", "team onboarding",
    "password resets", "regional data residency", "scheduled reports",
]
for i, area in enumerate(AREAS, 1):
    POLICY_SECTIONS.append(
        f"Section {i}: {area.title()}. Tickets about {area} are handled by the "
        f"{area} rotation. First verify the reporter's plan tier, because free-tier "
        f"limits differ from paid tiers in throughput, retention, and support "
        f"response time. If the ticket describes an outage or data loss, escalate "
        f"immediately with severity high regardless of plan. If the ticket asks for "
        f"a feature that already exists, respond with category question and include "
        f"the documentation path. If the behavior contradicts documented behavior, "
        f"category is bug and severity follows the user impact table: single user "
        f"inconvenienced is low, team blocked is medium, org-wide failure or any "
        f"security concern is high. Feature requests are category feature_request "
        f"with severity low unless a paying customer reports churn risk, which is "
        f"medium. Always fill the affected_area field with '{area}' when this "
        f"section applies. When two sections apply, pick the one naming the failing "
        f"component rather than the surface where it was noticed. Never guess "
        f"account state; use the lookup tool when plan tier matters."
    )

FEW_SHOT = """
Worked examples:
Ticket: "Exports have been stuck at 0% for two days, whole team blocked."
Decision: {"category": "bug", "severity": "medium", "affected_area": "data export", "needs_lookup": true}
Ticket: "Can you add dark mode to the dashboard?"
Decision: {"category": "feature_request", "severity": "low", "affected_area": "mobile app", "needs_lookup": false}
Ticket: "We were charged twice this month."
Decision: {"category": "billing_issue", "severity": "medium", "affected_area": "billing", "needs_lookup": true}
"""

SYSTEM_PROMPT = (
    "You are a support ticket triage assistant. Classify each ticket into a "
    "category (bug, feature_request, billing_issue, question), a severity "
    "(low, medium, high), and an affected_area, following the policy manual "
    "below exactly. Use the lookup_account tool when plan tier matters. "
    "Respond with a single JSON object and nothing else.\n\n"
    + "\n\n".join(POLICY_SECTIONS)
    + "\n" + FEW_SHOT
)

TOOLS = [
    {
        "name": "lookup_account",
        "description": "Look up a customer's plan tier and account standing.",
        "input_schema": {
            "type": "object",
            "properties": {"email": {"type": "string"}},
            "required": ["email"],
        },
    }
]

TICKETS = [
    ("t1", "amy@example.com", "The webhook retries stopped firing after Tuesday's maintenance window. Our whole order pipeline is down."),
    ("t2", "raj@example.com", "How do I rotate my API keys without downtime?"),
    ("t3", "kim@example.com", "We were invoiced for 12 seats but only have 9 active users."),
    ("t4", "lee@example.com", "Please add support for exporting audit logs as CSV."),
    ("t5", "sam@example.com", "SSO login loops back to the sign-in page for everyone in our org since this morning."),
]


def lookup_account(email: str) -> str:
    with span("lookup_account", kind="tool", attributes={"request": {"email": email}}):
        return json.dumps({"email": email, "plan": "team", "standing": "good"})


def system_blocks(cached: bool):
    block = {"type": "text", "text": SYSTEM_PROMPT}
    if cached:
        block["cache_control"] = {"type": "ephemeral"}
    return [block]


def triage(ticket_id: str, email: str, text: str, cached: bool, *,
           client, max_tool_rounds: int = 2) -> dict:
    if max_tool_rounds < 0:
        raise ValueError("max_tool_rounds must be nonnegative")
    with span(f"triage-{ticket_id}", kind="agent"):
        messages = [{"role": "user", "content": f"Ticket from {email}: {text}"}]
        response = client.messages.create(
            model=MODEL,
            max_tokens=300,
            system=system_blocks(cached),
            tools=TOOLS,
            messages=messages,
        )
        tool_rounds = 0
        while response.stop_reason == "tool_use":
            if tool_rounds >= max_tool_rounds:
                raise RuntimeError(
                    f"ticket {ticket_id} exceeded {max_tool_rounds} tool rounds; "
                    "stopping before another API request"
                )
            tool_rounds += 1
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                raise RuntimeError("provider requested tool use without a tool call")
            results = []
            for use in tool_uses:
                if use.name != "lookup_account":
                    raise RuntimeError(f"unsupported tool: {use.name}")
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": use.id,
                        "content": lookup_account(**use.input),
                    }
                )
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": results})
            response = client.messages.create(
                model=MODEL,
                max_tokens=300,
                system=system_blocks(cached),
                tools=TOOLS,
                messages=messages,
            )
        text_out = "".join(b.text for b in response.content if b.type == "text")
        try:
            return json.loads(text_out)
        except json.JSONDecodeError:
            return {"raw": text_out}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("uncached", "cached"), nargs="?", default="uncached")
    args = parser.parse_args(argv)
    # Parse before importing or constructing the client, so --help and invalid
    # arguments work without an SDK installation or API key.
    from anthropic import Anthropic

    traceburn.install()
    with Anthropic(max_retries=0) as client:
        cached = args.variant == "cached"
        with session(f"triage-{args.variant}"):
            with span(f"ticket-triage-{args.variant}", kind="agent") as root:
                for ticket_id, email, text in TICKETS:
                    decision = triage(ticket_id, email, text, cached, client=client)
                    print(f"{ticket_id}: {json.dumps(decision)}")
    print(f"\nvariant={args.variant} trace={root.span.trace_id}")


if __name__ == "__main__":
    main()
