"""A minimal anthropic agent, fully traced with three added lines.

Needs ANTHROPIC_API_KEY. Total cost of one run is well under a cent.

    python examples/raw_anthropic.py
    traceburn ls
"""

import traceburn
from traceburn import session, span

traceburn.install()

from anthropic import Anthropic

client = Anthropic()
MODEL = "claude-haiku-4-5"


def summarize(notes: list[str]) -> str:
    with span("collect-notes", kind="tool"):
        joined = "\n".join(f"- {note}" for note in notes)
    message = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system="You summarize notes into two crisp sentences.",
        messages=[{"role": "user", "content": f"Summarize:\n{joined}"}],
    )
    return message.content[0].text


if __name__ == "__main__":
    notes = [
        "Standup moved to 9:30 starting next week.",
        "The staging deploy failed twice on Tuesday; root cause was a stale env var.",
        "New tracing tool trial approved for the platform team.",
    ]
    with session("notes-agent"):
        with span("notes-agent", kind="agent"):
            print(summarize(notes))
    print("\nrecorded into ./.traceburn/traces.db; next: traceburn ls")
