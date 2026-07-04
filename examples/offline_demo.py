"""Record a demo trace with no API keys and no network.

This simulates a small research agent using the explicit traceburn API, so
you can see a full trace, token counts, and cost estimates in under a
minute. Run it, then open the viewer:

    python examples/offline_demo.py
    traceburn ls
    traceburn show <trace-id>

The token counts below are made up but realistic. Real instrumentation
(``traceburn.install()``) records real usage; see raw_openai.py.
"""

import random
import time

from traceburn import session, span, trace


def fake_llm_call(name, model, input_tokens, output_tokens, cached_input_tokens=0, prompt=None):
    prompt = prompt or f"({name} prompt)"
    attributes = {
        "gen_ai.system": "openai",
        "gen_ai.request.model": model,
        "gen_ai.response.model": model,
        "gen_ai.usage.input_tokens": input_tokens,
        "gen_ai.usage.output_tokens": output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "request": {"messages": [{"role": "user", "content": prompt}]},
        "response": {"text": f"({name} response)"},
        "request_hash": f"demo-{hash(prompt) & 0xFFFFFFFF:08x}",
        "finish_reason": "stop",
        "stream": False,
    }
    with span(f"chat {model}", kind="llm", attributes=attributes):
        time.sleep(input_tokens / 500_000 + output_tokens / 50_000)


@trace("research-agent", kind="agent")
def run_agent(question):
    fake_llm_call("plan", "gpt-5-mini", 1_850, 310)
    with span("search-web", kind="tool"):
        time.sleep(random.uniform(0.08, 0.15))
    with span("fetch-pages", kind="tool"):
        time.sleep(random.uniform(0.2, 0.4))
    with span("retrieve-chunks", kind="retrieval"):
        time.sleep(random.uniform(0.03, 0.08))
    # A classic agent bug, on purpose: page 1 gets summarized twice, so the
    # waste report has something real to show you.
    for page in (1, 1, 2):
        fake_llm_call(
            f"summarize-page-{page}", "gpt-5-mini", 6_400, 400,
            prompt=f"Summarize the following page:\n(page {page} contents)",
        )
    fake_llm_call("synthesize", "gpt-5", 9_200, 880, cached_input_tokens=4_100)
    fake_llm_call("format-title", "gpt-5", 140, 12)


if __name__ == "__main__":
    with session("offline-demo"):
        run_agent("What changed in agent observability this year?")
    print("recorded one trace into ./.traceburn/traces.db")
    print("next: traceburn ls")
