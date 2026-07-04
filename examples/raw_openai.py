"""A minimal openai agent, fully traced with three added lines.

Needs OPENAI_API_KEY. Total cost of one run is well under a cent.

    python examples/raw_openai.py
    traceburn ls
"""

import json

import traceburn
from traceburn import session, span

traceburn.install()

from openai import OpenAI

client = OpenAI()
MODEL = "gpt-5-mini"


def check_weather(city: str) -> str:
    with span("check_weather", kind="tool"):
        return json.dumps({"city": city, "forecast": "sunny", "high_f": 78})


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_weather",
            "description": "Get the weather forecast for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def run(question: str) -> str:
    messages = [{"role": "user", "content": question}]
    first = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
    message = first.choices[0].message
    if not message.tool_calls:
        return message.content
    messages.append(message)
    for call in message.tool_calls:
        result = check_weather(**json.loads(call.function.arguments))
        messages.append(
            {"role": "tool", "tool_call_id": call.id, "content": result}
        )
    second = client.chat.completions.create(model=MODEL, messages=messages)
    return second.choices[0].message.content


if __name__ == "__main__":
    with session("weather-agent"):
        with span("weather-agent", kind="agent"):
            answer = run("Should I bike to work in San Diego tomorrow?")
    print(answer)
    print("\nrecorded into ./.traceburn/traces.db; next: traceburn ls")
