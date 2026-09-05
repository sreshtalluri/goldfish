"""OpenAICompatibleAdapter's tool-call-vs-text branch is real logic (it
decides whether an episode records a tool action or a probe answer), so it
gets an offline check like every other branch in this project, using plain
namespace objects instead of a real key or network call.
"""
import json
from types import SimpleNamespace

from goldfish.models import _parse_openai_response


def _response(message, prompt_tokens=10, completion_tokens=5, cached_tokens=0):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        ),
    )


def test_parses_tool_call():
    call = SimpleNamespace(function=SimpleNamespace(name="post_transaction", arguments=json.dumps({"tx_id": "TX-001"})))
    resp = _response(SimpleNamespace(content=None, tool_calls=[call]), cached_tokens=4)
    action = _parse_openai_response(resp)
    assert action.kind == "tool"
    assert action.name == "post_transaction"
    assert action.args == {"tx_id": "TX-001"}
    assert action.usage == {"input": 10, "output": 5, "cache_read": 4, "cache_write": 0}


def test_parses_text_when_no_tool_call():
    resp = _response(SimpleNamespace(content="I no longer have that.", tool_calls=None))
    action = _parse_openai_response(resp)
    assert action.kind == "text"
    assert action.content == "I no longer have that."


def test_handles_missing_prompt_tokens_details():
    """Not every OpenAI-compatible provider (e.g. Groq) reports cache stats."""
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, prompt_tokens_details=None),
    )
    action = _parse_openai_response(resp)
    assert action.usage["cache_read"] == 0
