"""A dependency-free ReAct + function-calling agent runtime, written to be read.

Everything here uses the standard library on purpose: the point of this file is
that you can trace every step of the tool-calling loop in a debugger, then walk
into an interview and reproduce it on a whiteboard.

Run it:
    uv run --env-file .env python agent.py "列出当前目录，并计算 (12+8)*3"

Test it (offline, no API key needed):
    uv run python -m unittest discover -s tests -v
"""

from __future__ import annotations

import ast
import json
import operator
import os
import pathlib
import ssl
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

# --------------------------------------------------------------------------
# Local config: read `.env` next to this file so the project runs with no
# shell setup at all.
# --------------------------------------------------------------------------


def load_env_file(path=None) -> None:
    """Load `.env` into the process environment.

    A value from the file overrides the ambient environment on purpose: the
    file is the explicit project configuration, and it keeps `python agent.py`
    working even when a stray shell variable is polluted with junk.
    """
    env_path = path or pathlib.Path(__file__).with_name(".env")
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


load_env_file()
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_SYSTEM_PROMPT = (
    "You are a careful assistant. Use the provided tools when they help; "
    "otherwise answer directly. Never invent tool results."
)

# --------------------------------------------------------------------------
# Arithmetic helper: a safe evaluator, because eval() on model output is a
# remote-code-execution hole.
# --------------------------------------------------------------------------

_ALLOWED_BINARY_OPS: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_MAX_EXPONENT = 64


def safe_calc(expression: str) -> float:
    """Evaluate a pure arithmetic expression without using eval()."""
    node = ast.parse(expression, mode="eval").body

    def _eval(current: ast.AST) -> float:
        if isinstance(current, ast.Constant) and not isinstance(current.value, bool):
            if isinstance(current.value, (int, float)):
                return current.value
        elif isinstance(current, ast.BinOp) and type(current.op) in _ALLOWED_BINARY_OPS:
            left = _eval(current.left)
            right = _eval(current.right)
            if isinstance(current.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
                raise ValueError("exponent too large")
            return _ALLOWED_BINARY_OPS[type(current.op)](left, right)
        elif isinstance(current, ast.UnaryOp) and isinstance(current.op, (ast.UAdd, ast.USub)):
            value = _eval(current.operand)
            return value if isinstance(current.op, ast.UAdd) else -value
        raise ValueError(f"unsupported syntax: {ast.dump(current)}")

    return _eval(node)


# --------------------------------------------------------------------------
# Tool layer
# --------------------------------------------------------------------------

WORKSPACE = pathlib.Path(os.environ.get("AGENT_WORKSPACE", ".")).resolve()


@dataclass(frozen=True)
class Tool:
    """A callable the model may invoke. `parameters` must be a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[..., str]

    def to_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _resolve_workspace_path(raw_path: str) -> pathlib.Path:
    """Resolve a user-supplied path and refuse anything outside WORKSPACE."""
    candidate = pathlib.Path(raw_path)
    if not candidate.is_absolute():
        candidate = WORKSPACE / candidate
    candidate = candidate.resolve()
    if candidate != WORKSPACE and WORKSPACE not in candidate.parents:
        raise ValueError(f"path escapes the workspace: {raw_path}")
    return candidate


def _list_files(path: str = ".", pattern: str = "*") -> str:
    target = _resolve_workspace_path(path)
    if not target.is_dir():
        raise ValueError(f"not a directory: {path}")
    entries = sorted(target.glob(pattern))[:200]
    rendered = [
        str(entry.relative_to(WORKSPACE)) + ("/" if entry.is_dir() else "") for entry in entries
    ]
    return "\n".join(rendered) if rendered else "(empty)"


def _read_file(path: str, max_bytes: int = 4000) -> str:
    target = _resolve_workspace_path(path)
    if not target.is_file():
        raise ValueError(f"not a file: {path}")
    return target.read_bytes()[:max_bytes].decode("utf-8", errors="replace")


def default_tools() -> list[Tool]:
    return [
        Tool(
            name="calculator",
            description=(
                "Evaluate a pure arithmetic expression, for example '(3+4)*2**3'. "
                "Supports + - * / // % ** and parentheses. No variables, no functions."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression, e.g. '12*7+5'.",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
            run=lambda expression: str(safe_calc(expression)),
        ),
        Tool(
            name="list_files",
            description="List files inside the workspace directory.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory relative to the workspace."},
                    "pattern": {"type": "string", "description": "Glob pattern, defaults to '*'."},
                },
                "required": [],
                "additionalProperties": False,
            },
            run=_list_files,
        ),
        Tool(
            name="read_file",
            description="Read a UTF-8 text file inside the workspace, truncated to max_bytes.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the workspace."},
                    "max_bytes": {"type": "integer", "description": "Read limit, defaults to 4000."},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            run=_read_file,
        ),
    ]


def build_tool_specs(tools: Sequence[Tool]) -> list[dict[str, Any]]:
    return [tool.to_spec() for tool in tools]


def dispatch_tool(tools: Sequence[Tool], name: str, raw_arguments: str) -> str:
    """Run one tool call and ALWAYS return a string.

    Failures are returned as text rather than raised. That single decision is
    what lets the model read its own error and self-correct on the next step,
    instead of the whole run crashing on a typo in the arguments.
    """
    tool = next((item for item in tools if item.name == name), None)
    if tool is None:
        available = ", ".join(item.name for item in tools)
        return f"ERROR: unknown tool '{name}'. Available tools: {available}"

    if raw_arguments is None or not raw_arguments.strip():
        arguments: dict[str, Any] = {}
    else:
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as error:
            return (
                f"ERROR: arguments for '{name}' are not valid JSON ({error}). "
                f"Received: {raw_arguments[:200]}"
            )

    if not isinstance(arguments, dict):
        return f"ERROR: arguments for '{name}' must be a JSON object."

    try:
        result = tool.run(**arguments)
    except TypeError as error:
        return f"ERROR: bad arguments for '{name}': {error}"
    except Exception as error:  # noqa: BLE001 - intentionally surfaced to the model
        return f"ERROR: {type(error).__name__}: {error}"

    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


# --------------------------------------------------------------------------
# LLM transport: raw HTTP so there is no SDK magic in the way
# --------------------------------------------------------------------------


class ChatCompletionClient(Protocol):
    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return the raw OpenAI-compatible /chat/completions payload."""
        ...


class OpenAICompatibleClient:
    """Minimal client for any OpenAI-compatible endpoint (OpenAI, vLLM, DeepSeek, ...)."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        self.model = model or os.environ.get("MODEL", DEFAULT_MODEL)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (copy .env.example and fill it in)")

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout, context=ssl.create_default_context()
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"LLM request failed: HTTP {error.code} {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"LLM request failed: {error.reason}") from error


# --------------------------------------------------------------------------
# Context management
# --------------------------------------------------------------------------


def _message_size(messages: Sequence[dict[str, Any]]) -> int:
    return len(json.dumps(list(messages), ensure_ascii=False))


def _group_units(messages: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group messages so an assistant tool call is never split from its results.

    A `tool` message whose `tool_call_id` has no matching assistant message is
    rejected by the API, so history trimming has to operate on whole units.
    """
    units: list[list[dict[str, Any]]] = []
    for message in messages:
        if message.get("role") == "tool" and units and units[-1][0].get("role") == "assistant":
            units[-1].append(message)
        else:
            units.append([message])
    return units


def trim_messages(
    messages: Sequence[dict[str, Any]],
    *,
    max_chars: int = 24_000,
    keep_recent_units: int = 4,
) -> list[dict[str, Any]]:
    """Drop the oldest units until the transcript fits the character budget."""
    system = [message for message in messages if message.get("role") == "system"]
    units = _group_units([message for message in messages if message.get("role") != "system"])

    while len(units) > keep_recent_units:
        flattened = [message for unit in units for message in unit]
        if _message_size(system + flattened) <= max_chars:
            break
        units.pop(0)

    return system + [message for unit in units for message in unit]


# --------------------------------------------------------------------------
# The agent loop
# --------------------------------------------------------------------------


@dataclass
class Step:
    index: int
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    observation: str | None = None
    content: str | None = None


@dataclass
class Agent:
    client: ChatCompletionClient
    tools: list[Tool] = field(default_factory=default_tools)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_steps: int = 8
    max_context_chars: int = 24_000
    verbose: bool = True
    steps: list[Step] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.tools = list(self.tools)

    def run(self, user_input: str) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]
        specs = build_tool_specs(self.tools)
        seen_calls: set[tuple[str, str]] = set()

        for index in range(1, self.max_steps + 1):
            messages = trim_messages(messages, max_chars=self.max_context_chars)
            completion = self.client.chat(messages, tools=specs)
            message = completion["choices"][0]["message"]
            calls = message.get("tool_calls") or []

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": message.get("content"),
            }
            if calls:
                assistant_message["tool_calls"] = calls
            messages.append(assistant_message)

            if not calls:
                final = message.get("content") or ""
                self._record(Step(index=index, content=final))
                return final

            for call in calls:
                name = call["function"]["name"]
                raw_arguments = call["function"].get("arguments") or "{}"
                signature = (name, raw_arguments)

                try:
                    parsed_arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    parsed_arguments = {}

                if signature in seen_calls:
                    observation = (
                        f"ERROR: you already called '{name}' with these exact arguments. "
                        "Do not repeat it - use the earlier observation or try a different approach."
                    )
                else:
                    seen_calls.add(signature)
                    observation = dispatch_tool(self.tools, name, raw_arguments)

                messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": observation}
                )
                self._record(
                    Step(index=index, tool=name, arguments=parsed_arguments, observation=observation)
                )

        final = f"ERROR: stopped after {self.max_steps} steps without a final answer."
        self._record(Step(index=self.max_steps + 1, content=final))
        return final

    def _record(self, step: Step) -> None:
        self.steps.append(step)
        if not self.verbose:
            return
        if step.tool:
            payload = json.dumps(step.arguments, ensure_ascii=False)
            print(f"[step {step.index}] {step.tool}({payload}) -> {(step.observation or '')[:200]}")
        elif step.content:
            print(f"[step {step.index}] final -> {step.content[:200]}")


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print('usage: python agent.py "your task here"', file=sys.stderr)
        return 2

    agent = Agent(
        client=OpenAICompatibleClient(),
        max_steps=int(os.environ.get("MAX_STEPS", "8")),
    )
    print(f"[config] base_url={agent.client.base_url}  model={agent.client.model}")
    answer = agent.run(" ".join(args))
    print(f"\n=== answer ===\n{answer}\n")
    print(f"steps={len(agent.steps)}  workspace={WORKSPACE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())