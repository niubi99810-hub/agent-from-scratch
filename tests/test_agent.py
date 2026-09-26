"""Offline unit tests for the agent runtime.

They must pass without an API key: the LLM is replaced by a scripted fake, so
`python -m unittest discover -s tests -v` is safe to run in CI.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import (  # noqa: E402
    Agent,
    Tool,
    build_tool_specs,
    default_tools,
    dispatch_tool,
    safe_calc,
    trim_messages,
)


class ScriptedClient:
    """Fake LLM client: replays queued replies and records every request."""

    def __init__(self, replies: list[dict]) -> None:
        self.replies = list(replies)
        self.requests: list[dict] = []

    def chat(self, messages, tools=None):  # noqa: ANN001, ANN201
        self.requests.append({"messages": [dict(message) for message in messages], "tools": tools})
        if not self.replies:
            raise AssertionError("the agent asked for more replies than the script provides")
        return self.replies.pop(0)


def tool_call_reply(call_id: str, name: str, arguments: str) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                }
            }
        ]
    }


def final_reply(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


class SafeCalcTests(unittest.TestCase):
    def test_basic_arithmetic(self) -> None:
        self.assertEqual(safe_calc("2+3*4"), 14)
        self.assertEqual(safe_calc("(1+2)**3"), 27)
        self.assertEqual(safe_calc("-4//2"), -2)

    def test_rejects_function_calls(self) -> None:
        with self.assertRaises(ValueError):
            safe_calc("__import__('os').system('echo pwned')")

    def test_rejects_huge_exponent(self) -> None:
        with self.assertRaises(ValueError):
            safe_calc("9**100000")


class DispatchTests(unittest.TestCase):
    def test_successful_call_returns_tool_output(self) -> None:
        result = dispatch_tool(default_tools(), "calculator", '{"expression": "6*7"}')
        self.assertEqual(result, "42")

    def test_unknown_tool_is_reported_not_raised(self) -> None:
        result = dispatch_tool(default_tools(), "nope", "{}")
        self.assertTrue(result.startswith("ERROR: unknown tool"))
        self.assertIn("calculator", result)

    def test_invalid_json_is_reported(self) -> None:
        result = dispatch_tool(default_tools(), "calculator", "{not json")
        self.assertIn("not valid JSON", result)

    def test_bad_arguments_are_reported(self) -> None:
        result = dispatch_tool(default_tools(), "calculator", '{"wrong": "name"}')
        self.assertTrue(result.startswith("ERROR: bad arguments"))

    def test_tool_exception_is_converted_to_text(self) -> None:
        def boom() -> str:
            raise ValueError("boom")

        tool = Tool(
            name="boom",
            description="always fails",
            parameters={"type": "object"},
            run=boom,
        )
        result = dispatch_tool([tool], "boom", "{}")
        self.assertEqual(result, "ERROR: ValueError: boom")

    def test_read_file_cannot_escape_the_workspace(self) -> None:
        read_file = next(tool for tool in default_tools() if tool.name == "read_file")
        result = dispatch_tool([read_file], "read_file", '{"path": "../../../../etc/passwd"}')
        self.assertIn("ERROR", result)


class TrimMessagesTests(unittest.TestCase):
    def test_system_prompt_is_never_dropped(self) -> None:
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]
        trimmed = trim_messages(messages, max_chars=1, keep_recent_units=0)
        self.assertEqual(trimmed[0]["role"], "system")

    def test_tool_call_never_separated_from_its_result(self) -> None:
        messages: list[dict] = [{"role": "system", "content": "sys"}]
        for index in range(6):
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {"name": "calculator", "arguments": "{}"},
                        }
                    ],
                }
            )
            messages.append({"role": "tool", "tool_call_id": f"call_{index}", "content": "x" * 500})

        trimmed = trim_messages(messages, max_chars=10, keep_recent_units=2)

        self.assertEqual(trimmed[0]["role"], "system")
        self.assertEqual(trimmed[1]["role"], "assistant")
        self.assertEqual(len(trimmed), 1 + 2 * 2)
        for position, message in enumerate(trimmed):
            if message["role"] == "tool":
                self.assertEqual(trimmed[position - 1]["role"], "assistant")

    def test_short_transcript_is_untouched(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        self.assertEqual(trim_messages(messages, max_chars=10_000), messages)


class AgentTests(unittest.TestCase):
    def test_agent_calls_a_tool_then_answers(self) -> None:
        client = ScriptedClient(
            [
                tool_call_reply("call_1", "calculator", json.dumps({"expression": "2+3*4"})),
                final_reply("结果是 14"),
            ]
        )
        agent = Agent(client=client, verbose=False)

        self.assertEqual(agent.run("2+3*4 等于多少？"), "结果是 14")

        follow_up = client.requests[1]["messages"]
        tool_messages = [message for message in follow_up if message.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0]["content"], "14")
        self.assertEqual(tool_messages[0]["tool_call_id"], "call_1")
        self.assertEqual(agent.steps[0].tool, "calculator")
        self.assertEqual(agent.steps[0].observation, "14")

    def test_agent_gives_up_after_max_steps(self) -> None:
        reply = tool_call_reply("call_1", "calculator", '{"expression": "1+1"}')
        client = ScriptedClient([reply, reply, reply])
        agent = Agent(client=client, max_steps=3, verbose=False)

        result = agent.run("loop forever")

        self.assertTrue(result.startswith("ERROR: stopped after 3 steps"))
        self.assertEqual(len(client.requests), 3)

    def test_repeating_the_same_call_is_flagged(self) -> None:
        client = ScriptedClient(
            [
                tool_call_reply("call_1", "calculator", '{"expression": "1+1"}'),
                tool_call_reply("call_2", "calculator", '{"expression": "1+1"}'),
                final_reply("done"),
            ]
        )
        agent = Agent(client=client, verbose=False)
        agent.run("go")

        observations = [step.observation or "" for step in agent.steps]
        self.assertTrue(any("already called" in observation for observation in observations))

    def test_failing_tool_is_fed_back_to_the_model(self) -> None:
        client = ScriptedClient(
            [
                tool_call_reply("call_1", "calculator", '{"expression": "not arithmetic"}'),
                final_reply("ok"),
            ]
        )
        agent = Agent(client=client, verbose=False)
        agent.run("go")

        self.assertTrue((agent.steps[0].observation or "").startswith("ERROR"))


class ToolSpecTests(unittest.TestCase):
    def test_specs_match_the_function_calling_schema(self) -> None:
        specs = build_tool_specs(default_tools())

        self.assertEqual({spec["type"] for spec in specs}, {"function"})
        names = {spec["function"]["name"] for spec in specs}
        self.assertEqual(names, {"calculator", "list_files", "read_file"})
        for spec in specs:
            self.assertEqual(spec["function"]["parameters"]["type"], "object")
            self.assertIn("required", spec["function"]["parameters"])


if __name__ == "__main__":
    unittest.main()