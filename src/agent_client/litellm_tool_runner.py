from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import litellm

from src.agent_client.environment.terminal_access_tool_provider import (
    TerminalAccessToolImplementationProvider,
)


MERGE_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "view_current_merge_conflict_with",
            "description": "View the current merge conflict and optional surrounding context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "context_window_size": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Number of context lines around the current conflict.",
                    },
                    "reason": {"type": "string"},
                },
                "required": ["context_window_size", "reason"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_merge_conflict_at",
            "description": "View a merge conflict by zero-based conflict index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "conflict_index": {"type": "integer", "minimum": 0},
                    "context_window_size": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Number of context lines around the conflict.",
                    },
                    "reason": {"type": "string"},
                },
                "required": ["conflict_index", "context_window_size", "reason"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_current_merge_conflict_with",
            "description": "Resolve the current merge conflict with replacement text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Replacement content for the conflict region.",
                    },
                    "reason": {"type": "string"},
                },
                "required": ["content", "reason"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_diff_for",
            "description": "View the diff between merge parent commits for a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "relative_path_from_project_root": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["relative_path_from_project_root", "reason"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_file_at",
            "description": "View the full content of a file in the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "relative_path_from_project_root": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["relative_path_from_project_root", "reason"],
                "additionalProperties": False,
            },
        },
    },
]


@dataclass
class LiteLLMToolRunnerResult:
    completed: bool
    turns: int
    finish_reason: str | None
    transcript: list[dict[str, Any]]
    usage: dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    remaining_conflicts: int | None = None
    tool_error: str | None = None


@dataclass
class ToolDispatchResult:
    content: str
    success: bool


class LiteLLMMergeToolRunner:
    """Run GitGoodBench merge tools through LiteLLM function calling."""

    def __init__(
        self,
        tool_provider: TerminalAccessToolImplementationProvider,
        model: str,
        max_turns: int = 40,
        temperature: float = 0.0,
        completion_fn: Callable[..., Any] | None = None,
    ):
        self._tool_provider = tool_provider
        self._model = model
        self._max_turns = max_turns
        self._temperature = temperature
        self._completion_fn = completion_fn or litellm.completion

    def run(self, system_prompt: str, user_prompt: str) -> LiteLLMToolRunnerResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        transcript: list[dict[str, Any]] = [
            {"turn": 0, "message": messages[0]},
            {"turn": 0, "message": messages[1]},
        ]
        usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        total_cost = 0.0
        finish_reason: str | None = None

        for turn in range(1, self._max_turns + 1):
            response = self._completion_fn(
                model=self._model,
                messages=messages,
                tools=MERGE_TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=self._temperature,
            )
            self._accumulate_usage(usage, response)
            total_cost += self._response_cost(response)

            choice = response.choices[0]
            message = choice.message
            finish_reason = getattr(choice, "finish_reason", None)
            assistant_message = self._assistant_message(message)
            messages.append(assistant_message)
            transcript.append({"turn": turn, "message": assistant_message})

            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                remaining_conflicts = self._remaining_conflicts()
                completed = remaining_conflicts in (0, None)
                return LiteLLMToolRunnerResult(
                    completed=completed,
                    turns=turn,
                    finish_reason=finish_reason
                    if completed
                    else "stopped_with_unresolved_conflicts",
                    transcript=transcript,
                    usage=usage,
                    cost_usd=total_cost,
                    remaining_conflicts=remaining_conflicts,
                )

            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                raw_arguments = tool_call["function"].get("arguments") or "{}"
                dispatch_result = self._dispatch_tool(tool_name, raw_arguments)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": dispatch_result.content,
                }
                messages.append(tool_message)
                transcript.append(
                    {"turn": turn, "tool_name": tool_name, "message": tool_message}
                )
                remaining_conflicts = self._remaining_conflicts()
                if not dispatch_result.success:
                    return LiteLLMToolRunnerResult(
                        completed=False,
                        turns=turn,
                        finish_reason="tool_error",
                        transcript=transcript,
                        usage=usage,
                        cost_usd=total_cost,
                        remaining_conflicts=remaining_conflicts,
                        tool_error=dispatch_result.content,
                    )
                if remaining_conflicts == 0:
                    return LiteLLMToolRunnerResult(
                        completed=True,
                        turns=turn,
                        finish_reason="all_conflicts_resolved",
                        transcript=transcript,
                        usage=usage,
                        cost_usd=total_cost,
                        remaining_conflicts=remaining_conflicts,
                    )

        remaining_conflicts = self._remaining_conflicts()
        return LiteLLMToolRunnerResult(
            completed=False,
            turns=self._max_turns,
            finish_reason=finish_reason or "max_turns",
            transcript=transcript,
            usage=usage,
            cost_usd=total_cost,
            remaining_conflicts=remaining_conflicts,
        )

    @staticmethod
    def _assistant_message(message: Any) -> dict[str, Any]:
        content = getattr(message, "content", None)
        tool_calls = getattr(message, "tool_calls", None) or []
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": content or "",
        }
        if tool_calls:
            assistant_message["tool_calls"] = [
                LiteLLMMergeToolRunner._tool_call_to_dict(tool_call)
                for tool_call in tool_calls
            ]
        return assistant_message

    @staticmethod
    def _tool_call_to_dict(tool_call: Any) -> dict[str, Any]:
        function = getattr(tool_call, "function", None)
        return {
            "id": getattr(tool_call, "id", ""),
            "type": getattr(tool_call, "type", "function"),
            "function": {
                "name": getattr(function, "name", ""),
                "arguments": getattr(function, "arguments", "{}"),
            },
        }

    def _dispatch_tool(self, tool_name: str, raw_arguments: str) -> ToolDispatchResult:
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            return ToolDispatchResult(
                content=f"Invalid JSON arguments for {tool_name}: {exc}",
                success=False,
            )

        arguments.setdefault("reason", "called by LiteLLM parity runner")

        try:
            if tool_name == "view_current_merge_conflict_with":
                return ToolDispatchResult(
                    content=self._tool_provider.view_current_merge_conflict_with(
                        context_window_size=int(arguments.get("context_window_size", 5)),
                        reason=str(arguments["reason"]),
                    ),
                    success=True,
                )
            if tool_name == "view_merge_conflict_at":
                return ToolDispatchResult(
                    content=self._tool_provider.view_merge_conflict_at(
                        conflict_index=int(arguments["conflict_index"]),
                        context_window_size=int(arguments.get("context_window_size", 5)),
                        reason=str(arguments["reason"]),
                    ),
                    success=True,
                )
            if tool_name == "resolve_current_merge_conflict_with":
                return ToolDispatchResult(
                    content=self._tool_provider.resolve_current_merge_conflict_with(
                        content=str(arguments["content"]),
                        reason=str(arguments["reason"]),
                    ),
                    success=True,
                )
            if tool_name == "view_diff_for":
                return ToolDispatchResult(
                    content=self._tool_provider.view_diff_for(
                        relative_path_from_project_root=str(
                            arguments["relative_path_from_project_root"]
                        ),
                        reason=str(arguments["reason"]),
                    ),
                    success=True,
                )
            if tool_name == "view_file_at":
                return ToolDispatchResult(
                    content=self._tool_provider.view_file_at(
                        relative_path_from_project_root=str(
                            arguments["relative_path_from_project_root"]
                        ),
                        reason=str(arguments["reason"]),
                    ),
                    success=True,
                )
            return ToolDispatchResult(content=f"Unknown tool: {tool_name}", success=False)
        except Exception as exc:
            return ToolDispatchResult(
                content=f"Tool {tool_name} failed: {type(exc).__name__}: {exc}",
                success=False,
            )

    @staticmethod
    def _accumulate_usage(usage: dict[str, int], response: Any) -> None:
        response_usage = getattr(response, "usage", None)
        if response_usage is None:
            return
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(response_usage, key, None)
            if isinstance(value, int):
                usage[key] += value

    @staticmethod
    def _response_cost(response: Any) -> float:
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            return 0.0
        return float(cost or 0.0)

    def _remaining_conflicts(self) -> int | None:
        manager = getattr(self._tool_provider, "scenario_environment_manager", None)
        unresolved = getattr(manager, "unresolved_merge_conflicts", None)
        if unresolved is None:
            return None
        return len(unresolved)
