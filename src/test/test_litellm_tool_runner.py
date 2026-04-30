from types import SimpleNamespace

from src.agent_client.environment.terminal_access_tool_provider import (
    TerminalAccessToolImplementationProvider,
)
from src.agent_client.litellm_tool_runner import LiteLLMMergeToolRunner


def _response_with_tool_call(name: str, arguments: str):
    tool_call = SimpleNamespace(
        id="call-1",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    message = SimpleNamespace(content=None, tool_calls=[tool_call])
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        usage=usage,
    )


def _response_without_tool_calls():
    message = SimpleNamespace(content="I am done.", tool_calls=[])
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=usage,
    )


class FakeToolProvider:
    def __init__(self):
        self.resolutions = []
        self.scenario_environment_manager = SimpleNamespace(
            unresolved_merge_conflicts=["conflict"]
        )

    def resolve_current_merge_conflict_with(self, content: str, reason: str) -> str:
        self.resolutions.append((content, reason))
        self.scenario_environment_manager.unresolved_merge_conflicts = []
        return (
            "Successfully resolved merge conflict in merge. "
            "No conflicts remaining, you must now terminate."
        )


class FailingAfterQueueEmptyToolProvider(FakeToolProvider):
    def resolve_current_merge_conflict_with(self, content: str, reason: str) -> str:
        self.scenario_environment_manager.unresolved_merge_conflicts = []
        raise RuntimeError("git commit failed")


def test_litellm_runner_dispatches_resolve_tool_and_stops():
    tool_provider = FakeToolProvider()

    def completion_fn(**kwargs):
        assert kwargs["model"] == "openrouter/test-model"
        assert kwargs["tool_choice"] == "auto"
        return _response_with_tool_call(
            "resolve_current_merge_conflict_with",
            '{"content": "resolved\\n", "reason": "test"}',
        )

    result = LiteLLMMergeToolRunner(
        tool_provider=tool_provider,
        model="openrouter/test-model",
        completion_fn=completion_fn,
    ).run("system", "user")

    assert result.completed is True
    assert result.finish_reason == "all_conflicts_resolved"
    assert result.remaining_conflicts == 0
    assert result.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    assert tool_provider.resolutions == [("resolved\n", "test")]


def test_litellm_runner_does_not_complete_when_final_tool_errors():
    result = LiteLLMMergeToolRunner(
        tool_provider=FailingAfterQueueEmptyToolProvider(),
        model="openrouter/test-model",
        completion_fn=lambda **kwargs: _response_with_tool_call(
            "resolve_current_merge_conflict_with",
            '{"content": "resolved\\n", "reason": "test"}',
        ),
    ).run("system", "user")

    assert result.completed is False
    assert result.finish_reason == "tool_error"
    assert result.remaining_conflicts == 0
    assert result.tool_error == (
        "Tool resolve_current_merge_conflict_with failed: "
        "RuntimeError: git commit failed"
    )


def test_litellm_runner_does_not_complete_when_model_stops_with_conflicts_left():
    tool_provider = FakeToolProvider()

    result = LiteLLMMergeToolRunner(
        tool_provider=tool_provider,
        model="openrouter/test-model",
        completion_fn=lambda **kwargs: _response_without_tool_calls(),
    ).run("system", "user")

    assert result.completed is False
    assert result.finish_reason == "stopped_with_unresolved_conflicts"
    assert result.remaining_conflicts == 1


class FakeScenarioEnvironmentManager:
    def __init__(self):
        self.all_conflicts = ["c0", "c1", "c2"]
        self.unresolved_merge_conflicts = ["c1", "c2"]
        self.requested = None

    def view_conflict_at(self, conflict_index: int, context_window_size: int):
        self.requested = (conflict_index, context_window_size)
        return "ok"


def test_view_current_merge_conflict_uses_next_unresolved_index():
    manager = FakeScenarioEnvironmentManager()
    provider = TerminalAccessToolImplementationProvider(
        container=None,
        error_message=None,
        bash_timeout=180,
        max_num_chars_bash_output=30000,
        workdir="/tmp",
        scenario_environment_manager=manager,
    )

    assert (
        provider.view_current_merge_conflict_with(
            context_window_size=7,
            reason="test",
        )
        == "ok"
    )
    assert manager.requested == (1, 7)
