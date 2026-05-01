from types import SimpleNamespace

from src.agent_client.environment.terminal_access_tool_provider import (
    TerminalAccessToolImplementationProvider,
)
from src.agent_client.litellm_tool_runner import LiteLLMMergeToolRunner
from src.agent_client.utils.exceptions import ScenarioEnvironmentException


def _response_with_tool_call(name: str, arguments: str, call_id: str = "call-1"):
    tool_call = SimpleNamespace(
        id=call_id,
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


class RecoverableReadErrorToolProvider(FakeToolProvider):
    def view_file_at(self, relative_path_from_project_root: str, reason: str) -> str:
        assert relative_path_from_project_root == "missing.py"
        assert reason == "inspect missing file"
        return (
            "Could not fetch file at missing.py. "
            "The following error was raised: Path missing.py does not exist."
        )

    def resolve_current_merge_conflict_with(self, content: str, reason: str) -> str:
        assert content == "resolved\n"
        assert reason == "recover after read error"
        self.scenario_environment_manager.unresolved_merge_conflicts = []
        return (
            "Successfully resolved merge conflict in merge. "
            "No conflicts remaining, you must now terminate."
        )


class RecoverableDiffErrorToolProvider(FakeToolProvider):
    def view_diff_for(self, relative_path_from_project_root: str, reason: str) -> str:
        assert relative_path_from_project_root == "missing.py"
        assert reason == "inspect missing diff"
        return (
            "Could not compute diff between the scenario's parents for missing.py."
        )

    def resolve_current_merge_conflict_with(self, content: str, reason: str) -> str:
        assert content == "resolved\n"
        assert reason == "recover after diff error"
        self.scenario_environment_manager.unresolved_merge_conflicts = []
        return (
            "Successfully resolved merge conflict in merge. "
            "No conflicts remaining, you must now terminate."
        )


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

    assert result.conflicts_cleared is True
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

    assert result.conflicts_cleared is False
    assert result.finish_reason == "tool_error"
    assert result.remaining_conflicts == 0
    assert result.tool_error == (
        "Tool resolve_current_merge_conflict_with failed: "
        "RuntimeError: git commit failed"
    )


def test_litellm_runner_continues_after_recoverable_read_error():
    responses = [
        _response_with_tool_call(
            "view_file_at",
            (
                '{"relative_path_from_project_root": "missing.py", '
                '"reason": "inspect missing file"}'
            ),
            call_id="call-read",
        ),
        _response_with_tool_call(
            "resolve_current_merge_conflict_with",
            '{"content": "resolved\\n", "reason": "recover after read error"}',
            call_id="call-resolve",
        ),
    ]

    result = LiteLLMMergeToolRunner(
        tool_provider=RecoverableReadErrorToolProvider(),
        model="openrouter/test-model",
        completion_fn=lambda **kwargs: responses.pop(0),
    ).run("system", "user")

    assert result.conflicts_cleared is True
    assert result.turns == 2
    assert result.tool_error is None
    assert responses == []


def test_litellm_runner_continues_after_recoverable_diff_error():
    responses = [
        _response_with_tool_call(
            "view_diff_for",
            (
                '{"relative_path_from_project_root": "missing.py", '
                '"reason": "inspect missing diff"}'
            ),
            call_id="call-diff",
        ),
        _response_with_tool_call(
            "resolve_current_merge_conflict_with",
            '{"content": "resolved\\n", "reason": "recover after diff error"}',
            call_id="call-resolve",
        ),
    ]

    result = LiteLLMMergeToolRunner(
        tool_provider=RecoverableDiffErrorToolProvider(),
        model="openrouter/test-model",
        completion_fn=lambda **kwargs: responses.pop(0),
    ).run("system", "user")

    assert result.conflicts_cleared is True
    assert result.turns == 2
    assert result.tool_error is None
    assert responses == []


def test_litellm_runner_rejects_non_object_tool_arguments():
    result = LiteLLMMergeToolRunner(
        tool_provider=FakeToolProvider(),
        model="openrouter/test-model",
        completion_fn=lambda **kwargs: _response_with_tool_call(
            "view_file_at",
            '"missing.py"',
        ),
    ).run("system", "user")

    assert result.conflicts_cleared is False
    assert result.finish_reason == "tool_error"
    assert result.tool_error == (
        "Invalid JSON arguments for view_file_at: expected object, got str"
    )


def test_litellm_runner_does_not_complete_when_model_stops_with_conflicts_left():
    tool_provider = FakeToolProvider()

    result = LiteLLMMergeToolRunner(
        tool_provider=tool_provider,
        model="openrouter/test-model",
        completion_fn=lambda **kwargs: _response_without_tool_calls(),
    ).run("system", "user")

    assert result.conflicts_cleared is False
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


class MissingFileScenarioEnvironmentManager:
    def view_file_at(self, path: str):
        raise ScenarioEnvironmentException(f"Path {path} does not exist.")


class FailingDiffScenarioEnvironmentManager:
    def view_diff_between_merge_conflict_commits_for(self, path: str):
        raise ScenarioEnvironmentException(
            f"Could not compute diff between the scenario's parents for {path}."
        )


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


def test_view_file_error_is_recoverable_and_spaced():
    provider = TerminalAccessToolImplementationProvider(
        container=None,
        error_message=None,
        bash_timeout=180,
        max_num_chars_bash_output=30000,
        workdir="/tmp",
        scenario_environment_manager=MissingFileScenarioEnvironmentManager(),
    )

    result = provider.view_file_at(
        relative_path_from_project_root="missing.py",
        reason="test",
    )

    assert "missing.py. The following error" in result


def test_view_diff_error_is_recoverable_text():
    provider = TerminalAccessToolImplementationProvider(
        container=None,
        error_message=None,
        bash_timeout=180,
        max_num_chars_bash_output=30000,
        workdir="/tmp",
        scenario_environment_manager=FailingDiffScenarioEnvironmentManager(),
    )

    result = provider.view_diff_for(
        relative_path_from_project_root="missing.py",
        reason="test",
    )

    assert result == (
        "Could not compute diff between the scenario's parents for missing.py."
    )
