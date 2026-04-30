from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi

from src.agent_client.data.prompt_provider import PromptProvider
from src.agent_client.environment.docker_manager import DockerManager
from src.agent_client.environment.evaluator import Evaluator
from src.agent_client.environment.scenario_environment_manager import (
    ScenarioEnvironmentManager,
)
from src.agent_client.environment.scenario_type import ScenarioType
from src.agent_client.environment.terminal_access_tool_provider import (
    TerminalAccessToolImplementationProvider,
)
from src.agent_client.litellm_tool_runner import (
    MERGE_TOOL_SCHEMAS,
    LiteLLMMergeToolRunner,
)
from src.agent_client.utils.available_context import AvailableContext
from src.agent_client.utils.exceptions import ScenarioEnvironmentException
from src.data_processing_scripts.schemas import SampleDataRowV4


DATASET_NAME = "JetBrains/git_good_bench-lite"
DATASET_SPLIT = "train"
DEFAULT_IMAGE = "tolindenba/ytsaurus:python-3.10"


@dataclass
class LoadedSamples:
    samples: list[SampleDataRowV4]
    dataset_name: str | None
    dataset_split: str | None
    dataset_revision: str | None
    dataset_resolved_revision: str | None
    dataset_fingerprint: str | None
    data_paths: list[str]


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    loaded_samples = _load_samples(
        args.data_path,
        args.dataset_name,
        args.dataset_split,
        args.dataset_revision,
    )
    samples = _select_samples(
        loaded_samples.samples,
        task_ids=set(args.task_ids or []),
        limit=args.limit,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        for sample in samples:
            print(f"{sample.id}\t{sample.name}")
        return

    docker_manager = DockerManager(
        image=args.image,
        env_vars={},
        container_start_timeout=args.container_start_timeout_sec,
    )
    results_path = args.output_dir / "results.jsonl"
    results: list[dict[str, object]] = []
    manifest_path = args.output_dir / "run_manifest.json"
    try:
        docker_manager.setup_image()
        container = docker_manager.run_container()
        _prepare_container(container)
        manifest_path = _write_run_manifest(args, loaded_samples, samples, container)
        with results_path.open("w", encoding="utf-8") as results_file:
            for sample in samples:
                result = _run_sample(
                    sample=sample,
                    container=container,
                    docker_manager=docker_manager,
                    model=args.model,
                    max_turns=args.max_turns,
                    temperature=args.temperature,
                    output_dir=args.output_dir,
                    mock_solver=args.mock_solver,
                )
                results.append(result)
                results_file.write(json.dumps(result, sort_keys=True) + "\n")
                results_file.flush()
    finally:
        docker_manager.cleanup()

    solved = sum(1 for result in results if result.get("is_solved") is True)
    print(
        json.dumps(
            {
                "results_path": str(results_path),
                "manifest_path": str(manifest_path),
                "total": len(results),
                "solved": solved,
                "resolved_rate": solved / len(results) if results else 0.0,
                "model": args.model,
                "mock_solver": args.mock_solver,
            },
            indent=2,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GitGoodBench Lite merge parity with a LiteLLM tool agent."
    )
    parser.add_argument("--model", default="openrouter/anthropic/claude-sonnet-4.5")
    parser.add_argument(
        "--data-path",
        type=Path,
        action="append",
        default=None,
        help=(
            "Final SampleDataRowV4 parquet path. Defaults to the public "
            "JetBrains/git_good_bench-lite Hugging Face dataset."
        ),
    )
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--dataset-split", default=DATASET_SPLIT)
    parser.add_argument(
        "--dataset-revision",
        default=None,
        help=(
            "Optional Hugging Face dataset git revision. The resolved commit "
            "is recorded in the run manifest."
        ),
    )
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/litellm_merge"))
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--container-start-timeout-sec", type=int, default=300)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--mock-solver",
        choices=("none", "ground-truth"),
        default="none",
        help="Use ground truth instead of LiteLLM. For runner/evaluator smoke tests only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print selected merge samples; do not start Docker or call LiteLLM.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.max_turns < 1:
        parser.error("--max-turns must be positive")
    return args


def _load_samples(
    data_paths: Iterable[Path] | None,
    dataset_name: str,
    dataset_split: str,
    dataset_revision: str | None,
) -> LoadedSamples:
    samples: list[SampleDataRowV4] = []
    dataset_fingerprint = None
    resolved_revision = None
    if data_paths is None:
        dataset = load_dataset(
            dataset_name,
            split=dataset_split,
            revision=dataset_revision,
        )
        dataset_fingerprint = getattr(dataset, "_fingerprint", None)
        resolved_revision = _resolve_dataset_revision(dataset_name, dataset_revision)
        rows = [dict(row) for row in dataset]
    else:
        rows = []
        for data_path in data_paths:
            frame = pd.read_parquet(data_path)
            rows.extend(frame.to_dict(orient="records"))

    for row in rows:
        if row.get("sample_type") != ScenarioType.MERGE.value:
            continue
        samples.append(_sample_from_row(row))
    return LoadedSamples(
        samples=sorted(samples, key=lambda sample: sample.id),
        dataset_name=dataset_name if data_paths is None else None,
        dataset_split=dataset_split if data_paths is None else None,
        dataset_revision=dataset_revision if data_paths is None else None,
        dataset_resolved_revision=resolved_revision,
        dataset_fingerprint=dataset_fingerprint,
        data_paths=[] if data_paths is None else [str(path) for path in data_paths],
    )


def _sample_from_row(row: dict[str, object]) -> SampleDataRowV4:
    return SampleDataRowV4(
        id=str(row["id"]),
        name=str(row["name"]),
        default_branch=_optional_string(row.get("default_branch")),
        license=_optional_string(row.get("license")),
        stargazers=int(row.get("stargazers") or 0),
        created_at=_optional_string(row.get("created_at")),
        topics=_optional_string(row.get("topics")),
        programming_language=_optional_string(row.get("programming_language")),
        scenario=row["scenario"],
        sample_type=str(row["sample_type"]),
        project_size=_optional_string(row.get("project_size")),
        project_activity=_optional_string(row.get("project_activity")),
        difficulty=_optional_string(row.get("difficulty")),
    )


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, sort_keys=True)
    if pd.isna(value):
        return None
    return str(value)


def _select_samples(
    samples: list[SampleDataRowV4],
    task_ids: set[str],
    limit: int | None,
) -> list[SampleDataRowV4]:
    if task_ids:
        samples = [sample for sample in samples if sample.id in task_ids]
        missing = sorted(task_ids - {sample.id for sample in samples})
        if missing:
            raise ValueError(f"Unknown task ids: {', '.join(missing)}")
    if limit is not None:
        samples = samples[:limit]
    return samples


def _resolve_dataset_revision(dataset_name: str, revision: str | None) -> str | None:
    try:
        return HfApi().dataset_info(dataset_name, revision=revision).sha
    except Exception as exc:
        logging.warning("Could not resolve dataset revision for %s: %s", dataset_name, exc)
        return None


def _write_run_manifest(
    args: argparse.Namespace,
    loaded_samples: LoadedSamples,
    samples: list[SampleDataRowV4],
    container,
) -> Path:
    manifest_path = args.output_dir / "run_manifest.json"
    tool_schema_json = json.dumps(MERGE_TOOL_SCHEMAS, sort_keys=True)
    manifest = {
        "dataset": {
            "name": loaded_samples.dataset_name,
            "split": loaded_samples.dataset_split,
            "requested_revision": loaded_samples.dataset_revision,
            "resolved_revision": loaded_samples.dataset_resolved_revision,
            "fingerprint": loaded_samples.dataset_fingerprint,
            "data_paths": loaded_samples.data_paths,
        },
        "selection": {
            "task_ids": args.task_ids or [],
            "limit": args.limit,
            "selected_sample_ids": [sample.id for sample in samples],
        },
        "model": {
            "name": args.model,
            "temperature": args.temperature,
            "max_turns": args.max_turns,
            "provider": "litellm",
        },
        "environment": {
            "docker_image": args.image,
            "docker_image_id": getattr(container.image, "id", None),
            "container_start_timeout_sec": args.container_start_timeout_sec,
        },
        "tools": {
            "schema_sha256": hashlib.sha256(tool_schema_json.encode()).hexdigest(),
            "schemas": MERGE_TOOL_SCHEMAS,
        },
        "software": {
            "python": platform.python_version(),
            "litellm": _package_version("litellm"),
            "datasets": _package_version("datasets"),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _prepare_container(container) -> None:
    command = (
        'git config --global user.name "vcs-agent" && '
        'git config --global user.email "vcs@agent.takeover" && '
        'git config --global core.editor "true" && '
        "chmod u+x sequence_editor.sh"
    )
    err_code, output = container.exec_run(
        f'/bin/bash -c "{command}"',
        privileged=False,
        workdir="/usr/code",
    )
    if err_code != 0:
        raise ScenarioEnvironmentException(
            "Could not prepare container for parity run. "
            f"Error code: {err_code}, Output: {output.decode('utf-8')}"
        )


def _run_sample(
    sample: SampleDataRowV4,
    container,
    docker_manager: DockerManager,
    model: str,
    max_turns: int,
    temperature: float,
    output_dir: Path,
    mock_solver: str,
) -> dict[str, object]:
    started_at = time.time()
    host_agent_work_dir = os.path.join(
        os.getcwd(), docker_manager.agent_repo_dir, sample.name.split("/")[-1]
    )
    scenario = _parse_scenario(sample.scenario)
    manager = ScenarioEnvironmentManager(
        container=container,
        sample=sample,
        host_agent_work_dir=host_agent_work_dir,
    )
    evaluator = Evaluator(
        container=container,
        agent_target_branch_name=manager.AGENT_TARGET_BRANCH_NAME,
        repository_work_dir=manager.repository_work_dir,
        llm_client=None,
        host_agent_work_dir=host_agent_work_dir,
    )

    runner_result = None
    evaluation_metadata: dict[str, object] = {}
    is_solved = False
    error: str | None = None

    try:
        manager.setup_repository()
        manager.set_scenario(scenario)
        manager.set_scenario_type(ScenarioType.MERGE)
        manager.setup_scenario_preconditions()
        if mock_solver == "ground-truth":
            _apply_ground_truth_solution(container, manager.repository_work_dir, scenario)
        else:
            context = manager.provide_scenario_context(
                [
                    AvailableContext.PROGRAMMING_LANGUAGE,
                    AvailableContext.COMMIT_TEMPORAL_ORDERING,
                    AvailableContext.TOTAL_AMOUNT_OF_MERGE_CONFLICTS,
                    AvailableContext.FILES_WITH_CONFLICTS,
                    AvailableContext.ALL_MERGE_CONFLICTS,
                ]
            )
            context[AvailableContext.PROGRAMMING_LANGUAGE] = sample.programming_language
            user_prompt = PromptProvider.get_prompt_for(
                ScenarioType.MERGE,
                scenario,
                context=context,
            )
            tool = TerminalAccessToolImplementationProvider(
                container=container,
                error_message=None,
                max_num_chars_bash_output=30000,
                bash_timeout=180,
                workdir=manager.repository_work_dir,
                scenario_environment_manager=manager,
            )
            runner = LiteLLMMergeToolRunner(
                tool_provider=tool,
                model=model,
                max_turns=max_turns,
                temperature=temperature,
            )
            runner_result = runner.run(PromptProvider.get_system_prompt(), user_prompt)

        evaluator.set_scenario(scenario)
        evaluator.set_scenario_type(ScenarioType.MERGE)
        scenario["repository"] = sample.name
        is_solved = evaluator.evaluate()
        evaluation_metadata = evaluator.get_evaluation_metadata()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logging.exception("Sample %s failed", sample.id)
    finally:
        try:
            manager.teardown_scenario()
        except Exception as exc:
            logging.warning("Scenario teardown failed for %s: %s", sample.id, exc)
        try:
            manager.teardown_repository()
        except Exception as exc:
            logging.warning("Repository teardown failed for %s: %s", sample.id, exc)

    transcript_path = None
    if runner_result is not None:
        transcript_dir = output_dir / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = transcript_dir / f"{sample.id}.json"
        transcript_path.write_text(
            json.dumps(runner_result.transcript, indent=2),
            encoding="utf-8",
        )

    return {
        "sample_id": sample.id,
        "repository": sample.name,
        "model": model,
        "mock_solver": mock_solver,
        "is_solved": is_solved,
        "error": error,
        "execution_time_ms": int((time.time() - started_at) * 1000),
        "runner": None
        if runner_result is None
        else {
            "completed": runner_result.completed,
            "turns": runner_result.turns,
            "finish_reason": runner_result.finish_reason,
            "usage": runner_result.usage,
            "cost_usd": runner_result.cost_usd,
            "remaining_conflicts": runner_result.remaining_conflicts,
            "tool_error": runner_result.tool_error,
            "transcript_path": str(transcript_path),
        },
        "evaluation_metadata": evaluation_metadata,
    }


def _apply_ground_truth_solution(container, repository_work_dir: str, scenario: dict) -> None:
    command = (
        "git merge --abort || true; "
        f"git reset --hard {scenario['merge_commit_hash']}"
    )
    err_code, output = container.exec_run(
        f'/bin/bash -c "{command}"',
        privileged=False,
        workdir=repository_work_dir,
    )
    if err_code != 0:
        raise ScenarioEnvironmentException(
            "Ground-truth mock solver failed. "
            f"Error code: {err_code}, Output: {output.decode('utf-8')}"
        )


def _parse_scenario(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str):
        raise TypeError(f"Expected scenario to be a dict or string, got {type(value).__name__}")

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict):
        raise TypeError(f"Expected scenario to parse to dict, got {type(parsed).__name__}")
    return parsed


if __name__ == "__main__":
    main()
