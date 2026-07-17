"""Stress Harbor's Modal trial lifecycle without involving a trainer or TITO.

This is a manual diagnostic. It runs real AIME trials concurrently, directs
Mini-SWE at an ordinary OpenAI-compatible endpoint, and records the latency
between agent completion and verifier start—the interval containing Harbor's
artifact collection.

Requires ``CASTFORM_API_KEY``, ``MODAL_TOKEN_ID``, and
``MODAL_TOKEN_SECRET``. Run from ``core/benchmax`` with::

    uv run --with 'harbor[modal]==0.18.0' \
      python -m scripts.modal_artifact_stress
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, override
from uuid import uuid4

from benchmax.envs.harbor import HarborEnv, HarborTrialTemplate, ModalCredentials
from benchmax.envs.shared_types import RolloutRequest
from harbor import (
    DatasetConfig,
    EnvironmentType,
    TrialAgentConfig,
    TrialEnvironmentConfig,
    TrialVerifierConfig,
)
from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.environments.base import BaseEnvironment


class ReproducibleMiniSweAgent(MiniSweAgent):
    """Use the same known-good Mini-SWE/LiteLLM pair as trainer validation."""

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)

        version_spec = f"=={self._version}" if self._version else ""
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                'if [ -f "$HOME/.local/bin/env" ]; then '
                '. "$HOME/.local/bin/env"; '
                'else export PATH="$HOME/.local/bin:$PATH"; fi; '
                "uv tool install --force --with 'litellm==1.75.5.post1' "
                f"'mini-swe-agent{version_spec}' && "
                "mini-swe-agent --help"
            ),
        )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--model", default="grok-4-1-fast-non-reasoning")
    parser.add_argument("--base-url", default="https://llm.castform.dev/v1")
    parser.add_argument("--agent-timeout-secs", type=float, default=300)
    parser.add_argument("--transfer-timeout-secs", type=float, default=120)
    parser.add_argument(
        "--modal-app-name",
        default="benchmax-modal-artifact-stress",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/benchmax-modal-artifact-stress"),
    )
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.agent_timeout_secs <= 0:
        parser.error("--agent-timeout-secs must be positive")
    if args.transfer_timeout_secs <= 0:
        parser.error("--transfer-timeout-secs must be positive")
    return args


async def _main(args: argparse.Namespace) -> None:
    api_key = _required_environment("CASTFORM_API_KEY")
    credentials = ModalCredentials(
        token_id=_required_environment("MODAL_TOKEN_ID"),
        token_secret=_required_environment("MODAL_TOKEN_SECRET"),
    )

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{uuid4().hex[:8]}"
    output_dir = args.output_dir.expanduser().resolve() / run_id
    trials_dir = output_dir / "trials"
    output_dir.mkdir(parents=True, exist_ok=False)
    results_path = output_dir / "results.jsonl"

    environment = HarborEnv(
        dataset=DatasetConfig(name="aime/aime", ref="latest"),
        eval_ratio=0,
        trial=HarborTrialTemplate(
            agent=TrialAgentConfig(
                import_path=("scripts.modal_artifact_stress:ReproducibleMiniSweAgent"),
                kwargs={"version": "2.4.5"},
                max_timeout_sec=args.agent_timeout_secs,
            ),
            environment=TrialEnvironmentConfig(
                type=EnvironmentType.MODAL,
                import_path="benchmax.envs.harbor.modal:BoundedModalEnvironment",
                kwargs={
                    "app_name": args.modal_app_name,
                    "transfer_timeout_secs": args.transfer_timeout_secs,
                },
            ),
            verifier=TrialVerifierConfig(),
            trials_dir=trials_dir,
        ),
        sandbox_credentials=credentials,
        max_concurrent_trials=args.count,
    )

    dataset = await environment.create_dataset(
        "train",
        output_dir / "dataset",
    )
    print(
        f"run_id={run_id} trials={args.count} examples={len(dataset)} "
        f"output={output_dir}",
        flush=True,
    )

    records: list[dict[str, Any]] = []
    write_lock = asyncio.Lock()

    async def run_one(index: int) -> None:
        rollout_id = f"stress-{run_id}-{index:04d}"
        request = RolloutRequest(
            rollout_id=rollout_id,
            example=dataset[index % len(dataset)],
            model=args.model,
            base_url=args.base_url,
            api_key=api_key,
        )
        started = time.perf_counter()
        error: Exception | None = None
        try:
            await environment.run_rollout(request)
        except Exception as exc:
            error = exc
        wall_seconds = time.perf_counter() - started

        record = _trial_record(
            index=index,
            rollout_id=rollout_id,
            trial_dir=trials_dir / rollout_id,
            wall_seconds=wall_seconds,
            error=error,
        )
        async with write_lock:
            records.append(record)
            with results_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, sort_keys=True) + "\n")
            completed = len(records)
            if completed % 10 == 0 or completed == args.count:
                print(f"completed={completed}/{args.count}", flush=True)

    try:
        async with asyncio.TaskGroup() as group:
            for index in range(args.count):
                group.create_task(run_one(index), name=f"modal-stress-{index:04d}")
    finally:
        await environment.aclose()

    summary = _summary(records)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _trial_record(
    *,
    index: int,
    rollout_id: str,
    trial_dir: Path,
    wall_seconds: float,
    error: Exception | None,
) -> dict[str, Any]:
    result_path = trial_dir / "result.json"
    result = (
        json.loads(result_path.read_text(encoding="utf-8"))
        if result_path.exists()
        else {}
    )
    agent = result.get("agent_execution") or {}
    verifier = result.get("verifier") or {}
    exception = result.get("exception_info") or {}

    agent_started = _timestamp(agent.get("started_at"))
    agent_finished = _timestamp(agent.get("finished_at"))
    verifier_started = _timestamp(verifier.get("started_at"))
    verifier_finished = _timestamp(verifier.get("finished_at"))
    trial_started = _timestamp(result.get("started_at"))
    trial_finished = _timestamp(result.get("finished_at"))

    post_agent_end = verifier_started or trial_finished
    return {
        "index": index,
        "rollout_id": rollout_id,
        "wall_seconds": round(wall_seconds, 6),
        "trial_seconds": _difference(trial_started, trial_finished),
        "setup_seconds": _difference(trial_started, agent_started),
        "agent_seconds": _difference(agent_started, agent_finished),
        "post_agent_seconds": _difference(agent_finished, post_agent_end),
        "verifier_seconds": _difference(verifier_started, verifier_finished),
        "finalize_seconds": _difference(verifier_finished, trial_finished),
        "exception_type": exception.get("exception_type"),
        "runner_error_type": type(error).__name__ if error is not None else None,
        "runner_error": str(error)[:500] if error is not None else None,
    }


def _summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    latency_fields = (
        "wall_seconds",
        "trial_seconds",
        "setup_seconds",
        "agent_seconds",
        "post_agent_seconds",
        "verifier_seconds",
        "finalize_seconds",
    )
    latencies = {
        field: _distribution(
            [float(record[field]) for record in records if record[field] is not None]
        )
        for field in latency_fields
    }
    return {
        "completed": len(records),
        "exceptions": dict(
            sorted(
                Counter(
                    record["exception_type"] or "none" for record in records
                ).items()
            )
        ),
        "runner_errors": dict(
            sorted(
                Counter(
                    record["runner_error_type"] or "none" for record in records
                ).items()
            )
        ),
        "latencies": latencies,
    }


def _distribution(values: Sequence[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 6),
        "p50": round(_percentile(ordered, 0.50), 6),
        "p95": round(_percentile(ordered, 0.95), 6),
        "p99": round(_percentile(ordered, 0.99), 6),
        "max": round(ordered[-1], 6),
    }


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _difference(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return round((end - start).total_seconds(), 6)


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


if __name__ == "__main__":
    asyncio.run(_main(_arguments()))
