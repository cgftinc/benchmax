from __future__ import annotations

import asyncio
import json
import os
import shlex
import tempfile
from pathlib import Path
from typing import Any

from benchmax.envs.base_env import BaseEnv, RolloutResult


class HarborEnv(BaseEnv):
    """Run one Harbor trial as the rollout harness.

    This POC intentionally treats Harbor as an external harness process. The
    command receives the gateway endpoint through standard model env vars and
    writes a JSON result file with optional keys:

    {
      "messages": [...],
      "reward": {"correctness": 1.0},
      "truncated": false,
      "tool_calls_total": 0,
      "tool_calls_failed": 0,
      "artifacts": {...}
    }
    """

    def __init__(
        self,
        *,
        command: str | list[str],
        cwd: str | None = None,
        extra_env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.command = command
        self.cwd = cwd
        self.extra_env = extra_env or {}

    async def run_rollout(
        self,
        *,
        rollout_id: str,
        model: str,
        base_url: str,
        api_key: str,
        task: dict[str, Any],
        timeout: float,
    ) -> RolloutResult:
        tmp_path = Path(tempfile.mkdtemp(prefix=f"harbor-rollout-{rollout_id}-"))
        task_path = tmp_path / "task.json"
        result_path = tmp_path / "result.json"
        task_path.write_text(json.dumps(task, ensure_ascii=False), encoding="utf-8")

        env = {
            **os.environ,
            **self.extra_env,
            "OPENAI_API_KEY": api_key,
            "OPENAI_BASE_URL": base_url,
            "LLM_BASE_URL": base_url,
            "LITELLM_BASE_URL": base_url,
            "HARBOR_ROLLOUT_ID": rollout_id,
            "HARBOR_MODEL": model,
            "HARBOR_TASK_JSON": str(task_path),
            "HARBOR_RESULT_JSON": str(result_path),
        }
        argv = self._command_argv()
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self.cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return RolloutResult(
                messages=[],
                reward={"reward": 0.0},
                task=task,
                truncated=True,
                artifacts={"trial_dir": str(tmp_path), "error": "timeout"},
            )

        if proc.returncode != 0:
            raise RuntimeError(
                "Harbor rollout command failed "
                f"(exit={proc.returncode})\nSTDOUT:\n{stdout.decode(errors='replace')}\n"
                f"STDERR:\n{stderr.decode(errors='replace')}"
            )
        if not result_path.exists():
            raise RuntimeError(f"Harbor rollout command did not write {result_path}")

        payload = json.loads(result_path.read_text(encoding="utf-8"))
        reward = payload.get("reward") or payload.get("rewards") or {"reward": 0.0}
        return RolloutResult(
            messages=list(payload.get("messages") or []),
            reward={str(k): float(v) for k, v in reward.items()},
            task=payload.get("task", task),
            truncated=bool(payload.get("truncated", False)),
            tool_calls_total=int(payload.get("tool_calls_total", 0) or 0),
            tool_calls_failed=int(payload.get("tool_calls_failed", 0) or 0),
            artifacts={
                "trial_dir": payload.get("trial_dir", str(tmp_path)),
                **dict(payload.get("artifacts") or {}),
            },
        )

    def _command_argv(self) -> list[str]:
        if isinstance(self.command, str):
            return shlex.split(self.command)
        return list(self.command)
