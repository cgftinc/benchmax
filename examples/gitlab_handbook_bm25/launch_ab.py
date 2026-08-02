#!/usr/bin/env python3
"""Submit one or both prepared arms through Platform's internal scheduler route."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
ASSETS_PATH = ROOT / "artifacts" / "uploaded_assets.json"
SPEC_PATH = ROOT / "experiment.json"
DEFAULT_PLATFORM_URL = "https://api.castform.dev"
CASTFORM_REPO_URL = "https://github.com/castform-ai/castform.git"
ARMS = ("pre_harbor", "post_harbor")
TASK_VARIANTS = ("standard", "optimized")


def _load_assets() -> dict[str, Any]:
    if not ASSETS_PATH.is_file():
        raise SystemExit(
            f"missing {ASSETS_PATH}; run prepare.py, both build_bundle.py scripts, "
            "and upload_assets.py first"
        )
    return json.loads(ASSETS_PATH.read_text(encoding="utf-8"))


def _arm_envs(arm: str, assets: dict[str, Any]) -> dict[str, str]:
    arm_assets = assets["arms"][arm]
    dataset = assets["dataset"]
    envs = {
        "USER_DATA_ARG_ENV_CLS_PATH": arm_assets["env_cls_path"],
        "USER_DATA_ARG_ENV_METADATA_PATH": arm_assets["env_metadata_path"],
    }
    if arm == "pre_harbor":
        envs.update(
            {
                "USER_DATA_ARG_TRAIN_DATASET_PATH": dataset["train_path"],
                "USER_DATA_ARG_EVAL_DATASET_PATH": dataset["eval_path"],
            }
        )
    else:
        envs["USER_DATA_ARG_DATASET_PATH"] = dataset["prefix"]
    return envs


def _task_path(arm: str, task_variant: str) -> Path:
    if task_variant == "standard":
        return ROOT / arm / "task.yaml"
    if arm != "post_harbor":
        raise ValueError(
            "the optimized scheduler variant is post-Harbor only; "
            "select --arm post_harbor"
        )
    return ROOT / arm / "task_optimized.yaml"


def _task_body(
    arm: str,
    pool: str,
    assets: dict[str, Any],
    task_variant: str,
) -> dict[str, Any]:
    task_path = _task_path(arm, task_variant)
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    secrets = {name: value for name in ("WANDB_API_KEY",) if (value := os.environ.get(name))}
    return {
        "task_yaml": task_path.read_text(encoding="utf-8"),
        "envs": _arm_envs(arm, assets),
        "secrets": secrets,
        "private_workdir_auth": {
            "version": "github-https-v1",
            "source": CASTFORM_REPO_URL,
            "ref": spec["castform_refs"][arm],
        },
        "pool": pool,
    }


def _submit(
    *,
    arm: str,
    body: dict[str, Any],
    platform_url: str,
    auth_token: str,
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{platform_url.rstrip('/')}/v1/train/runs/internal/launch",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
            "User-Agent": "gitlab-bm25-gateway-ab/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        # Do not print the response body: a validation error could echo a
        # caller-supplied secret.
        raise RuntimeError(f"{arm}: scheduler launch rejected (HTTP {error.code})") from None
    except urllib.error.URLError as error:
        raise RuntimeError(f"{arm}: scheduler request failed: {error.reason}") from None
    if not isinstance(payload, dict) or not payload.get("runId") or not payload.get("jobId"):
        raise RuntimeError(f"{arm}: Platform returned an invalid scheduler response")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=(*ARMS, "both"),
        default="both",
        help="Arm to inspect or submit (default: both).",
    )
    parser.add_argument("--pool", default="gpu4")
    parser.add_argument(
        "--task-variant",
        choices=TASK_VARIANTS,
        default="standard",
        help=(
            "Scheduler task variant. 'optimized' is the pinned post-Harbor "
            "simple-optimized.yaml derivative."
        ),
    )
    parser.add_argument(
        "--platform-url",
        default=os.environ.get("CASTFORM_PLATFORM_URL", DEFAULT_PLATFORM_URL),
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Actually submit GPU jobs. Without this flag the command is a dry run.",
    )
    args = parser.parse_args()

    assets = _load_assets()
    selected = ARMS if args.arm == "both" else (args.arm,)
    if args.task_variant == "optimized" and selected != ("post_harbor",):
        parser.error("--task-variant optimized requires --arm post_harbor")
    bodies = {
        arm: _task_body(arm, args.pool, assets, args.task_variant)
        for arm in selected
    }
    if not args.launch:
        for arm, body in bodies.items():
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "pool": body["pool"],
                        "task": str(_task_path(arm, args.task_variant)),
                        "task_variant": args.task_variant,
                        "envs": body["envs"],
                        "secret_names": sorted(body["secrets"]),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        print("dry run only; add --launch to submit")
        return

    missing = [
        name for name in ("CASTFORM_AUTH_TOKEN", "WANDB_API_KEY") if not os.environ.get(name)
    ]
    if missing:
        raise SystemExit(f"launch requires these local environment variables: {', '.join(missing)}")
    auth_token = os.environ["CASTFORM_AUTH_TOKEN"]
    for arm, body in bodies.items():
        result = _submit(
            arm=arm,
            body=body,
            platform_url=args.platform_url,
            auth_token=auth_token,
        )
        print(
            json.dumps(
                {
                    "arm": arm,
                    "run_id": result["runId"],
                    "job_id": result["jobId"],
                    "warnings": result.get("warnings", []),
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
