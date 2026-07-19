"""Drive upstream mini-swe-agent with the Castform stdlib model.

Loads the package's own mini.yaml (prompts, observation and format-error
templates stay byte-identical to upstream) and runs DefaultAgent +
LocalEnvironment against one OpenAI-compatible endpoint.
"""

from __future__ import annotations

import argparse
import sys
from importlib import resources
from pathlib import Path

import yaml

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment

from castform_model import CastformToolcallModel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--step-limit", type=int, default=30)
    parser.add_argument("--shell-timeout", type=int, default=60)
    args = parser.parse_args()

    config = yaml.safe_load(
        resources.files("minisweagent.config").joinpath("mini.yaml").read_text()
    )
    agent_config = dict(config["agent"])
    agent_config.pop("mode", None)  # interactive-CLI knob, not an AgentConfig field
    agent_config.update(
        step_limit=args.step_limit,
        cost_limit=0.0,
        output_path=Path(args.output),
    )
    model_config = dict(config["model"])
    model_config.pop("model_kwargs", None)  # litellm-only passthrough

    agent = DefaultAgent(
        CastformToolcallModel(
            model_name=args.model,
            base_url=args.base_url,
            **model_config,
        ),
        LocalEnvironment(
            timeout=args.shell_timeout,
            env=config.get("environment", {}).get("env", {}),
        ),
        **agent_config,
    )
    result = agent.run(args.task)
    exit_status = result.get("exit_status", "")
    print(f"exit_status={exit_status} calls={agent.n_calls}")
    return 0 if exit_status in ("Submitted", "LimitsExceeded") else 1


if __name__ == "__main__":
    sys.exit(main())
