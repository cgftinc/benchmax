"""Minimal RAG search environment written by ``castform setup --template rag``.

The environment searches a hosted Castform corpus, answers inside
``<answer>...</answer>``, and earns separate correctness and citation rewards.
Everything needed to understand and launch the run lives in this script; the
concrete Postgres Search showcase under BenchMax examples is not a dependency.

Before validating, set ``CORPUS_NAME`` to an existing hosted corpus and replace
the committed seed JSONL with rows shaped as ``question``, ``answer``, and
``reference_chunks``. Use the public ``castform.rag`` library from
``generate_data`` when you want this script to prepare a corpus or generate QA
rows. The default stage sequence is data -> validate and deliberately stops
before the explicit, confirmed GPU launch.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import re
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

from benchmax.bundle import dump_bundle
from benchmax.envs import (
    BaseEnv,
    BaseRollout,
    DatasetSplit,
    Example,
    InjectedAuth,
    JsonlDataset,
    JsonRow,
    Tool,
    canonical_example_id,
)
from benchmax.rewards import (
    Judge,
    Rubric,
    clip01,
    evaluate_single_rubric,
    extract_completion_text,
)

from castform import config, validate_environment
from castform.platform.client import TrainerClient
from castform.platform.login import ensure_session
from castform.platform.environment_assets import upload_environment_assets
from castform.rag.corpus.postgres.search import PostgresSearch

CORPUS_NAME = "my-corpus"
JUDGE_MODEL = "gpt-5.4-mini"
MAX_SEARCH_CALLS = 6
MAX_TOOL_OUTPUT_CHARS = 8_000

CORRECTNESS_RUBRIC = Rubric(
    title="Answer correctness",
    description=(
        "The response correctly answers the question and is factually "
        "consistent with the reference answer."
    ),
    score_map={0: "Missing or incorrect.", 1: "Fully correct."},
)

_ANSWER_RE = re.compile(r"<answer\s*>(.*?)</answer\s*>", re.IGNORECASE | re.DOTALL)
_CITATION_RE = re.compile(r"\[Source:\s*([^\]]+)\]", re.IGNORECASE)


def _source_id(value: object) -> str:
    text = str(value or "").strip().lower().rsplit("/", 1)[-1]
    return text.rsplit(".", 1)[0] if "." in text else text


def _answer_text(messages: list[dict[str, Any]]) -> str:
    completion = extract_completion_text(messages)
    matches = list(_ANSWER_RE.finditer(completion))
    return matches[-1].group(1).strip() if matches else ""


class CustomSearchEnv(BaseEnv):
    """Small search env backed by a named Castform corpus."""

    reward_keys = ("answer_correctness", "citation_recall")
    system_prompt = f"""\
Answer the question using the search tool. You may search at most
{MAX_SEARCH_CALLS} times. Return the final response inside <answer>...</answer>
and cite supporting documents as [Source: <source_id>].
"""

    def __init__(self) -> None:
        super().__init__(
            max_turns=MAX_SEARCH_CALLS + 1,
            max_tool_calls=MAX_SEARCH_CALLS,
        )
        self._search = PostgresSearch(
            CORPUS_NAME,
            base_url=config.platform_url(),
        )
        self._judge = Judge(
            model=JUDGE_MODEL,
            base_url=config.llm_url(),
            auth=InjectedAuth("judge"),
            timeout=30.0,
        )

    def _canonicalize_id(self, value: object) -> str:
        """Customize this seam if your corpus uses a different source-id format."""

        return _source_id(value)

    async def create_dataset(
        self,
        split: DatasetSplit,
        base_dir: Path,
        *,
        max_examples: int | None = None,
    ) -> JsonlDataset[JsonRow]:
        return JsonlDataset(
            base_dir / f"{split}.jsonl",
            row_to_example=self._example_from_row,
            max_examples=max_examples,
        )

    def _example_from_row(self, row: JsonRow) -> Example[JsonRow]:
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise TypeError("dataset rows require a non-empty string 'question'")
        payload: JsonRow = {
            "prompt_messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
            ],
            "question": question,
            "ground_truth": row.get("answer", ""),
            "reference_chunks": row.get("reference_chunks", []),
        }
        return Example(id=canonical_example_id(payload), payload=payload)

    async def list_tools(self) -> list[Tool]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the configured corpus.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 20,
                                "default": 10,
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

    async def run_tool(
        self,
        rollout_id: str,
        tool_name: str,
        **tool_args: Any,
    ) -> str:
        if tool_name != "search":
            raise ValueError(f"Unknown tool: {tool_name}")
        query = tool_args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search requires a non-empty string 'query'")
        limit = tool_args.get("limit", 10)
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 20
        ):
            raise ValueError("search 'limit' must be an integer from 1 to 20")
        results = await self._search.search(
            query=query,
            mode="lexical",
            top_k=limit,
        )
        rendered = json.dumps(results, ensure_ascii=False, default=str)
        if len(rendered) <= MAX_TOOL_OUTPUT_CHARS:
            return rendered
        return rendered[:MAX_TOOL_OUTPUT_CHARS].rstrip() + "\n...[truncated]"

    async def compute_reward(self, rollout: BaseRollout) -> dict[str, float]:
        answer = _answer_text(rollout.messages)
        if not answer:
            return {key: 0.0 for key in self.reward_keys}

        question = str(rollout.example_args.get("question") or "")
        ground_truth = str(rollout.example_args.get("ground_truth") or "")
        judged = await evaluate_single_rubric(
            rubric=CORRECTNESS_RUBRIC,
            question=question,
            ground_truth=ground_truth,
            response=answer,
            judge=self._judge,
        )

        references = rollout.example_args.get("reference_chunks", [])
        gold_sources: set[str] = set()
        if isinstance(references, list):
            for chunk in references:
                if not isinstance(chunk, dict):
                    continue
                metadata = chunk.get("metadata")
                if not isinstance(metadata, dict):
                    continue
                source = self._canonicalize_id(
                    metadata.get("file") or metadata.get("file_path")
                )
                if source:
                    gold_sources.add(source)
        cited_sources = {
            self._canonicalize_id(match) for match in _CITATION_RE.findall(answer)
        }
        citation_recall = (
            len(gold_sources & cited_sources) / len(gold_sources)
            if gold_sources
            else 0.0
        )
        return {
            "answer_correctness": clip01(judged.score),
            "citation_recall": citation_recall,
        }


VALIDATE_CONFIG = {
    "model": "gpt-5.4-mini",
    "include_remote": True,
}

LAUNCH_CONFIG = {
    "max_context_len": 16_384,
    "num_epochs": 2,
}

TRAIN_FILE = "train.jsonl"
EVAL_FILE = "eval.jsonl"
ENV_ARGS: dict[str, Any] = {}

# Data preparation uses the project's ``castform[rag]`` dependency, while the
# remote rollout imports only the base Castform Postgres search client.
RUNTIME_DEPENDENCIES = [f"castform=={version('castform')}"]


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in Path(path).read_text("utf-8").splitlines():
        if raw.strip():
            rows.append(json.loads(raw))
    return rows


def _run_name() -> str:
    return str(LAUNCH_CONFIG.get("name") or CustomSearchEnv.__name__.lower())


def generate_data(force: bool = False) -> bool:
    """Keep committed seed rows, or replace this with public ``castform.rag`` calls.

    Corpus ingestion and QA generation are project-specific and are intentionally
    explicit here instead of being hidden behind a Castform CLI orchestration layer.
    """

    have = Path(TRAIN_FILE).exists() and Path(EVAL_FILE).exists()
    if have and not force:
        print(f"data: {TRAIN_FILE} / {EVAL_FILE} present — skipping (--force to redo)")
        return True
    print(
        "data: add your corpus-ingestion / QA-generation calls to generate_data() "
        "using the public castform.rag library, then write train/eval JSONL."
    )
    return False


def _print_scorecard(report: Any) -> None:
    for rollout_id, outcome in report.local.items():
        rewards = dict(outcome.rewards)
        total = sum(rewards.values())
        print(
            f"  {rollout_id}: termination_reason={outcome.termination_reason} "
            f"total={total:.3f} rewards={rewards}"
        )
    print(f"validate: {'PASS' if report.ok else 'FAIL'}")


async def _run_validation(env: CustomSearchEnv) -> Any:
    include_remote = bool(VALIDATE_CONFIG.get("include_remote", False))
    remote_assets = None
    if include_remote:
        bundle = dump_bundle(
            CustomSearchEnv,
            constructor_args=ENV_ARGS,
            pip_dependencies=RUNTIME_DEPENDENCIES,
        )
        remote_assets = upload_environment_assets(
            bundle=bundle,
            train_dataset=_load_jsonl(TRAIN_FILE),
            eval_dataset=(_load_jsonl(EVAL_FILE) if Path(EVAL_FILE).exists() else None),
            run_name=_run_name(),
        )
    return await validate_environment(
        env,
        model=str(VALIDATE_CONFIG["model"]),
        split="train",
        base_dir=Path("."),
        remote_assets=remote_assets,
    )


def validate() -> Any:
    env = CustomSearchEnv(**ENV_ARGS)
    report = asyncio.run(_run_validation(env))
    _print_scorecard(report)
    return report


def launch(assume_yes: bool = False) -> str | None:
    report = validate()
    if not report.ok:
        print("launch: validation failed; refusing to launch.", file=sys.stderr)
        return None
    if not assume_yes:
        reply = (
            input(
                f"Launch '{_run_name()}' on GPUs — this spends credits. Continue? [y/N] "
            )
            .strip()
            .lower()
        )
        if reply not in ("y", "yes"):
            print("launch: aborted.")
            return None

    bundle = dump_bundle(
        CustomSearchEnv,
        constructor_args=ENV_ARGS,
        pip_dependencies=RUNTIME_DEPENDENCIES,
    )
    uploaded = upload_environment_assets(
        bundle=bundle,
        train_dataset=_load_jsonl(TRAIN_FILE),
        eval_dataset=_load_jsonl(EVAL_FILE) if Path(EVAL_FILE).exists() else [],
        run_name=_run_name(),
    )
    launcher_args = {
        key: value
        for key, value in LAUNCH_CONFIG.items()
        if key not in ("name", "type")
    }
    with TrainerClient() as client:
        run_id = client.launch_training_run(
            name=_run_name(),
            launcher_args=launcher_args or None,
            **dataclasses.asdict(uploaded),
        )
    print(f"launch: started run {run_id}")
    return run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Run the Castform loop for this env: data -> validate -> launch.",
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=["data", "validate", "launch", "all"],
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the launch confirmation (it spends GPU credits).",
    )
    args = parser.parse_args(argv)

    ok = True
    if args.stage in ("data", "all"):
        ok = generate_data(force=args.force)
    if args.stage in ("validate", "all") and ok:
        ensure_session()  # local data preparation does not require platform login
        ok = bool(validate().ok)
    if args.stage == "launch":
        ensure_session()
        ok = launch(assume_yes=args.yes) is not None
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
