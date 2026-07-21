#!/usr/bin/env python3
"""Reproduce the GitLab handbook BM25 RAG training path.

The script is deliberately boring infrastructure:
1. Sparse-check out the GitLab handbook subdirectory at a ref/commit.
2. Upload it to the Castform Corpora API/Postgres BM25 backend with pinned
   chunking parameters.
3. Render templates/gitlab_bm25_run.py to run.py with the chosen corpus name.
4. Stage train/eval datasets from local files or hosted URLs.
5. Run validate, then optionally launch and monitor a training run.

Launching spends GPU credits, so launch is opt-in via --launch.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]
ROOT = EXAMPLE_DIR
CASTFORM_BIN = os.environ.get(
    "CASTFORM_BIN",
    str(REPO_ROOT / ".venv" / "bin" / "castform")
    if (REPO_ROOT / ".venv" / "bin" / "castform").exists()
    else "castform",
)
DEFAULT_GITLAB_COMMIT = "3078d0213524f8ca0c0e3a70680a21929a9f65ff"
DEFAULT_CORPUS_NAME = "gitlab-handbook-bm25-3078d0213524-staging"
DEFAULT_TREE_URL = (
    "https://gitlab.com/gitlab-com/content-sites/handbook/-/tree/main/content/handbook"
    "?ref_type=heads"
)
DEFAULT_HF_DATASET_BASE_URL = (
    "https://huggingface.co/datasets/wingedbreadsticks/"
    "gitlab-handbook-bm25-3078d0213524/resolve/"
    "gold-qwen35-4b-bm25-2026-07-08/"
)
DEFAULT_WORK_DIR = ROOT / "work" / "gitlab_bm25_work"
TEMPLATE = ROOT / "templates" / "gitlab_bm25_run.py"
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts"
TERMINAL_STATUSES = {
    "complete",
    "failed",
    "stalled",
    "cancelled",
    "out_of_credits",
    "billing_error",
}
BAD_TERMINAL_STATUSES = TERMINAL_STATUSES - {"complete"}


@dataclass
class TreeSpec:
    repo_url: str
    ref: str
    subdir: str
    project_path: str


def info(message: str) -> None:
    print(f"[bm25-repro] {message}", flush=True)


def load_env_file(path: Path | None) -> None:
    if path is None:
        return
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.exists():
        info(f"env file not found, skipping: {path}")
        return

    loaded: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if not key or key in os.environ:
            continue
        os.environ[key] = value
        loaded.append(key)
    if loaded:
        info(f"loaded env keys from {path}: {', '.join(sorted(loaded))}")


def preflight_platform_api_key() -> None:
    if not os.environ.get("PLATFORM_API_KEY"):
        return
    client = None
    try:
        from benchmax import config
        from benchmax.rag.corpus.postgres.client import CorpusClient

        client = CorpusClient(base_url=config.platform_url())
        client.list_corpora()
    except ImportError:
        return
    except Exception as exc:  # noqa: BLE001 - normalize optional dependency errors.
        if exc.__class__.__name__ == "AuthenticationError" or "invalid" in str(exc).lower():
            os.environ.pop("PLATFORM_API_KEY", None)
            info(
                "PLATFORM_API_KEY was rejected by the corpus API; "
                "falling back to the Castform login / ACT_AS credential seam"
            )
            return
        info(f"could not preflight PLATFORM_API_KEY ({exc.__class__.__name__}); continuing")
    finally:
        if client is not None:
            client.close()


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path = ROOT,
    capture: bool = False,
    check: bool = True,
    env: dict[str, str] | None = None,
    echo_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    info("$ " + " ".join(cmd))
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=False,
    )
    if capture and echo_output and completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if check and completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed


def castform_cmd(*args: str) -> list[str]:
    return [CASTFORM_BIN, *args]


def load_run_configs() -> dict[str, dict[str, Any]]:
    run_py = ROOT / "run.py"
    if not run_py.exists():
        return {"launch": {}, "validate": {}}
    spec = importlib.util.spec_from_file_location("_gitlab_bm25_run_config", run_py)
    if spec is None or spec.loader is None:
        return {"launch": {}, "validate": {}}
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - config loading should not block setup.
        info(f"could not load run.py config ({exc.__class__.__name__}); using CLI defaults")
        return {"launch": {}, "validate": {}}
    return {
        "launch": dict(getattr(module, "LAUNCH_CONFIG", {}) or {}),
        "validate": dict(getattr(module, "VALIDATE_CONFIG", {}) or {}),
    }


def launch_set_keys(items: list[str] | None) -> set[str]:
    keys: set[str] = set()
    for item in items or []:
        if "=" in item:
            keys.add(item.split("=", 1)[0])
    return keys


def parse_tree_url(url: str) -> TreeSpec:
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.strip("/")
    if "/-/" not in path:
        raise ValueError(f"not a GitLab tree URL: {url}")
    project_path, route = path.split("/-/", 1)
    parts = route.split("/")
    if len(parts) < 2 or parts[0] != "tree":
        raise ValueError(f"not a GitLab tree URL: {url}")

    tail = "/".join(parts[1:])
    marker = "content/handbook"
    if marker in tail:
        i = tail.index(marker)
        ref = tail[:i].strip("/")
        subdir = tail[i:].strip("/")
    else:
        ref = parts[1]
        subdir = "/".join(parts[2:])
    if not ref:
        raise ValueError(f"could not parse ref from GitLab tree URL: {url}")
    if not subdir:
        raise ValueError(f"could not parse subdir from GitLab tree URL: {url}")
    return TreeSpec(
        repo_url=f"{parsed.scheme}://{parsed.netloc}/{project_path}.git",
        ref=ref,
        subdir=subdir,
        project_path=project_path,
    )


def resolve_source_spec(args: argparse.Namespace) -> TreeSpec:
    spec = parse_tree_url(args.gitlab_url)
    return TreeSpec(
        repo_url=args.repo_url or spec.repo_url,
        ref=args.ref or spec.ref,
        subdir=args.subdir or spec.subdir,
        project_path=spec.project_path,
    )


def docs_dir_git_metadata(args: argparse.Namespace, docs_dir: Path) -> dict[str, Any]:
    """Best-effort GitLab source metadata for an already-checked-out docs dir."""
    spec = resolve_source_spec(args)
    metadata: dict[str, Any] = {**asdict(spec), "docs_dir": str(docs_dir)}

    try:
        commit = subprocess.run(
            ["git", "-C", str(docs_dir), "rev-parse", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        ).stdout.strip()
        top = subprocess.run(
            ["git", "-C", str(docs_dir), "rev-parse", "--show-toplevel"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        ).stdout.strip()
    except OSError:
        commit = ""
        top = ""

    if re.fullmatch(r"[0-9a-f]{40}", commit or "", flags=re.IGNORECASE):
        metadata["resolved_commit"] = commit
    elif re.fullmatch(r"[0-9a-f]{40}", spec.ref or "", flags=re.IGNORECASE):
        metadata["resolved_commit"] = spec.ref

    if top:
        try:
            rel = docs_dir.relative_to(Path(top).resolve()).as_posix()
        except ValueError:
            rel = ""
        if rel:
            metadata["subdir"] = rel
    return metadata


def sparse_checkout(spec: TreeSpec, work_dir: Path) -> tuple[Path, str]:
    repo_dir = work_dir / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    if not (repo_dir / ".git").exists():
        run_cmd(["git", "init"], cwd=repo_dir)
        run_cmd(["git", "remote", "add", "origin", spec.repo_url], cwd=repo_dir)
        run_cmd(["git", "sparse-checkout", "init", "--cone"], cwd=repo_dir)
    else:
        run_cmd(["git", "remote", "set-url", "origin", spec.repo_url], cwd=repo_dir)

    run_cmd(["git", "sparse-checkout", "set", spec.subdir], cwd=repo_dir)
    fetched = run_cmd(
        ["git", "fetch", "--depth=1", "origin", spec.ref],
        cwd=repo_dir,
        capture=True,
        check=False,
    )
    if fetched.returncode != 0:
        info("shallow fetch failed; retrying without --depth")
        run_cmd(["git", "fetch", "origin", spec.ref], cwd=repo_dir)
    run_cmd(["git", "checkout", "--force", "FETCH_HEAD"], cwd=repo_dir)

    commit = (
        run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_dir, capture=True, echo_output=False)
        .stdout.strip()
        .splitlines()[-1]
    )
    docs_dir = repo_dir / spec.subdir
    if not docs_dir.is_dir():
        raise SystemExit(f"checked-out subdir does not exist: {docs_dir}")
    return docs_dir, commit


def ingest_corpus(args: argparse.Namespace, docs_dir: Path) -> dict[str, Any]:
    info(
        "ingesting BM25 corpus "
        f"name={args.corpus_name!r} min={args.min_chars} max={args.max_chars} "
        f"overlap={args.overlap_chars}"
    )
    try:
        from benchmax.rag.chunkers.inspector import ChunkInspector
        from benchmax.rag.chunkers.markdown import MarkdownChunker
        from benchmax.rag.corpus.postgres.source import PostgresChunkSource
    except ImportError as exc:
        raise SystemExit(
            f"RAG dependencies are missing: {exc}. Install with `uv pip install 'castform[rag]'`."
        ) from exc

    source = PostgresChunkSource(corpus_name=args.corpus_name)
    source._client.max_retries = args.upload_retries
    source._client.retry_backoff_seconds = args.upload_retry_backoff

    info(f"chunking documents from {docs_dir}")
    chunker = MarkdownChunker(
        min_char=args.min_chars,
        max_char=args.max_chars,
        chunk_overlap=args.overlap_chars,
    )
    collection = chunker.chunk_folder(str(docs_dir), file_extensions=[".md", ".mdx"])
    ChunkInspector(collection).summary(max_depth=3, max_files_per_folder=4)

    source.collection = collection
    source._corpus = source._client.get_or_create_corpus(source._corpus_name, on_limit="error")
    info(
        f"uploading {len(collection)} chunks to corpus {source._corpus.name} "
        f"with batch_size={args.batch_size}, workers={args.upload_workers}, "
        f"retries={args.upload_retries}"
    )
    upload_result = source._client.upload_chunks(
        corpus_id=source._corpus.id,
        collection=collection,
        batch_size=args.batch_size,
        show_progress=True,
        max_workers=args.upload_workers,
    )
    info(f"upload complete: inserted={upload_result.inserted_count}")
    return {"corpus_name": args.corpus_name, "corpus_id": source.corpus_id, "chunks": source.get_chunk_count()}


def render_run_py(corpus_name: str, *, backup: bool) -> Path:
    if not TEMPLATE.exists():
        raise SystemExit(f"missing template: {TEMPLATE}")
    run_py = ROOT / "run.py"
    content = TEMPLATE.read_text(encoding="utf-8")
    marker = 'CORPUS_NAME = "__CORPUS_NAME__"'
    replacement = f"CORPUS_NAME = {corpus_name!r}"
    if marker not in content:
        raise SystemExit(f"template marker not found in {TEMPLATE}")
    rendered = content.replace(marker, replacement)

    if backup and run_py.exists():
        backup_dir = ROOT / "work" / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup_path = backup_dir / f"run.py.{stamp}.bak"
        shutil.copy2(run_py, backup_path)
        info(f"backed up existing run.py to {backup_path}")
    run_py.write_text(rendered, encoding="utf-8")
    info(f"wrote {run_py} for corpus {corpus_name!r}")
    return run_py


def looks_like_url(value: str) -> bool:
    return urllib.parse.urlparse(value).scheme in {"http", "https"}


def default_dataset_source(kind: str, args: argparse.Namespace) -> str | Path:
    filename = f"{kind}_dataset.jsonl"
    if args.dataset_base_url:
        return urllib.parse.urljoin(args.dataset_base_url.rstrip("/") + "/", filename)

    if args.dataset_preset == "gold-curriculum":
        return urllib.parse.urljoin(DEFAULT_HF_DATASET_BASE_URL, filename)

    if args.dataset_preset == "root":
        candidate = ROOT / filename
        if candidate.exists():
            return candidate
        raise SystemExit(f"root {kind} dataset not found at {candidate}")

    preferred = ROOT / "datagen_b" / filename
    if preferred.exists():
        return preferred
    fallback = ROOT / filename
    if fallback.exists():
        return fallback
    raise SystemExit(
        f"no default {kind} dataset found. Pass --{kind}-source as a local path or URL."
    )


def ensure_dataset_aliases(path: Path) -> None:
    """Add prompt/ground_truth aliases expected by generic launch loaders."""
    changed = False
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{line_no} is not a JSON object")
            if "prompt" not in row and row.get("question") is not None:
                row["prompt"] = row["question"]
                changed = True
            if "ground_truth" not in row and row.get("answer") is not None:
                row["ground_truth"] = row["answer"]
                changed = True
            rows.append(row)
    if not changed:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)
    info(f"added prompt/ground_truth aliases in {path}")


def stage_one_dataset(source: str | Path, dest: Path) -> None:
    source_s = str(source)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if looks_like_url(source_s):
        info(f"downloading {source_s} -> {dest}")
        with urllib.request.urlopen(source_s, timeout=120) as response:
            tmp.write_bytes(response.read())
        tmp.replace(dest)
        ensure_dataset_aliases(dest)
        return

    src_path = Path(source_s).expanduser()
    if not src_path.is_absolute():
        src_path = (ROOT / src_path).resolve()
    if src_path.resolve() == dest.resolve():
        info(f"dataset already staged: {dest}")
        ensure_dataset_aliases(dest)
        return
    if not src_path.exists():
        raise SystemExit(f"dataset source does not exist: {src_path}")
    shutil.copy2(src_path, dest)
    ensure_dataset_aliases(dest)
    info(f"copied {src_path} -> {dest}")


def stage_datasets(args: argparse.Namespace) -> dict[str, str]:
    if args.generate_qa:
        qa_paths = generate_qa(args)
        train_source = qa_paths["train"]
        eval_source = qa_paths["eval"]
    else:
        train_source = args.train_source or default_dataset_source("train", args)
        eval_source = args.eval_source or default_dataset_source("eval", args)
    train_dest = ROOT / "train_dataset.jsonl"
    eval_dest = ROOT / "eval_dataset.jsonl"
    stage_one_dataset(train_source, train_dest)
    stage_one_dataset(eval_source, eval_dest)
    return {
        "train_dataset": str(train_dest),
        "eval_dataset": str(eval_dest),
        "train_source": str(train_source),
        "eval_source": str(eval_source),
        "dataset_preset": args.dataset_preset,
        "dataset_base_url": args.dataset_base_url
        or (DEFAULT_HF_DATASET_BASE_URL if args.dataset_preset == "gold-curriculum" else None),
    }


def generate_qa(args: argparse.Namespace) -> dict[str, str]:
    info(
        f"generating fresh QA from corpus {args.corpus_name!r}: "
        f"samples={args.qa_samples}, out={args.qa_out_dir}"
    )
    try:
        from benchmax.rag.qa_generation import pipeline as qa_pipeline
        from benchmax.rag.qa_generation.pipeline_config import (
            CorpusConfig,
            CorpusContextConfig,
            FilteringConfig,
            GenerationConfig,
            LLMDirectGenerationConfig,
            MicroBatchConfig,
            OutputConfig,
            PipelineConfig,
            PlatformConfig,
            RefinementConfig,
            TargetsConfig,
        )
    except ImportError as exc:
        raise SystemExit(
            f"RAG generation dependencies are missing: {exc}. "
            "Install with `uv pip install 'castform[rag]'`."
        ) from exc

    out_dir = Path(args.qa_out_dir)
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    concise_system_prompt = (
        "You are an expert QA dataset author for a retrieval-augmented QA benchmark. "
        "Write a single-focus question that asks for one specific fact, value, name, "
        "step, policy, owner, or decision from the GitLab handbook. Write a concise "
        "answer, normally under 60 words, with no preamble and no restating the "
        "question. Phrase the question in natural user language instead of copying "
        "the handbook sentence verbatim."
    )
    handbook_description = (
        "The GitLab handbook is a public, Markdown-based company handbook covering "
        "GitLab teams, policies, product and engineering processes, finance, people "
        "operations, sales, support, security, data, and internal ways of working. "
        "Questions should look like realistic employee or teammate searches against "
        "the handbook and should be answerable from the cited source chunks."
    )
    handbook_queries = [
        "What is the process for submitting a finance expense?",
        "Who owns a specific GitLab handbook process?",
        "What are the steps for a people group policy?",
        "How should a team use a particular handbook template?",
        "What does the handbook say about a product management responsibility?",
        "Which tool or system is used for a named internal workflow?",
        "What criteria apply to a named engineering or security process?",
    ]
    filters = ["quality_gate", "grounding_llm", "hop_count_validity"]
    if args.qa_filter_too_easy:
        filters.insert(1, "retrieval_too_easy_llm")

    cfg = PipelineConfig(
        platform=PlatformConfig(),
        corpus=CorpusConfig(
            corpus_name=args.corpus_name,
            corpus_id="",
            min_chunk_chars=args.qa_min_chunk_chars,
        ),
        corpus_context=CorpusContextConfig(
            enabled=False,
            description=handbook_description,
            example_queries=handbook_queries,
            generate_entity_patterns=args.qa_generate_entities,
        ),
        targets=TargetsConfig(
            total_samples=args.qa_samples,
            primary_type_distribution={
                "lookup": args.qa_lookup_fraction,
                "multi_hop": max(0.0, 1.0 - args.qa_lookup_fraction),
            },
            reasoning_mode_distribution={
                "factual": 1.0,
                "temporal": 0.0,
                "inference": 0.0,
                "sequential": 0.0,
            },
            hop_distribution={
                1: args.qa_single_hop_fraction,
                2: max(0.0, 1.0 - args.qa_single_hop_fraction),
            },
        ),
        generation=GenerationConfig(
            llm_direct=LLMDirectGenerationConfig(
                system_prompt=concise_system_prompt,
                max_concurrent=args.qa_max_concurrent,
            )
        ),
        filtering=FilteringConfig(filters=filters),
        refinement=RefinementConfig(),
        micro_batch=MicroBatchConfig(
            batch_size=args.qa_batch_size,
            max_parallel_batches=args.qa_max_parallel_batches,
            keep_checkpoints=args.qa_keep_checkpoints,
        ),
        output=OutputConfig(
            dir=str(out_dir),
            train_jsonl="train_dataset.jsonl",
            eval_jsonl="eval_dataset.jsonl",
        ),
        random_seed=args.qa_seed,
    )
    original_auto_tune = getattr(qa_pipeline, "auto_tune", None)
    if not args.qa_auto_tune and original_auto_tune is not None:
        info("disabled QA auto-tune so the requested BM25 curriculum mix is reproducible")
        qa_pipeline.auto_tune = lambda *args_, **kwargs_: {}
    try:
        result = qa_pipeline.run_pipeline(cfg, source_factory=None)
    finally:
        if original_auto_tune is not None:
            qa_pipeline.auto_tune = original_auto_tune
    if isinstance(result, dict):
        train_count = len(result.get("train_dataset") or [])
        eval_count = len(result.get("eval_dataset") or [])
        rejected_count = len(result.get("rejected") or result.get("rejected_dataset") or [])
        info(
            "QA generation result: "
            f"{train_count} train, {eval_count} eval"
            + (f", {rejected_count} rejected" if rejected_count else "")
        )
    else:
        info(f"QA generation result: {type(result).__name__}")

    train = out_dir / "train_dataset.jsonl"
    eval_ = out_dir / "eval_dataset.jsonl"
    if not train.exists() or not eval_.exists():
        raise SystemExit(f"qa-gen did not write expected files in {out_dir}")
    return {"train": str(train), "eval": str(eval_), "out_dir": str(out_dir)}


def validate_env(args: argparse.Namespace) -> None:
    validate_config = load_run_configs()["validate"]
    cmd = castform_cmd("validate")
    if args.reward_audit:
        cmd.append("--reward-audit")
    examples = args.validate_examples
    if examples is None:
        examples = validate_config.get("examples")
    max_turns = args.validate_max_turns
    if max_turns is None:
        max_turns = validate_config.get("max_turns")
    max_tool_calls = args.validate_max_tool_calls
    if max_tool_calls is None:
        max_tool_calls = validate_config.get("max_tool_calls")
    if examples is not None:
        cmd.extend(["--examples", str(examples)])
    if max_turns is not None:
        cmd.extend(["--max-turns", str(max_turns)])
    if max_tool_calls is not None:
        cmd.extend(["--max-tool-calls", str(max_tool_calls)])
    run_cmd(cmd)


def extract_json_object(text: str) -> Any:
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            obj, end = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if text[i + end :].strip():
            continue
        return obj
    raise ValueError("no JSON object found in command output")


def launch_run(args: argparse.Namespace) -> dict[str, Any]:
    launch_config = load_run_configs()["launch"]
    if not os.environ.get("PLATFORM_API_KEY"):
        info(
            "warning: PLATFORM_API_KEY is not set. The env template will capture "
            "the current Castform login bearer at bundle time; for the most stable "
            "long-run judge auth, set PLATFORM_API_KEY before launch."
        )
    run_name = args.run_name or str(
        launch_config.get("name") or f"gitlab-handbook-bm25-{args.corpus_name}"
    )
    cmd = castform_cmd("launch", "--name", run_name, "--yes", "--json")
    if args.skip_launch_validate:
        cmd.append("--skip-validate")
    user_set_keys = launch_set_keys(args.launch_set)
    for key, value in launch_config.items():
        if key in {"name", "type"} or value is None or key in user_set_keys:
            continue
        cmd.extend(["--set", f"{key}={value}"])
    for item in args.launch_set or []:
        cmd.extend(["--set", item])
    completed = run_cmd(cmd, capture=True)
    launch_info = extract_json_object(completed.stdout or "")
    info(f"launched {launch_info.get('run_id')} at {launch_info.get('url')}")
    return launch_info


def castform_json(cmd: list[str], *, check: bool = False) -> Any | None:
    completed = run_cmd(cmd, capture=True, check=check, echo_output=False)
    if completed.returncode != 0:
        return None
    try:
        return extract_json_object(completed.stdout or "")
    except ValueError:
        return None


def latest_matching_scalar(
    scalars: dict[str, list[dict[str, Any]]],
    needles: tuple[str, ...],
) -> tuple[str, int | None, float] | None:
    for key, series in scalars.items():
        lower = key.lower()
        if all(n in lower for n in needles) and series:
            last = series[-1]
            value = last.get("value")
            if isinstance(value, (int, float)):
                return key, last.get("step"), float(value)
    return None


def scalar_report(scalars: dict[str, list[dict[str, Any]]]) -> dict[str, tuple[str, int | None, float] | None]:
    return {
        "answer_correctness": latest_matching_scalar(scalars, ("answer_correctness",)),
        "retrieval_hit": latest_matching_scalar(scalars, ("retrieval_hit",)),
        "truncated": latest_matching_scalar(scalars, ("truncated",)),
        "reward": latest_matching_scalar(scalars, ("reward", "mean")),
    }


def fmt_metric(item: tuple[str, int | None, float] | None) -> str:
    if not item:
        return "n/a"
    _name, step, value = item
    step_s = "?" if step is None else str(step)
    return f"{value:.4g}@{step_s}"


def inspect_one_rollout(run_id: str, mode: str) -> list[str]:
    warnings: list[str] = []
    summary = castform_json(
        castform_cmd("runs", "rollouts", run_id, "--mode", mode, "--limit", "5", "--json")
    )
    examples = (summary or {}).get("examples") if isinstance(summary, dict) else None
    if not examples:
        return ["no stored eval rollout summary yet"]
    first = examples[0]
    example_id = first.get("promptMessageId") or first.get("id")
    if not example_id:
        return ["stored rollout summary did not include an example id"]

    heatmap = castform_json(
        castform_cmd(
            "runs",
            "rollouts",
            run_id,
            "--mode",
            mode,
            "--example",
            str(example_id),
            "--json",
        )
    )
    if not isinstance(heatmap, list) or not heatmap:
        return ["no rollout heatmap yet for first eval example"]
    rollout = max(heatmap, key=lambda r: r.get("step") if isinstance(r.get("step"), int) else -1)
    rollout_id = rollout.get("id")
    if not rollout_id:
        return ["rollout heatmap row did not include rollout id"]

    details = castform_json(
        castform_cmd(
            "runs",
            "rollout",
            run_id,
            str(rollout_id),
            "--dataset",
            "eval_dataset.jsonl",
            "--json",
        )
    )
    if not isinstance(details, dict):
        return ["could not read rollout details"]

    messages = details.get("messages") or []
    tool_text = "\n".join(m.get("content") or "" for m in messages if m.get("role") == "tool")
    assistant_text = "\n".join(
        m.get("content") or "" for m in messages if m.get("role") == "assistant"
    )
    reward_names = {r.get("name") for r in details.get("rewards") or []}
    expected = {
        "answer_correctness",
        "conciseness",
        "citation_recall",
        "citation_grounding",
        "retrieval_hit",
    }

    if "[source:" not in tool_text.lower():
        warnings.append("sample rollout has no [source:] search results")
    if "error:" in tool_text.lower():
        warnings.append("sample rollout tool output contains Error:")
    if "<answer" not in assistant_text.lower():
        warnings.append("sample rollout has no committed <answer> block")
    missing_rewards = sorted(k for k in expected if k not in reward_names)
    if missing_rewards:
        warnings.append(f"sample rollout missing reward keys: {', '.join(missing_rewards)}")
    if not warnings:
        info(f"rollout sample ok: {rollout_id}")
    return warnings


def monitor_run(args: argparse.Namespace, run_id: str) -> int:
    deadline = time.time() + args.monitor_minutes * 60
    rollout_checked = False
    saw_eval_scalar = False
    last_status = ""

    while True:
        status_payload = castform_json(castform_cmd("runs", "status", run_id, "--json")) or {}
        status = str(status_payload.get("status") or "unknown")
        last_status = status

        eval_payload = castform_json(
            castform_cmd("runs", "scalars", run_id, "--mode", "eval", "--json")
        )
        eval_scalars = {}
        if isinstance(eval_payload, dict) and isinstance(eval_payload.get("scalars"), dict):
            eval_scalars = eval_payload["scalars"]
            saw_eval_scalar = bool(eval_scalars)
        eval_report = scalar_report(eval_scalars)

        train_payload = castform_json(
            castform_cmd("runs", "scalars", run_id, "--mode", "train", "--json")
        )
        train_scalars = {}
        if isinstance(train_payload, dict) and isinstance(train_payload.get("scalars"), dict):
            train_scalars = train_payload["scalars"]
        train_report = scalar_report(train_scalars)

        info(
            "status={status} eval_reward={eval_reward} eval_correct={eval_correct} "
            "eval_retrieval={eval_retrieval} train_truncated={train_truncated}".format(
                status=status,
                eval_reward=fmt_metric(eval_report["reward"]),
                eval_correct=fmt_metric(eval_report["answer_correctness"]),
                eval_retrieval=fmt_metric(eval_report["retrieval_hit"]),
                train_truncated=fmt_metric(train_report["truncated"] or eval_report["truncated"]),
            )
        )

        trunc = train_report["truncated"] or eval_report["truncated"]
        if trunc and trunc[2] > args.max_truncated:
            info(f"warning: rollout/truncated {trunc[2]:.4g} exceeds {args.max_truncated}")
        if eval_report["retrieval_hit"] and eval_report["retrieval_hit"][2] <= args.min_retrieval_hit:
            info(
                "warning: eval retrieval_hit is at or below "
                f"{args.min_retrieval_hit}; inspect rollouts before trusting scalars"
            )

        if saw_eval_scalar and not rollout_checked:
            rollout_warnings = inspect_one_rollout(run_id, "eval")
            if rollout_warnings:
                for warning in rollout_warnings:
                    info(f"rollout warning: {warning}")
            else:
                rollout_checked = True

        if status in TERMINAL_STATUSES:
            break
        if not args.monitor_until_terminal and saw_eval_scalar and rollout_checked:
            info("first eval scalar and rollout sample checked; leaving run active")
            break
        if time.time() >= deadline:
            info("monitor time budget reached; leaving run active")
            break
        time.sleep(args.poll_seconds)

    return 1 if last_status in BAD_TERMINAL_STATUSES else 0


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    info(f"wrote manifest {path}")


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def upload_dataset(path: Path) -> dict[str, Any] | None:
    payload = castform_json(castform_cmd("data", "upload", str(path), "--json"), check=False)
    if isinstance(payload, dict):
        return {
            k: v
            for k, v in payload.items()
            if k in {"blobPath", "expiresAt", "willOverwrite"}
        }
    return None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_slug(manifest: dict[str, Any], corpus_name: str) -> str:
    commit = ((manifest.get("gitlab") or {}).get("resolved_commit") or "")[:12]
    base = corpus_name if not commit or commit in corpus_name else f"{corpus_name}-{commit}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", base).strip("-")


def package_artifacts(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(args.artifact_dir)
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()
    out_dir = out_dir / artifact_slug(manifest, args.corpus_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}
    for name in ("train_dataset.jsonl", "eval_dataset.jsonl", "run.py"):
        src = ROOT / name
        if src.exists():
            dst = out_dir / name
            shutil.copy2(src, dst)
            files[name] = str(dst)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files["manifest.json"] = str(manifest_path)

    dataset_uploads: dict[str, Any] = {}
    if args.upload_datasets:
        for name in ("train_dataset.jsonl", "eval_dataset.jsonl"):
            path = ROOT / name
            if path.exists():
                uploaded = upload_dataset(path)
                if uploaded:
                    dataset_uploads[name] = uploaded
                    info(f"uploaded {name}: {uploaded}")

    hashes = {
        name: file_sha256(Path(path))
        for name, path in files.items()
        if Path(path).exists()
    }
    result = {"dir": str(out_dir), "files": files, "sha256": hashes, "dataset_uploads": dataset_uploads}
    launch = manifest.get("launch") or {}
    if launch.get("run_id"):
        result["latest_run_id"] = launch["run_id"]
    if launch.get("url"):
        result["latest_run_url"] = launch["url"]
    (out_dir / "artifact_index.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    info(f"packaged artifacts in {out_dir}")
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gitlab-url", default=DEFAULT_TREE_URL, help="GitLab tree URL for the handbook subdir")
    p.add_argument("--repo-url", help="Override repo URL parsed from --gitlab-url")
    p.add_argument(
        "--ref",
        default=DEFAULT_GITLAB_COMMIT,
        help="Commit SHA, tag, or branch to fetch (defaults to the gold reference SHA)",
    )
    p.add_argument("--subdir", help="Repo subdir to ingest (overrides tree URL path)")
    p.add_argument("--docs-dir", help="Use an existing local docs folder instead of sparse-checkout")
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--corpus-name", default=DEFAULT_CORPUS_NAME)
    p.add_argument("--min-chars", type=int, default=1024)
    p.add_argument("--max-chars", type=int, default=2048)
    p.add_argument("--overlap-chars", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--upload-workers", type=int, default=1)
    p.add_argument("--upload-retries", type=int, default=12)
    p.add_argument("--upload-retry-backoff", type=float, default=1.0)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-ingest", action="store_true")
    p.add_argument("--skip-run-py", action="store_true")
    p.add_argument("--no-run-py-backup", action="store_true")
    p.add_argument("--skip-datasets", action="store_true")
    p.add_argument("--generate-qa", action="store_true", help="Generate fresh QA from the ingested corpus")
    p.add_argument("--qa-samples", type=int, default=320)
    p.add_argument("--qa-out-dir", default="generated/gitlab_bm25")
    p.add_argument("--qa-min-chunk-chars", type=int, default=400)
    p.add_argument("--qa-lookup-fraction", type=float, default=1.0)
    p.add_argument("--qa-single-hop-fraction", type=float, default=1.0)
    p.add_argument("--qa-seed", type=int, default=42)
    p.add_argument(
        "--qa-generate-entities",
        action="store_true",
        help="Run KeyBERT entity-pattern extraction during QA generation; slower on large corpora",
    )
    p.add_argument("--qa-max-concurrent", type=int, default=8)
    p.add_argument("--qa-batch-size", type=int, default=0, help="0 lets qa-gen choose automatically")
    p.add_argument(
        "--qa-max-parallel-batches",
        type=int,
        default=0,
        help="0 lets qa-gen choose automatically",
    )
    p.add_argument("--qa-keep-checkpoints", action="store_true")
    p.add_argument(
        "--qa-auto-tune",
        action="store_true",
        help="Allow qa-gen corpus auto-tune to override the requested curriculum mix",
    )
    p.add_argument(
        "--qa-filter-too-easy",
        action="store_true",
        help="Also prune lexically trivial BM25 questions; off by default for BM25 curriculum mix",
    )
    p.add_argument(
        "--dataset-preset",
        choices=("gold-curriculum", "legacy", "root"),
        default="gold-curriculum",
        help=(
            "Default dataset source when --train-source/--eval-source are omitted. "
            "gold-curriculum downloads the public reference split; legacy preserves the old "
            "datagen_b-then-root lookup; root uses ./train_dataset.jsonl and ./eval_dataset.jsonl."
        ),
    )
    p.add_argument(
        "--dataset-base-url",
        help=(
            "Public URL prefix containing train_dataset.jsonl and eval_dataset.jsonl. "
            "Overrides --dataset-preset defaults unless --train-source/--eval-source are set."
        ),
    )
    p.add_argument("--train-source", help="Local path or URL for train_dataset.jsonl")
    p.add_argument("--eval-source", help="Local path or URL for eval_dataset.jsonl")
    p.add_argument("--skip-validate", action="store_true")
    p.add_argument("--reward-audit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--validate-examples", type=int)
    p.add_argument("--validate-max-turns", type=int)
    p.add_argument("--validate-max-tool-calls", type=int)
    p.add_argument("--launch", action="store_true", help="Launch a GPU run after validation")
    p.add_argument("--skip-launch-validate", action="store_true")
    p.add_argument("--run-name")
    p.add_argument("--launch-set", action="append", help="Extra castform launch --set key=value")
    p.add_argument("--run-id", help="Monitor an existing run id and skip prepare/launch")
    p.add_argument("--no-monitor", action="store_true")
    p.add_argument("--monitor-until-terminal", action="store_true")
    p.add_argument("--monitor-minutes", type=float, default=60.0)
    p.add_argument("--poll-seconds", type=float, default=60.0)
    p.add_argument("--max-truncated", type=float, default=0.01)
    p.add_argument("--min-retrieval-hit", type=float, default=0.0)
    p.add_argument("--manifest", type=Path, default=ROOT / "gitlab_bm25_manifest.json")
    p.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    p.add_argument("--upload-datasets", action="store_true")
    p.add_argument(
        "--castform-base-domain",
        help="Castform environment base domain, e.g. castform.dev for staging",
    )
    p.add_argument(
        "--staging",
        action="store_true",
        help="Shortcut for --castform-base-domain castform.dev",
    )
    p.add_argument(
        "--env-file",
        type=Path,
        default=ROOT / ".env",
        help="Load credentials from this env file before calling Castform/RAG APIs",
    )
    p.add_argument("--no-env-file", action="store_true", help="Do not load --env-file")
    return p


def configure_platform(args: argparse.Namespace) -> dict[str, str]:
    if args.staging and not args.castform_base_domain:
        args.castform_base_domain = "castform.dev"
    if args.castform_base_domain:
        os.environ["CASTFORM_BASE_DOMAIN"] = args.castform_base_domain

    from benchmax import config

    platform = {
        "base_domain": config.base_domain(),
        "platform_url": config.platform_url(),
        "web_app_url": config.web_app_url(),
        "llm_url": config.llm_url(),
        "auth_url": config.auth_url(),
    }
    info(f"using Castform platform: {platform['platform_url']}")
    return platform


def main() -> int:
    args = build_parser().parse_args()
    platform = configure_platform(args)
    load_env_file(None if args.no_env_file else args.env_file)
    preflight_platform_api_key()
    previous_manifest = read_manifest(args.manifest)

    if args.run_id:
        if args.no_monitor:
            info("--run-id supplied with --no-monitor; nothing to do")
            return 0
        return monitor_run(args, args.run_id)

    manifest: dict[str, Any] = {
        "platform": platform,
        "corpus_name": args.corpus_name,
        "chunking": {
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "overlap_chars": args.overlap_chars,
        },
    }

    docs_dir: Path | None = None
    if args.docs_dir:
        docs_dir = Path(args.docs_dir).expanduser().resolve()
        if not docs_dir.is_dir():
            raise SystemExit(f"--docs-dir is not a folder: {docs_dir}")
        manifest["docs_dir"] = str(docs_dir)
        manifest["gitlab"] = docs_dir_git_metadata(args, docs_dir)
    elif not args.skip_download:
        spec = resolve_source_spec(args)
        docs_dir, commit = sparse_checkout(spec, args.work_dir)
        manifest["gitlab"] = {**asdict(spec), "resolved_commit": commit, "docs_dir": str(docs_dir)}
        if not re.fullmatch(r"[0-9a-f]{40}", spec.ref or "", flags=re.IGNORECASE):
            info(f"moving ref {spec.ref!r} resolved to commit {commit}; use that SHA for exact replay")
    else:
        info("skipping download; no docs dir will be available for ingest")

    if not args.skip_ingest:
        if docs_dir is None:
            raise SystemExit("ingest needs docs; pass --docs-dir or do not use --skip-download")
        manifest["corpus"] = ingest_corpus(args, docs_dir)
    elif (
        previous_manifest.get("corpus_name") == args.corpus_name
        and isinstance(previous_manifest.get("corpus"), dict)
    ):
        manifest["corpus"] = previous_manifest["corpus"]

    if not args.skip_run_py:
        render_run_py(args.corpus_name, backup=not args.no_run_py_backup)
        manifest["run_py"] = str(ROOT / "run.py")

    if not args.skip_datasets:
        manifest["datasets"] = stage_datasets(args)

    write_manifest(args.manifest, manifest)

    if not args.skip_validate:
        validate_env(args)

    launch_info: dict[str, Any] | None = None
    if args.launch:
        launch_info = launch_run(args)
        manifest["launch"] = launch_info
        write_manifest(args.manifest, manifest)
    else:
        info("prepared BM25 run. Pass --launch to start GPU training.")

    if launch_info and not args.no_monitor:
        rc = monitor_run(args, str(launch_info["run_id"]))
        manifest["artifacts"] = package_artifacts(args, manifest)
        write_manifest(args.manifest, manifest)
        return rc
    manifest["artifacts"] = package_artifacts(args, manifest)
    write_manifest(args.manifest, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
