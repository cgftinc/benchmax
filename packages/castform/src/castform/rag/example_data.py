"""Reusable corpus and dataset pipeline for Benchmax RAG examples.

Provider examples own their database ingestion adapter. This module owns the
shared documents -> chunks -> grounded train/eval JSONL work, which never runs
inside a training rollout or an environment bundle.
"""

from __future__ import annotations

import json
import os
import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmax.auth import ModelAuth, create_openai_client
from castform import config
from castform.model_auth import model_auth_for_endpoint
from castform.rag.chunkers.markdown import MarkdownChunker
from castform.rag.chunkers.models import Chunk, ChunkCollection

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_QA_MODEL = "gpt-5.4-mini"
DEFAULT_QUESTION_COUNT = 20
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class RagDataModelConfig:
    """Explicit model credentials used only by a local data-build stage."""

    base_url: str
    auth: ModelAuth
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    qa_model: str = DEFAULT_QA_MODEL

    @classmethod
    def from_env(cls, prefix: str) -> RagDataModelConfig:
        normalized = prefix.strip().upper()
        base_url = os.environ.get(f"{normalized}_MODEL_BASE_URL", "").strip()
        base_url = base_url or config.llm_url()
        api_key = os.environ.get(f"{normalized}_MODEL_API_KEY", "").strip()
        return cls(
            base_url=base_url,
            auth=model_auth_for_endpoint(
                api_key=api_key,
                base_url=base_url,
                purpose=f"{normalized} data generation",
            ),
            embedding_model=os.environ.get(
                f"{normalized}_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
            ),
            qa_model=os.environ.get(f"{normalized}_QA_MODEL", DEFAULT_QA_MODEL),
        )


@dataclass(frozen=True, slots=True)
class SyncOpenAIEmbedder:
    """Synchronous embedding callable for provider ingestion clients."""

    config: RagDataModelConfig
    request_id: str

    def __call__(self, texts: list[str]) -> list[list[float]]:
        client = create_openai_client(
            model=self.config.embedding_model,
            base_url=self.config.base_url,
            auth=self.config.auth,
            request_id=self.request_id,
            max_retries=2,
        )
        try:
            response = client.embeddings.create(
                model=self.config.embedding_model,
                input=texts,
                timeout=120,
            )
            return [item.embedding for item in response.data]
        finally:
            client.close()


@dataclass(frozen=True, slots=True)
class RagExampleData:
    """Shared filesystem, chunking, and QA-generation driver for an example."""

    name: str
    root: Path
    env_prefix: str

    @property
    def documents_dir(self) -> Path:
        return self.root / "documents"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    def model_config(self) -> RagDataModelConfig:
        return RagDataModelConfig.from_env(self.env_prefix)

    def dataset_files(self) -> dict[str, Path]:
        return {
            "train.jsonl": self.data_dir / "train.jsonl",
            "eval.jsonl": self.data_dir / "eval.jsonl",
        }

    def require_dataset_files(self) -> dict[str, Path]:
        files = self.dataset_files()
        missing = [str(path) for path in files.values() if not path.is_file()]
        if missing:
            raise RuntimeError(
                f"the {self.name} dataset has not been generated; run "
                f"`uv run python main.py data` first (missing: {', '.join(missing)})"
            )
        return files

    def build_chunks(self) -> ChunkCollection:
        """Deterministically chunk every Markdown document in the example."""

        paths = sorted((*self.documents_dir.rglob("*.md"), *self.documents_dir.rglob("*.mdx")))
        if not paths:
            raise RuntimeError(f"no Markdown documents found in {self.documents_dir}")

        chunker = MarkdownChunker(min_char=700, max_char=1_800, chunk_overlap=120)
        chunks: list[Chunk] = []
        for path in paths:
            chunks.extend(
                chunker.chunk(
                    path.read_text(encoding="utf-8"),
                    file=path.relative_to(self.documents_dir).as_posix(),
                    preprocess_mdx=path.suffix == ".mdx",
                )
            )
        if len(chunks) < 2:
            raise RuntimeError("the example needs at least two chunks for train/eval splitting")
        return ChunkCollection(chunks)

    def generate_question_data(
        self,
        chunks: ChunkCollection,
        config: RagDataModelConfig,
        *,
        force: bool = False,
        max_questions: int = 40,
        target_questions: int | None = None,
    ) -> dict[str, Path]:
        """Generate grounded QA rows and split them deterministically."""

        files = self.dataset_files()
        if not force and _dataset_is_reusable(files, target_questions):
            print("questions: using existing train.jsonl and eval.jsonl")
            return files
        if max_questions < 2:
            raise ValueError("max_questions must be at least two")
        if target_questions is not None and target_questions < 2:
            raise ValueError("target_questions must be at least two")

        question_count = target_questions or min(len(chunks), max_questions)
        selected = list(chunks)[: min(len(chunks), question_count)]
        per_chunk, extra = divmod(question_count, len(selected))
        client = create_openai_client(
            model=config.qa_model,
            base_url=config.base_url,
            auth=config.auth,
            request_id=f"{self.name}-data",
            max_retries=2,
        )
        try:
            rows = []
            for index, chunk in enumerate(selected):
                rows.extend(
                    _generate_rows(
                        client,
                        config.qa_model,
                        chunk,
                        count=per_chunk + (1 if index < extra else 0),
                    )
                )
        finally:
            client.close()

        random.Random(0).shuffle(rows)
        eval_count = max(1, round(len(rows) * 0.2))
        train_rows = rows[:-eval_count]
        eval_rows = rows[-eval_count:]
        if not train_rows:
            raise RuntimeError("question generation produced no training rows")

        self.data_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(files["train.jsonl"], train_rows)
        _write_jsonl(files["eval.jsonl"], eval_rows)
        print(f"questions: wrote {len(train_rows)} train / {len(eval_rows)} eval rows")
        return files

    def prepare(
        self,
        ingest: Callable[[ChunkCollection, RagDataModelConfig], None],
        *,
        force: bool = False,
        target_questions: int | None = None,
    ) -> dict[str, Path]:
        """Run documents -> chunks -> provider ingest -> QA dataset."""

        files = self.dataset_files()
        if not force and _dataset_is_reusable(files, target_questions):
            print("data: using existing corpus inputs and question files; pass --force to rebuild")
            return files
        config = self.model_config()
        chunks = self.build_chunks()
        ingest(chunks, config)
        return self.generate_question_data(
            chunks,
            config,
            force=force,
            target_questions=target_questions,
        )


def _generate_rows(
    client: Any,
    model: str,
    chunk: Chunk,
    *,
    count: int,
) -> list[dict[str, Any]]:
    source = str(chunk.get_metadata("file", "unknown"))
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    f"Create exactly {count} distinct, useful, self-contained questions that can "
                    "be answered only from the supplied source excerpt. Cover different facts, "
                    "constraints, or relationships. Return a JSON array of objects with exactly "
                    "two string fields each: question and answer. Every answer must be concise, "
                    "fully supported by the excerpt, and must not mention that an excerpt was "
                    "supplied."
                ),
            },
            {"role": "user", "content": f"Source: {source}\n\n{chunk.content}"},
        ],
    )
    content = response.choices[0].message.content or ""
    try:
        payload = json.loads(_JSON_FENCE_RE.sub("", content).strip())
    except json.JSONDecodeError as error:
        raise RuntimeError(f"QA model returned invalid JSON for {source}") from error
    if not isinstance(payload, list) or len(payload) != count:
        raise RuntimeError(
            f"QA model returned {len(payload) if isinstance(payload, list) else 0} "
            f"questions for {source}; expected {count}"
        )
    rows: list[dict[str, Any]] = []
    for item in payload:
        question = item.get("question") if isinstance(item, dict) else None
        answer = item.get("answer") if isinstance(item, dict) else None
        if not isinstance(question, str) or not question.strip():
            raise RuntimeError(f"QA model returned an empty question for {source}")
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError(f"QA model returned an empty answer for {source}")
        rows.append(
            {
                "question": question.strip(),
                "answer": answer.strip(),
                "reference_chunks": [chunk.to_dict()],
            }
        )
    return rows


def _dataset_is_reusable(
    files: dict[str, Path],
    target_questions: int | None,
) -> bool:
    if not all(path.is_file() for path in files.values()):
        return False
    if target_questions is None:
        return True
    row_count = sum(
        1
        for path in files.values()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return row_count == target_questions


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(f"{json.dumps(row, ensure_ascii=False, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )


__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_QA_MODEL",
    "DEFAULT_QUESTION_COUNT",
    "RagDataModelConfig",
    "RagExampleData",
    "SyncOpenAIEmbedder",
]
