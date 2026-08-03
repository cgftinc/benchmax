from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from benchmax.auth import StaticBearerAuth
from castform.model_auth import CastformModelAuth
from castform.rag.chunkers.models import Chunk, ChunkCollection
from castform.rag.example_data import RagDataModelConfig, RagExampleData


def test_model_config_defaults_to_castform_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        "castform.rag.example_data.config.llm_url", lambda: "https://llm.castform.com/v1"
    )
    monkeypatch.delenv("NEON_RAG_MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("NEON_RAG_MODEL_API_KEY", raising=False)

    config = RagDataModelConfig.from_env("NEON_RAG")

    assert config.base_url == "https://llm.castform.com/v1"
    assert isinstance(config.auth, CastformModelAuth)


def test_model_config_accepts_external_endpoint_with_key(monkeypatch) -> None:
    monkeypatch.setattr(
        "castform.rag.example_data.config.llm_url", lambda: "https://llm.castform.com/v1"
    )
    monkeypatch.setenv("NEON_RAG_MODEL_BASE_URL", "https://models.example/v1")
    monkeypatch.setenv("NEON_RAG_MODEL_API_KEY", "external-key")

    config = RagDataModelConfig.from_env("NEON_RAG")

    assert config.base_url == "https://models.example/v1"
    assert isinstance(config.auth, StaticBearerAuth)


def test_model_config_rejects_external_endpoint_without_key(monkeypatch) -> None:
    monkeypatch.setattr(
        "castform.rag.example_data.config.llm_url", lambda: "https://llm.castform.com/v1"
    )
    monkeypatch.setenv("NEON_RAG_MODEL_BASE_URL", "https://models.example/v1")
    monkeypatch.delenv("NEON_RAG_MODEL_API_KEY", raising=False)

    with pytest.raises(ValueError, match="requires an explicit API key"):
        RagDataModelConfig.from_env("NEON_RAG")


def test_target_question_count_is_distributed_and_split_80_20(monkeypatch, tmp_path) -> None:
    driver = RagExampleData(name="test-rag", root=tmp_path, env_prefix="TEST_RAG")
    chunks = ChunkCollection(
        [Chunk(content=f"chunk {index}", metadata=(("file", f"{index}.md"),)) for index in range(3)]
    )
    config = RagDataModelConfig(
        base_url="https://models.example/v1",
        auth=StaticBearerAuth("test-key"),
    )
    counts: list[int] = []

    monkeypatch.setattr(
        "castform.rag.example_data.create_openai_client",
        lambda **kwargs: SimpleNamespace(close=lambda: None),
    )

    def generate_rows(client, model, chunk, *, count):
        del client, model
        counts.append(count)
        return [
            {
                "question": f"question {chunk.content} {index}",
                "answer": f"answer {index}",
                "reference_chunks": [chunk.to_dict()],
            }
            for index in range(count)
        ]

    monkeypatch.setattr("castform.rag.example_data._generate_rows", generate_rows)

    files = driver.generate_question_data(
        chunks,
        config,
        target_questions=12,
    )

    train_rows = [json.loads(line) for line in files["train.jsonl"].read_text().splitlines()]
    eval_rows = [json.loads(line) for line in files["eval.jsonl"].read_text().splitlines()]
    assert counts == [4, 4, 4]
    assert len(train_rows) == 10
    assert len(eval_rows) == 2


def test_target_question_count_invalidates_a_smaller_cached_dataset(monkeypatch, tmp_path) -> None:
    driver = RagExampleData(name="test-rag", root=tmp_path, env_prefix="TEST_RAG")
    driver.data_dir.mkdir()
    for path in driver.dataset_files().values():
        path.write_text("{}\n", encoding="utf-8")
    chunks = ChunkCollection(
        [Chunk(content=f"chunk {index}", metadata=(("file", f"{index}.md"),)) for index in range(2)]
    )
    config = RagDataModelConfig(
        base_url="https://models.example/v1",
        auth=StaticBearerAuth("test-key"),
    )
    calls: list[int] = []

    monkeypatch.setattr(
        "castform.rag.example_data.create_openai_client",
        lambda **kwargs: SimpleNamespace(close=lambda: None),
    )

    def generate_rows(client, model, chunk, *, count):
        del client, model, chunk
        calls.append(count)
        return [
            {
                "question": f"question {index}",
                "answer": f"answer {index}",
                "reference_chunks": [],
            }
            for index in range(count)
        ]

    monkeypatch.setattr("castform.rag.example_data._generate_rows", generate_rows)

    driver.generate_question_data(chunks, config, target_questions=4)

    assert calls == [2, 2]
