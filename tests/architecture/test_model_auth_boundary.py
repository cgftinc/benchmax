from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).parents[2]
CASTFORM_SOURCE = ROOT / "packages" / "castform" / "src" / "castform"
QA_GENERATION_SOURCE = CASTFORM_SOURCE / "rag" / "qa_generation"
MODEL_RUNTIME_SOURCES = (
    CASTFORM_SOURCE / "model_auth.py",
    CASTFORM_SOURCE / "rag" / "corpus" / "embed.py",
)


def test_bootstrap_auth_stays_outside_model_and_environment_code() -> None:
    """CASTFORM_AUTH_TOKEN is limited to login/bootstrap control-plane code."""

    allowed = {
        CASTFORM_SOURCE / "cli" / "_auth.py",
        CASTFORM_SOURCE / "platform" / "credentials.py",
    }
    violations = [
        source_file
        for source_file in CASTFORM_SOURCE.rglob("*.py")
        if "CASTFORM_AUTH_TOKEN" in source_file.read_text(encoding="utf-8")
        and source_file not in allowed
    ]

    assert not violations, "\n".join(map(str, violations))


def test_retired_credential_resolvers_are_not_reintroduced() -> None:
    retired = ("resolve_judge_key", "platform_embed_fn")
    violations: list[str] = []

    for source_file in CASTFORM_SOURCE.rglob("*.py"):
        source = source_file.read_text(encoding="utf-8")
        for name in retired:
            if name in source:
                violations.append(f"{source_file}: contains retired {name}")

    assert not violations, "\n".join(violations)


def test_qa_generation_constructs_openai_clients_through_model_auth() -> None:
    """Keep QA-generation calls on the request-scoped ModelAuth factory."""

    violations: list[str] = []
    for source_file in QA_GENERATION_SOURCE.rglob("*.py"):
        tree = ast.parse(source_file.read_text(encoding="utf-8"), source_file)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if isinstance(function, ast.Name) and function.id in {
                "OpenAI",
                "AsyncOpenAI",
            }:
                violations.append(
                    f"{source_file}:{node.lineno}: constructs {function.id} directly"
                )

    assert not violations, "\n".join(violations)


def test_model_code_cannot_use_platform_bearer_resolvers() -> None:
    """Control/data-plane bearer resolvers are never model credentials."""

    forbidden = {"platform_bearer", "runtime_platform_bearer"}
    source_files = [
        *MODEL_RUNTIME_SOURCES,
        *QA_GENERATION_SOURCE.rglob("*.py"),
    ]
    violations: list[str] = []

    for source_file in source_files:
        tree = ast.parse(source_file.read_text(encoding="utf-8"), source_file)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in forbidden:
                violations.append(f"{source_file}:{node.lineno}: uses {node.id}")
            elif isinstance(node, ast.Attribute) and node.attr in forbidden:
                violations.append(f"{source_file}:{node.lineno}: uses {node.attr}")
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if alias.name in forbidden:
                        violations.append(
                            f"{source_file}:{node.lineno}: imports {alias.name}"
                        )

    assert not violations, "\n".join(violations)
