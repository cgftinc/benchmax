"""Repository-wide validation conventions for maintained examples."""

import ast
from pathlib import Path


def test_examples_smoke_validate_the_required_train_split() -> None:
    repository_root = Path(__file__).parents[4]
    offenders = []
    for main_py in sorted((repository_root / "examples").glob("*/main.py")):
        tree = ast.parse(main_py.read_text(encoding="utf-8"))
        validation_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == "validate_environment")
                or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "validate_environment"
                )
            )
        ]
        splits = [
            keyword.value.value
            for call in validation_calls
            for keyword in call.keywords
            if keyword.arg == "split" and isinstance(keyword.value, ast.Constant)
        ]
        if validation_calls and splits != ["train"]:
            offenders.append(main_py.parent.name)

    assert offenders == []
