"""
Preprocess a huggingface/benchmax dataset to a multiturn format suitable for a benchmax environment.
"""

import argparse
import logging
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Type
import asyncio
import inspect

from benchmax.envs.base_env import BaseEnv

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

# Set logging level to WARNING and above
logging.basicConfig(level=logging.WARNING)


def load_class(dotted_path: str) -> Type[BaseEnv]:
    """
    Load and return the class specified by `dotted_path`.
    Example: "benchmax.envs.wikipedia.wiki_env.WikipediaEnv"
    """
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
    except ValueError as exc:
        raise ImportError(
            f'"{dotted_path}" doesn\'t look like "package.module.Class"'
        ) from exc

    module: ModuleType = import_module(module_path)
    try:
        cls: Type[BaseEnv] = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f'Module "{module_path}" has no attribute "{class_name}"'
        ) from exc

    return cls


def get_canonical_class_name(cls: Type[BaseEnv]) -> str:
    """
    Get the canonical class name, removing local/skypilot prefix/suffix if the parent class
    has the same name without that prefix/suffix.
    """
    class_name = cls.__name__

    # Check for prefixes/suffixes to strip
    prefixes = ["local", "skypilot"]
    suffixes = ["local", "skypilot"]

    # Try to find a matching parent class without the prefix/suffix
    for base_cls in cls.__bases__:
        base_name = base_cls.__name__

        # Check if current class has prefix that base doesn't
        for prefix in prefixes:
            if class_name.lower().startswith(
                prefix
            ) and not base_name.lower().startswith(prefix):
                # Check if removing prefix gives us the base name
                stripped = class_name[len(prefix) :]
                if stripped == base_name:
                    return base_name

        # Check if current class has suffix that base doesn't
        for suffix in suffixes:
            if class_name.lower().endswith(suffix) and not base_name.lower().endswith(
                suffix
            ):
                # Check if removing suffix gives us the base name
                stripped = class_name[: -len(suffix)]
                if stripped == base_name:
                    return base_name

    # No matching parent found, return original name
    return class_name


async def get_system_prompt(cls: Type[BaseEnv]) -> str:
    """Setup env and get system prompt in async context."""
    # Initialize env with num_local_servers=1 if supported
    init_signature = inspect.signature(cls.__init__)
    if "num_local_servers" in init_signature.parameters:
        env = cls(num_local_servers=1)  # type: ignore
    else:
        env = cls()

    # Get system prompt (async function)
    prompt = await env.get_system_prompt(add_tool_defs=True)

    await env.shutdown()
    return prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir",
        required=True,
        help="Local directory where processed train/test parquet files will be written.",
    )
    parser.add_argument(
        "--dataset_name",
        required=True,
        help="Identifier of the HuggingFace dataset to load (e.g., 'squad', 'wikitext').",
    )
    parser.add_argument(
        "--env_path",
        required=True,
        help=(
            "Dotted path to the BaseEnv subclass to use for preprocessing, "
            "e.g. 'benchmax.envs.wikipedia.wiki_env.WikipediaEnv'."
        ),
    )

    args = parser.parse_args()

    print(f"Loading {args.dataset_name} dataset...", flush=True)
    benchmax_cls: Type[BaseEnv] = load_class(args.env_path)
    raw_dataset, dataset_path = benchmax_cls.load_dataset(args.dataset_name)

    from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

    if isinstance(raw_dataset, (IterableDataset, IterableDatasetDict)):
        raise TypeError(
            f"Iterable datasets are currently not supported. Got {type(raw_dataset).__name__}. "
        )

    if not isinstance(raw_dataset, (DatasetDict, Dataset)):
        raise TypeError(
            f"Expected DatasetDict or Dataset, but got {type(raw_dataset).__name__}."
        )

    print("Getting system prompt...", flush=True)
    system_prompt = asyncio.run(get_system_prompt(benchmax_cls))

    # Get canonical class name (strips local/skypilot if parent matches)
    canonical_name = get_canonical_class_name(benchmax_cls)

    def process_example(example):
        """Single mapping function that does all processing."""
        # First apply dataset-specific preprocessing
        standardized = benchmax_cls.dataset_preprocess(
            example, dataset_path=dataset_path
        )

        # Then format as multiturn prompt
        prompt = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": standardized["prompt"]},
        ]
        result = {
            **standardized,
            "prompt": prompt,
            "env_class": canonical_name,
            "data_source": canonical_name,
        }

        # Remove keys with None values
        result = {k: v for k, v in result.items() if v is not None}

        return result

    print("Processing examples...", flush=True)
    processed_dataset = raw_dataset.map(process_example)

    if isinstance(processed_dataset, DatasetDict) and set(
        processed_dataset.keys()
    ) == set(["train", "test"]):
        # If train and test dataset split already exist
        train_dataset = processed_dataset["train"]
        test_dataset = processed_dataset["test"]
    else:
        if isinstance(processed_dataset, DatasetDict):
            from datasets import concatenate_datasets

            processed_dataset = concatenate_datasets(
                [ds for ds in processed_dataset.values()]
            ).shuffle(seed=42)

        split = processed_dataset.train_test_split(
            test_size=0.2, seed=42, shuffle=False
        )
        train_dataset = split["train"]
        test_dataset = split["test"]

    print(f"Saving to {args.local_dir}...", flush=True)
    local_dir = Path(args.local_dir)
    train_dataset.to_parquet(local_dir / "train.parquet")
    test_dataset.to_parquet(local_dir / "test.parquet")
