from collections.abc import Iterator, Sequence

from benchmax.envs.shared_types import Example

__all__ = ["Dataset", "validate_max_examples"]


def validate_max_examples(max_examples: int | None) -> int | None:
    """Validate the optional source-construction limit shared by environments."""

    if max_examples is None:
        return None
    if (
        isinstance(max_examples, bool)
        or not isinstance(max_examples, int)
        or max_examples <= 0
    ):
        raise ValueError("max_examples must be a positive integer or None")
    return max_examples


class Dataset[Payload]:
    """Fixed, ordered collection of examples.

    The collection snapshots its membership and order at construction time. Example
    payloads are stored as provided and are not made deeply immutable.
    """

    def __init__(self, examples: Sequence[Example[Payload]]) -> None:
        self._examples = tuple(examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> Example[Payload]:
        return self._examples[index]

    def __iter__(self) -> Iterator[Example[Payload]]:
        return iter(self._examples)
