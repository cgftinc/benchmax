from collections.abc import Iterator, Sequence

from benchmax.envs.shared_types import Example

__all__ = ["Dataset"]


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
