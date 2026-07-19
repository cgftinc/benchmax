from abc import abstractmethod
from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable

from benchmax.envs.shared_types import Example

__all__ = ["Dataset", "FrozenDataset"]


@runtime_checkable
class Dataset[Payload](Protocol):
    """Finite, deterministic dataset snapshot."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the exact number of examples in this snapshot."""

    @abstractmethod
    def __getitem__(self, index: int) -> Example[Payload]:
        """Return the deterministic example at ``index``."""

    @abstractmethod
    def __iter__(self) -> Iterator[Example[Payload]]: ...


class FrozenDataset[Payload]:
    """Immutable in-memory dataset backed by an ordered tuple of examples."""

    def __init__(self, examples: Sequence[Example[Payload]]) -> None:
        self._examples = tuple(examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> Example[Payload]:
        return self._examples[index]

    def __iter__(self) -> Iterator[Example[Payload]]:
        return iter(self._examples)
