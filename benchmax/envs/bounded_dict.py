from collections import OrderedDict
from typing import TypeVar, Generic

K = TypeVar('K')  # Key type
V = TypeVar('V')  # Value type

class BoundedDict(OrderedDict[K, V], Generic[K, V]):
    """
    A dictionary with a fixed maximum number of elements.

    Maintains insertion order and automatically removes the oldest entries
    when new items are added beyond the specified maximum length.

    Attributes:
        maxlen (int): The maximum number of items to retain in the dictionary.
    """

    maxlen: int

    def __init__(self, maxlen: int, *args: tuple[K, V], **kwargs: V):
        self.maxlen = maxlen
        super().__init__(*args, **kwargs)
        self._check_size()

    def __setitem__(self, key: K, value: V) -> None:
        if key in self:
            del self[key]
        super().__setitem__(key, value)
        self._check_size()

    def _check_size(self) -> None:
        while len(self) > self.maxlen:
            self.popitem(last=False)