from __future__ import annotations
from collections import UserDict, defaultdict

from collections.abc import MutableSet
from copy import deepcopy
from typing import Dict, Iterator, Iterable, TypeVar, Union


T = TypeVar('T')

class IdentitySet(MutableSet[T]):
    def __init__(self, iterable: Iterable[T] = ()) -> None:
        self.map: Dict[int, T] = {}
        self.extend(iterable)
    
    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return '(' + ', '.join(map(str, self.map.values())) + ')'

    def __len__(self) -> int:
        return len(self.map)

    def __iter__(self) -> Iterator[T]:
        return iter(self.map.values())

    def __contains__(self, x: T) -> bool:
        return id(x) in self.map
    
    def __deepcopy__(self, memo: dict) -> IdentitySet[T]:
        new = type(self)()
        memo[id(new)] = new
        for v in self.map.values():
            new._add(deepcopy(v, memo))
        return new
    
    def __sub__(self, other: Iterable[T]) -> IdentitySet[T]:
        res = type(self)(self)
        for item in other:
            res.discard(item)
        return res

    def add(self, value: T) -> None:
        self._add(value)

    def _add(self, value: T) -> None:
        self.map[id(value)] = value

    def discard(self, value: T) -> None:
        self.map.pop(id(value), None)

    def extend(self, other: Iterable[T]) -> None:
        for item in other:
            self._add(item)
    
    def union(self, other: Iterable[T]) -> IdentitySet[T]:
        res = type(self)(self)
        res.extend(other)
        return res

    def intersection(self, other: Iterable[T]) -> IdentitySet[T]:
        res = type(self)()
        for item in other:
            if item in self:
                res._add(item)
        return res


class IdentityFrozenSet(IdentitySet[T]):
    def add(self, value: T) -> None:
        raise NotImplementedError()

    def __hash__(self) -> int:
        return tuple(self.map.keys()).__hash__()


class IdentityDefaultdict:
    def __init__(self, *args, **kwargs) -> None:
        self._map = defaultdict(*args, **kwargs)
        self._keys = {}

    def __getitem__(self, key):
        self._keys[id(key)] = key
        return self._map[id(key)]

    def __setitem__(self, key, value):
        self._keys[id(key)] = key
        self._map[id(key)] = value
   
    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return '{' + ', '.join(f'{self._keys[k]}: {v}' for k, v in self._map.items()) + '}'

    def items(self):
        for k, v in self._map.items():
            yield (self._keys[k], v)

    def keys(self):
        return self._keys.values()

    def values(self):
        return self._map.values()

