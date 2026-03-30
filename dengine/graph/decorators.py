from typing import Optional, Callable, Dict, TypeVar, Type

from .graph import Graph

M = TypeVar("M", bound=Graph)


BUILTIN_GRAPHS: Dict[str, Type[Graph]] = {}


def register_graph(name: Optional[str] = None) -> Callable[[Type[M]], Type[M]]:
    def loader_wrapper(cls: Type[M]) -> Type[M]:
        key = name if name is not None else cls.__name__
        BUILTIN_GRAPHS[key] = cls
        return cls
    return loader_wrapper
