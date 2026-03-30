from typing import Optional, Callable, Dict, Type, TypeVar

from dengine.interfaces import ClientCallbackInterface


M = TypeVar("M", bound=ClientCallbackInterface)

BUILTIN_CALLBACKS: Dict[str, Type[ClientCallbackInterface]] = {}


def register_callback(name: Optional[str] = None) -> Callable[[Type[M]], Type[M]]:
    def loader_wrapper(cls: Type[M]) -> Type[M]:
        key = name if name is not None else cls.__name__
        if key in BUILTIN_CALLBACKS:
            raise ValueError(f"An entry is already registered under the name '{key}'.")
        BUILTIN_CALLBACKS[key] = cls
        return cls
    return loader_wrapper
