from typing import Optional, Callable, Dict, Type, TypeVar

from dengine.interfaces import ModuleBase


TModule = TypeVar("TModule", bound=ModuleBase)

BUILTIN_MODELS: Dict[str, Type[ModuleBase]] = {}


def register_model(name: Optional[str] = None) -> Callable[[Type[TModule]], Type[TModule]]:
    def loader_wrapper(fn: Type[TModule]) -> Type[TModule]:
        key = name if name is not None else fn.__name__
        if key in BUILTIN_MODELS:
            raise ValueError(f"An entry is already registered under the name '{key}'.")
        BUILTIN_MODELS[key] = fn
        return fn
    return loader_wrapper
