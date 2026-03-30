from typing import Optional, Callable, Dict, TypeVar, Type

from dengine.interfaces import LocalTrainingEngineInterface


GenericTrainingEngine = TypeVar("GenericTrainingEngine", bound=LocalTrainingEngineInterface)

BUILTIN_TRAINING_ENGINES: Dict[str, Type[LocalTrainingEngineInterface]] = {}


def register_local_training(name: Optional[str] = None) -> Callable[[Type[GenericTrainingEngine]], Type[GenericTrainingEngine]]:
    def loader_wrapper(fn: Type[GenericTrainingEngine]) -> Type[GenericTrainingEngine]:
        key = name if name is not None else fn.__name__
        if key in BUILTIN_TRAINING_ENGINES:
            raise ValueError(f"An entry is already registered under the name '{key}'.")
        BUILTIN_TRAINING_ENGINES[key] = fn
        return fn
    return loader_wrapper
