from typing import Optional, Callable, Dict, Type, TypeVar

from dengine.interfaces import ScenarioEngineInterface, ClientInterface


M = TypeVar("M", bound=ScenarioEngineInterface)

BUILTIN_SCENARIOS: Dict[str, Type[ScenarioEngineInterface]] = {}


def register_scenario(name: Optional[str] = None) -> Callable[[Type[M]], Type[M]]:
    def loader_wrapper(cls: Type[M]) -> Type[M]:
        key = name if name is not None else cls.__name__
        BUILTIN_SCENARIOS[key] = cls
        return cls
    return loader_wrapper


N = TypeVar("N", bound=ClientInterface)

BUILTIN_CLIENTS: Dict[str, Type[ClientInterface]] = {}


def register_client(name: Optional[str] = None) -> Callable[[Type[N]], Type[N]]:
    def loader_wrapper(cls: Type[N]) -> Type[N]:
        key = name if name is not None else cls.__name__
        BUILTIN_CLIENTS[key] = cls
        return cls
    return loader_wrapper
