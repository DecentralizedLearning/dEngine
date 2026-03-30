from typing import Optional, Callable, Tuple, Protocol, Dict

from .dataset import SupervisedDataset


class TDatasetLoader(Protocol):
    def __call__(
        self,
        *args,
        **kwargs
    ) -> SupervisedDataset:
        ...


BUILTIN_DATASETS: Dict[str, TDatasetLoader] = {}


def register_dataset(name: Optional[str] = None) -> Callable[[TDatasetLoader], TDatasetLoader]:
    def loader_wrapper(fn: TDatasetLoader) -> TDatasetLoader:
        key = name if name is not None else fn.__name__
        if key in BUILTIN_DATASETS:
            raise ValueError(f"An entry is already registered under the name '{key}'.")
        BUILTIN_DATASETS[key] = fn
        return fn
    return loader_wrapper


class TPostProcessing(Protocol):
    def __call__(
        self,
        dataset: SupervisedDataset,
        test: SupervisedDataset,
        *args,
        **kwargs
    ) -> Tuple[SupervisedDataset, SupervisedDataset]:
        ...


BUILTIN_POSTPROCESSING: Dict[str, TPostProcessing] = {}


def register_postprocessing(name: Optional[str] = None) -> Callable[[TPostProcessing], TPostProcessing]:
    def loader_wrapper(fn: TPostProcessing) -> TPostProcessing:
        key = name if name is not None else fn.__name__
        if key in BUILTIN_POSTPROCESSING:
            raise ValueError(f"An entry is already registered under the name '{key}'.")
        BUILTIN_POSTPROCESSING[key] = fn
        return fn
    return loader_wrapper
