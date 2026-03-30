from typing import Optional, Callable, Dict, Protocol

from dengine.dataset import SupervisedDataset
from dengine.partitioning import TYPE_DATASET_PARTITIONING
from dengine.graph import Graph


class TPartitioning(Protocol):
    def __call__(
        self,
        dataset: SupervisedDataset,
        test: SupervisedDataset,
        partitions: TYPE_DATASET_PARTITIONING,
        graph: Graph,
        *args,
        **kwargs
    ) -> TYPE_DATASET_PARTITIONING:
        ...


BUILTIN_PARTITIONING: Dict[str, TPartitioning] = {}


def register_partitioning(name: Optional[str] = None) -> Callable[[TPartitioning], TPartitioning]:
    def loader_wrapper(fn: TPartitioning) -> TPartitioning:
        key = name if name is not None else fn.__name__
        if key in BUILTIN_PARTITIONING:
            raise ValueError(f"An entry is already registered under the name '{key}'.")
        BUILTIN_PARTITIONING[key] = fn
        return fn
    return loader_wrapper
