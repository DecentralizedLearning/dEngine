from typing import Tuple, Any, List, Optional
import threading
from dataclasses import dataclass

import torch

from torch import Tensor
from torchvision.datasets import VisionDataset


@dataclass
class _Cache:
    data: Tensor
    targets: Tensor
    samples: Tensor
    labels: List[Any]


class SupervisedDataset(VisionDataset):
    def __init__(
        self,
        data: Tensor,
        targets: Tensor,
        use_cache: bool = False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        # Avoids multiple workers trying to build the cache together
        self._cache_lock = threading.Lock()
        self.data = data
        self.targets = targets

        self._use_cache = use_cache
        self._cache: Optional[_Cache] = None

    @property
    def unique_targets(self) -> Tensor:
        return self.targets.unique()

    @property
    def _cache_needs_rebuild(self) -> bool:
        return (
            self._cache is None
            or self._cache.data is not self.data
            or self._cache.targets is not self.targets
        )

    @property
    def cache(self):
        if self._cache_needs_rebuild:
            with self._cache_lock:
                if self._cache_needs_rebuild:
                    data, targets = self.data, self.targets
                    samples = torch.stack([self._process_img(img) for img in data])
                    labels = [self._process_target(int(t)) for t in targets.tolist()]
                    self._cache = _Cache(data, targets, samples, labels)
        return (self._cache.samples, self._cache.labels)

    def _process_img(self, img: Tensor) -> Any:
        if self.transform is not None:
            img = self.transform(img)
        return img

    def _process_target(self, target: int) -> Any:
        if self.target_transform is not None:
            target = self.target_transform(target)
        return target

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self._use_cache:
            samples, labels = self.cache
            return samples[index], labels[index]

        img, target = self.data[index], int(self.targets[index])
        return self._process_img(img), self._process_target(target)

    def __len__(self) -> int:
        return len(self.data)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_cache_lock"]
        state["_cache"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cache_lock = threading.Lock()
