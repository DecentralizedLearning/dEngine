import PIL.Image as pil

from typing import Tuple, Any
from torch import Tensor
from torchvision.datasets import VisionDataset


class SupervisedDataset(VisionDataset):
    def __init__(
        self,
        data: Tensor,
        targets: Tensor,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.data = data
        self.targets = targets
        self._pil_read_mode = 'L' if len(data.shape) < 4 else 'RGB'

    @property
    def unique_targets(self) -> Tensor:
        return self.targets.unique()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        img = pil.fromarray(img.numpy(), mode=self._pil_read_mode)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)
