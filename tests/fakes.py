"""Minimal stand-ins for real models and training engines.

They are registered like built-ins, so scenarios can build them from a
normal ``ClientModuleConfig`` without training anything.
"""
from typing import Any, Dict, List, Literal, Optional

import torch
from torch import Tensor, nn
from torch.utils.data import Subset

from dengine.callbacks import DummyCallback
from dengine.dataset import SupervisedDataset
from dengine.interfaces import ClientCallbackInterface, LocalTrainingEngineInterface, ModuleBase
from dengine.models.decorators import register_model
from dengine.training_strategies.decorators import register_local_training


@register_model()
class ScalarModel(ModuleBase):
    """A model with a single scalar weight, so averages are easy to check by hand."""

    def __init__(self, dataset: Subset, *args: Any, **kwargs: Any):
        super().__init__(dataset)
        self.weight = nn.Parameter(torch.zeros(1))


@register_local_training()
class RecordingTrainingEngine(LocalTrainingEngineInterface):
    """Does not train: it only records when training and testing were requested."""

    def __init__(self, *args: Any, callback: Optional[ClientCallbackInterface] = None, **kwargs: Any):
        self._callback: ClientCallbackInterface = callback or DummyCallback()
        self.train_calls: List[float] = []
        self.test_calls: int = 0

    @property
    def callback(self) -> ClientCallbackInterface:
        return self._callback

    @callback.setter
    def callback(self, value: ClientCallbackInterface):
        self._callback = value

    def train(self, model: ModuleBase, current_time: float) -> ModuleBase:
        self.train_calls.append(current_time)
        return model

    def compute_loss_in_production(
        self,
        model: ModuleBase,
        data: SupervisedDataset,
    ) -> Dict[Literal["output", "loss"], Tensor]:
        self.test_calls += 1
        return {"output": torch.empty(0), "loss": torch.empty(0)}
