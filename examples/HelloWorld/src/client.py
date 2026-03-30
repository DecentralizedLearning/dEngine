from dataclasses import dataclass

import torch

from dengine.dataset import SupervisedDataset
from dengine.callbacks import PeriodicCallback
from dengine import AbstractClient, MessageBase

from dengine.scenarios.decorators import register_client
from dengine.callbacks.decorators import register_callback


@dataclass
class CustomMessage(MessageBase):
    time: float
    normalized_contact_time: float


@register_client()
class CustomClient(AbstractClient[CustomMessage]):
    @torch.no_grad()
    def test(self, current_time: float, dataset: SupervisedDataset):
        pass

    @torch.no_grad()
    def aggregation(self, messages):
        pass


@register_callback()
class CustomCallback(PeriodicCallback):
    def on_local_training_end(self, *args, **kwargs):
        print("Done!")
        return super().on_local_training_end(*args, **kwargs)
