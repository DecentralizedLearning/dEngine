from copy import deepcopy
from typing import List, Dict
from warnings import deprecated

import torch
from torch import nn

from .decentralized import VanillaDecentralizedMessage
from .decorators import register_client
from .decentralized import DecAvgClient, TGenericVanillaDecentralizedMessage, model_wise_weighted_average


def _dec_diff_update(
    local_model: nn.Module,
    models_state_dict_average: Dict,
    p_norm=2
):
    if len(models_state_dict_average) == 0:
        return local_model.state_dict()

    m = deepcopy(local_model.state_dict())
    m = {key: value.cpu() for key, value in m.items()}
    for k in models_state_dict_average.keys():
        # FedDiff update rule: w_local  = w_local - (w_local - w_avg)/||w_local - w_avg||_2
        dist = m[k] - models_state_dict_average[k]
        lp_dist = torch.norm(dist, p=p_norm) + 1
        m[k] = m[k] - (dist) / (lp_dist)

    return m


@register_client()
class DecDiffClient(DecAvgClient[TGenericVanillaDecentralizedMessage]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._include_myself:
            raise ValueError("DecDiff update rule does not include myself, ensure client.arguments.include_myself=False")

    def aggregation(self, messages: List[TGenericVanillaDecentralizedMessage]):
        self.raise_on_non_neighbor_messages(messages)
        if len(messages) == 0:
            return

        if self._include_myself:
            messages.append(
                VanillaDecentralizedMessage(time=messages[0].time, source_client=self)  # type: ignore
            )

        model_avg = model_wise_weighted_average(
            [m.source_client.model.state_dict() for m in messages],
            self._get_alphas(messages)
        )

        model_update = _dec_diff_update(
            local_model=self.model,
            models_state_dict_average=model_avg,
        )
        self.model.load_state_dict(model_update)


@deprecated("This class has been refactored to CentralizedClient, please consider refactoring.")
@register_client()
class DecDiffPaiv(DecDiffClient):
    def __init__(self, *args, test_batch_size: int, **kwargs):
        super().__init__(*args, **kwargs)
