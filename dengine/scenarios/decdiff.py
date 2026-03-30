from copy import deepcopy
from typing import List, Dict
import logging
from warnings import deprecated

import torch
from torch import nn

from .decentralized import VanillaDecentralizedMessage
from .decorators import register_client
from .decentralized import DecAvgClient, TGenericVanillaDecentralizedMessage, _client_wise_alpha_weighted_avg


def _dec_diff_update(
    local_model: nn.Module,
    models_state_dict_average: Dict,
    p_norm=2
):
    if len(models_state_dict_average) == 0:
        return local_model.state_dict()

    m = deepcopy(local_model.state_dict())
    for k in models_state_dict_average.keys():
        # FedDiff update rule: w_local  = w_local - (w_local - w_avg)/||w_local - w_avg||_2
        dist = m[k] - models_state_dict_average[k]
        lp_dist = torch.norm(dist, p=p_norm) + 1
        m[k] = m[k] - (dist) / (lp_dist)

    return m


@register_client()
class DecDiffClient(DecAvgClient[TGenericVanillaDecentralizedMessage]):
    def __init__(
        self,
        include_myself: bool = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._include_myself = include_myself

    def aggregation(self, messages: List[TGenericVanillaDecentralizedMessage]):
        messages = self.filter_message_from_neighborhood(messages)
        if len(messages) == 0:
            return

        message_origins = [_p.source_client.UUID for _p in messages]
        message_times = [_p.time for _p in messages]
        origins_with_times = [f"{src} at {t}" for src, t in zip(message_origins, message_times)]
        logging.info(
            f"PAIV-{self.UUID}: Aggregating {len(messages)} models. "
            f"Sources/Timestamps: [{', '.join(origins_with_times)}]"
        )

        if self._include_myself:
            messages.append(
                VanillaDecentralizedMessage(time=messages[0].time, source_client=self)  # type: ignore
            )

        model_avg = _client_wise_alpha_weighted_avg(
            messages,
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
