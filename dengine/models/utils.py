from pathlib import Path
import logging
from copy import deepcopy
from collections import defaultdict
from typing import List, Dict, Sequence, Any, Optional
from traceback import format_exc

import torch
from torch.utils.data import Subset
from torch import Tensor
from pickle import UnpicklingError

from dengine.config import constants
from dengine.interfaces import ClientInterface


def load_last_checkpoint(clients: Sequence[ClientInterface], checkpoint_dir: Path):
    logging.info(f"Trying to reload checkpoint {constants.LAST_CHECKPOINT_FILENAME} from {checkpoint_dir}")
    for c in clients:
        title = f"[Paiv-{c.UUID}]"
        last_checkpoint_fpath = checkpoint_dir / f"{c.UUID}/{constants.LAST_CHECKPOINT_FILENAME}"
        if not last_checkpoint_fpath.exists():
            logging.warning(
                f"{title} checkpoint file not found: {last_checkpoint_fpath}"
            )
            continue

        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            ckpt_state_dict = torch.load(last_checkpoint_fpath, map_location=torch.device(device))
            c.model.load_state_dict(ckpt_state_dict)
        except UnpicklingError as e:
            logging.error(
                f"{title} error when trying to reload the checkpoint: {last_checkpoint_fpath} \n"
                f"{e} \n"
                f"{format_exc()}"
            )
            continue

        logging.warning(f"{title} loaded checkpoint: {last_checkpoint_fpath}")


def model_wise_weighted_average(L: Sequence[Dict[str, Tensor]], alpha_list: Sequence[float]):
    '''
    Computes the weighted federated averaged of the models
    '''
    w_avg: Dict[str, Tensor] = deepcopy(L[0])
    for layer in w_avg.keys():
        w_avg[layer] = w_avg[layer].float() * alpha_list[0]

    for i in range(1, len(L)):
        ith_model = L[i]
        for layer in w_avg.keys():
            w_avg[layer] += ith_model[layer].float() * alpha_list[i]

    return w_avg


@torch.no_grad()
def unsafe_compute_layer_wise_weighted_average(
    model_state_dicts: Sequence[Dict[str, Tensor]],
    reference_dict: Dict[str, Tensor],
    alpha: Optional[Sequence[float]] = None
):
    '''
    Computes the weighted federated averaged of the models. Allow for models to be partial
    '''
    grouped_layers: Dict[str, List[Tensor]] = defaultdict(list)
    custom_layers_alphas: Dict[str, List[float]] = defaultdict(list)
    for i, ith_model in enumerate(model_state_dicts):
        for layer in ith_model.keys():
            if alpha is not None:
                custom_layers_alphas[layer].append(alpha[i])
            grouped_layers[layer].append(ith_model[layer].float().cpu())

    t_custom_layers_alphas = {k: torch.Tensor(v) / sum(v) for k, v in custom_layers_alphas.items()}

    updated_dict = deepcopy(reference_dict)
    for layer_name, weight_list in grouped_layers.items():
        if layer_name in t_custom_layers_alphas:
            layer_alpha = t_custom_layers_alphas[layer_name]
        else:
            layer_alpha = torch.Tensor(len(weight_list))
        stacked_weights = torch.stack(weight_list)
        layer_alpha_broadcast = layer_alpha.view(-1, *([1] * (stacked_weights.ndim - 1)))
        updated_dict[layer_name] = (stacked_weights * layer_alpha_broadcast).sum(dim=0)
    return updated_dict


def get_image_size(dataset: Subset) -> int:
    return dataset.dataset.data.size(1)  # type: ignore


def get_image_channels(dataset: Subset) -> int:
    dataset = dataset.dataset  # type: ignore
    if len(dataset.data.shape) == 3:  # type: ignore
        return 1
    else:
        return dataset.data.size(3)  # type: ignore


def get_unique_targets(dataset: Subset) -> Sequence[Any]:
    return dataset.dataset.targets.unique().tolist()  # type: ignore
