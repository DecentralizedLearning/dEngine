from pathlib import Path
import logging
from collections import defaultdict
from typing import List, Dict, Sequence, Any, Optional
from traceback import format_exc

import torch
from torch.utils.data import Subset
from torch import Tensor
from pickle import UnpicklingError

from dengine.config import constants
from dengine.interfaces import ClientInterface


def fast_state_dict_deepcopy(d: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return {k: v.detach().clone() for k, v in d.items()}


def _state_dict_device(state_dict: Dict[str, Tensor]):
    return next(iter(state_dict.values())).device


def load_last_checkpoint(clients: Sequence[ClientInterface], checkpoint_dir: Path):
    logging.info(f"Trying to reload checkpoint {constants.LAST_CHECKPOINT_FILENAME} from {checkpoint_dir}")
    for c in clients:
        title = f"[Client-{c.UUID}]"
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
    """
    Compute a weighted average of several complete state dicts.

    Every model must contain all the layers of the first model. Each layer of
    the output is the sum of the corresponding layers weighted by `alpha_list`.
    The weights are used as given and are not normalized, so they should
    usually sum to 1.

    Args:
        L: State dicts to average. All must share the keys of `L[0]`.
        alpha_list: One weight per model, in the same order as `L`.

    Returns:
        A new state dict with the averaged layers (float32 tensors on CPU).

    Raises:
        KeyError: If a model is missing a layer present in `L[0]`
    """
    w_avg: Dict[str, Tensor] = fast_state_dict_deepcopy(L[0])
    for layer in w_avg.keys():
        w_avg[layer] = w_avg[layer].float().cpu() * alpha_list[0]

    for i in range(1, len(L)):
        ith_model = L[i]
        for layer in w_avg.keys():
            w_avg[layer] += ith_model[layer].float().cpu() * alpha_list[i]

    return w_avg


@torch.no_grad()
def unsafe_layer_wise_weighted_average(
    model_state_dicts: Sequence[Dict[str, Tensor]],
    reference_dict: Dict[str, Tensor],
    alpha: Optional[Sequence[float]] = None
):
    """
    Compute a weighted average of several state dicts, layer by layer.

    Models may be partial: each layer is averaged only over the models that
    contain it, and the weights are renormalized over those models. Layers not
    present in any model keep their value from `reference_dict`.

    Args:
        model_state_dicts: State dicts to average. Each may contain a subset of layers.
        reference_dict: Full state dict used as the base for the output.
        alpha: One weight per model. If None, models are weighted equally.

    Returns:
        A copy of `reference_dict` with the averaged layers replaced
        (averaged tensors are float32 on CPU).
    """
    device = _state_dict_device(reference_dict)

    grouped_layers: Dict[str, List[Tensor]] = defaultdict(list)
    custom_layers_alphas: Dict[str, List[float]] = defaultdict(list)
    for i, ith_model in enumerate(model_state_dicts):
        for layer in ith_model.keys():
            if alpha is not None:
                custom_layers_alphas[layer].append(alpha[i])
            grouped_layers[layer].append(ith_model[layer].float().cpu())

    t_custom_layers_alphas = {k: torch.Tensor(v) / sum(v) for k, v in custom_layers_alphas.items()}
    updated_dict = fast_state_dict_deepcopy(reference_dict)
    for layer_name, weight_list in grouped_layers.items():
        if layer_name in t_custom_layers_alphas:
            layer_alpha = t_custom_layers_alphas[layer_name]
        else:
            layer_alpha = torch.ones(len(weight_list)) / len(weight_list)
        stacked_weights = torch.stack(weight_list)
        layer_alpha_broadcast = layer_alpha.view(-1, *([1] * (stacked_weights.ndim - 1)))
        updated_dict[layer_name] = (stacked_weights * layer_alpha_broadcast).sum(dim=0).to(device)
    return updated_dict


@torch.no_grad()
def unsafe_element_wise_weighted_average(
    model_state_dicts: Sequence[Dict[str, Tensor]],
    reference_dict: Dict[str, Tensor],
    alpha: Optional[Sequence[float]] = None
):
    """
    Compute a weighted average of several state dicts, entry by entry,
    ignoring NaN values.

    Models may be partial at two levels: a model can be missing whole layers,
    and within a layer, missing entries can be marked as NaN. Each entry is
    averaged only over the models that provide a non-NaN value for it, with
    the weights renormalized over those models. Entries that are NaN in every
    model that has the layer, and layers not present in any model, keep their
    value from `reference_dict`.

    Args:
        model_state_dicts: State dicts to average. Missing entries are NaN.
        reference_dict: Full state dict used as the base for the output. Must
            contain every layer that appears in `model_state_dicts`.
        alpha: One weight per model. Required; None is not supported yet.

    Returns:
        A copy of `reference_dict` with the averaged layers replaced
        (averaged tensors are float32 on CPU).

    Raises:
        NotImplementedError: If `alpha` is None.
        KeyError: If a layer in `model_state_dicts` is missing from
            `reference_dict`.
    """
    device = _state_dict_device(reference_dict)

    grouped_layers: Dict[str, List[Tensor]] = defaultdict(list)
    custom_layers_alphas: Dict[str, List[float]] = defaultdict(list)
    for i, ith_model in enumerate(model_state_dicts):
        for layer in ith_model.keys():
            if alpha is not None:
                custom_layers_alphas[layer].append(alpha[i])
            grouped_layers[layer].append(ith_model[layer].float().cpu())

    updated_dict = fast_state_dict_deepcopy(reference_dict)
    for layer_name, weight_list in grouped_layers.items():
        stacked_weights = torch.stack(weight_list)
        if layer_name in custom_layers_alphas:
            # 1. build the alpha tensor
            _layer_alpha = torch.Tensor(custom_layers_alphas[layer_name])
            _alpha_tensor = torch.zeros_like(torch.stack(weight_list))
            _alpha_tensor[:] = _layer_alpha.view(-1, *([1] * (_alpha_tensor.ndim - 1)))
            # 2. set to zero unreceived entries
            _alpha_tensor = torch.where(torch.isnan(stacked_weights), 0, _alpha_tensor)
            layer_alpha = _alpha_tensor / _alpha_tensor.sum(dim=0)
        else:
            raise NotImplementedError()
        updated_dict[layer_name] = torch.mul(stacked_weights.nan_to_num(0), layer_alpha).sum(dim=0)
        # If all models are nan for a layer-element, use the reference dict
        updated_dict[layer_name] = torch.where(
            updated_dict[layer_name].isnan().to(device),
            reference_dict[layer_name],
            updated_dict[layer_name].to(device)
        )
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
