#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Type
import importlib
import torch
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, ConcatDataset


def MeanAndStd(
    precision: torch.dtype,
    dataset: Type[VisionDataset],
    batch_size: int,
    output_path: Path,
    **kwargs
) -> torch.Tensor:
    torch.set_default_dtype(precision)
    mean = 0
    std = 0
    nb_samples = 0

    dt = ConcatDataset([
        dataset(
            root=output_path,
            transform=ToTensor(),
            train=True,
            download=True,
            **kwargs
        ),
        dataset(
            root=output_path,
            transform=ToTensor(),
            train=False,
            download=True,
            **kwargs
        )
    ])  # type: ignore
    loader = DataLoader(dt, batch_size=batch_size, shuffle=False)

    for X, _ in loader:
        bs, ch, _, _ = X.shape
        X_flat = X.view(bs, ch, -1)
        mean += X_flat.sum(0)
        nb_samples += bs
    mean = torch.mean(mean, dim=-1) / nb_samples

    for X, _ in loader:
        bs, ch, _, _ = X.shape
        bs_mean = torch.stack([mean] * bs).unsqueeze(-1)
        X_flat = X.view(bs, ch, -1)
        std += ((X_flat - bs_mean) ** 2).sum(0)
    std = torch.mean(std / nb_samples, dim=-1).sqrt()
    return mean, std


def main():
    parser = argparse.ArgumentParser(description="Compute dataset statistics (mean and std).")
    parser.add_argument("--precision", type=str, default="float16")
    parser.add_argument("--dataset", type=str, default="CIFAR100")
    parser.add_argument("--dataset_output_path", type=str, default="datasets/")
    parser.add_argument("--batch_size", type=int, default=256)
    args, unknown_args = parser.parse_known_args()
    extra_args = dict([ar.split('=') for ar in unknown_args])

    try:
        precision = getattr(torch, args.precision)
    except AttributeError:
        raise ValueError(f"Invalid precision: {args.precision}. Use 'float16', 'float32', or 'float64'.")

    try:
        dataset_module = importlib.import_module("torchvision.datasets")
        dataset_cls = getattr(dataset_module, args.dataset)
    except (ImportError, AttributeError):
        raise ValueError(f"Dataset {args.dataset} not found in torchvision.datasets.")

    output_path = Path(args.dataset_output_path)

    mean, std = MeanAndStd(precision, dataset_cls, args.batch_size, output_path, **extra_args)
    print(f"mean: {mean}\nstd: {std}")


if __name__ == "__main__":
    main()
