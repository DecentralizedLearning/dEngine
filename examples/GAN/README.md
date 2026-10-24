# Generative Adversarial Networks in dEngine

This example implements the conditional GAN from [BKHMSI/MNIST-Latent-Interpolation](https://github.com/BKHMSI/MNIST-Latent-Interpolation?tab=readme-ov-file#cli-arguments). The GAN memorizes the training samples in an embedding layer, which makes it possible to sample interpolations between them. Pretrained GANs for MNIST and CIFAR100 are available at this [link](https://drive.google.com/drive/u/0/folders/12ndP-cc7rYXLiaZzagE7UceuVkRGFxSq).

> [!NOTE]
> This example uses the **old way of executing dEngine**: a bash script ([`benchmark.bash`](benchmark.bash)) that chains several `dengine-simulate` calls. `dengine-simulate` is very easy to use and is perfect for a single run, but when you need to execute **multiple experiments**, an advanced Python config like the one in the [MNIST example](../MNIST/README.md) is more convenient: experiments are defined in code, validated together with `--sanity_check`, filtered by name and run in parallel, and the whole setup lives in one versioned file, which makes **versioning** much easier.

## What's in this folder

| File | Content |
|------|---------|
| [`scenario.py`](scenario.py) | `GANLocalUpdate`, a training engine that alternates discriminator and generator steps, and `GANPaiv`, a `DecAvgClient` that averages the embedding layer, generator and discriminator separately. |
| [`callbacks.py`](callbacks.py) | `GANCallback`, which appends generator/discriminator losses to `round_G_loss.npy` / `round_D_loss.npy` and saves interpolation GIFs under `samples/<client UUID>/`. |
| [`configs/memnet_1M.yml`](configs/memnet_1M.yml) | Small `MemNet` (~1M parameters), used for MNIST. |
| [`configs/memnet_30M.yml`](configs/memnet_30M.yml) | Large `MemNet` (~30M parameters), used for CIFAR10 and CIFAR100. |
| [`benchmark.bash`](benchmark.bash) | Runs the three experiments, one `dengine-simulate` call each. |
| `__init__.py` | Imports `scenario.py` and `callbacks.py` so the custom classes get registered. |

## How it works

`dengine-simulate` composes an experiment from a list of YAML files passed to `--config`; later files override earlier ones. Each run in `benchmark.bash` stacks:

1. a built-in **dataset** config (`configs/core/datasets/mnist.yml`, `cifar10.yml` or `cifar100.yml`);
2. the built-in **centralized scenario** (`configs/core/scenarios/centralized.yml`);
3. the built-in **IID partitioning** (`configs/core/partitioning/iid.yml`);
4. the local **MemNet config**, which sets `client.target: GANPaiv`, `training_engine.target: GANLocalUpdate`, the `MemNet` architecture and the callbacks.

Any extra positional `key=value` argument is a dot-notation override applied on top, e.g. `name=mnist,GAN` or `client.training_engine.arguments.epochs=250`.

The custom classes are found because, at startup, `dengine-simulate` checks the current working directory for an `__init__.py` and imports it as a package. This is why **the script must be launched from `examples/GAN`**: otherwise `GANPaiv`, `GANLocalUpdate` and `GANCallback` are never registered, and the relative `../../configs` paths don't resolve.

## Running

From the repository root, with dEngine installed in an active virtualenv:

```bash
cd examples/GAN
SEED=42 bash benchmark.bash
```

The script runs three experiments in sequence:

| Experiment | Dataset | Model | Epochs |
|------------|---------|-------|--------|
| `mnist,GAN` | MNIST | `memnet_1M` | 250 |
| `cifar10,GAN` | CIFAR10 | `memnet_30M` | 500 |
| `cifa100,GAN` | CIFAR100 | `memnet_30M` | 500 |

All of them use a training batch size of 1024, no validation split and no data augmentation.

### Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SEED` | yes | — | Random seed passed to `--seed`. |
| `OUTPUT_DIRECTORY_PREFIX` | no | `logs` | Results are written to `$OUTPUT_DIRECTORY_PREFIX/GAN/<experiment name>/`. |
| `SANITY_CHECK` | no | unset | If set to any value, adds `--sanity_check`: configs are validated and nothing is trained. |

The script exits early if no virtualenv is active or `SEED` is missing.

Check the configurations before a long run:

```bash
SEED=42 SANITY_CHECK=1 bash benchmark.bash
```

### Running a single experiment

Since each experiment is a plain `dengine-simulate` call, you can also run one directly:

```bash
cd examples/GAN
dengine-simulate \
    --config ../../configs/core/datasets/mnist.yml \
    ../../configs/core/scenarios/centralized.yml \
    ../../configs/core/partitioning/iid.yml \
    configs/memnet_1M.yml \
    --seed 42 --verbosity debug \
    --output_directory logs/GAN \
    name=mnist,GAN \
    client.training_engine.arguments.epochs=250
```

## Outputs

For each experiment, under `$OUTPUT_DIRECTORY_PREFIX/GAN/<experiment name>/`:

- `config.yaml` — the fully resolved configuration;
- `samples/<client UUID>/*.gif` — latent interpolations between consecutive classes (`from=<i>,to=<i+1>,<n>.gif`), refreshed every 5 local epochs;
- metrics folder → `<client UUID>/round_G_loss.npy` and `round_D_loss.npy` — generator and discriminator losses;
- model checkpoints, dumped every 5 local epochs by `ModelDumpOnEpochEndCallback`.

## Moving to the config-driven approach

The bash script works well for three runs, but it scales poorly: shared overrides live in shell strings, experiments can only run sequentially, there is no way to select a subset without editing the file, and config errors surface only when each run starts. For new experiments, follow the [MNIST example](../MNIST/main.py) instead, which builds every `ExperimentConfiguration` in Python with `load_experiment_from_yamls`:

```python
from pathlib import Path
from dengine import load_experiment_from_yamls
from dengine.config.builtins import BUILTINS
from dengine.config.utils import convert_to_nested_dict

config = load_experiment_from_yamls(
    files=[
        BUILTINS.CORE.DATASETS.MNIST,
        BUILTINS.CORE.SCENARIOS.CENTRALIZED,
        BUILTINS.CORE.PARTITIONING.IID,
        Path("configs/memnet_1M.yml"),
    ],
    overrides=convert_to_nested_dict({
        "client.training_engine.arguments.epochs": 250,
        "client.training_engine.arguments.training_batch_size": 1024,
        "partitioning.arguments.validation_percentage": 0,
    }),
    experiments_directory_root="logs/GAN",
    seed=42,
)
config.name = "mnist,GAN"
```

With that structure you get, for free:

- **one versioned source of truth** — every experiment and override is in a Python file you can diff and review;
- **`--sanity_check`** over all experiments at once, with resolved configs dumped to JSON;
- **`--experiment_name_filter`** to run a subset by regex;
- **`--concurrent_runs`** to run experiments in parallel processes.
