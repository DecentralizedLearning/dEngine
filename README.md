# dEngine 🍋 — A Decentralized Learning Framework
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DecentralizedLearning/dEngine/blob/main/examples/MNIST/dEngine-MNIST.ipynb)

**dEngine** is a Python simulation framework for research on **decentralized, federated, and gossip-style machine learning**. It lets you define a population of clients, a communication topology, a data partitioning strategy, and a training/aggregation scenario in a single YAML file, then run the whole simulation on one machine (CPU or GPU) with PyTorch.

It's built for researchers who want to iterate quickly on collaborative-learning algorithms (federated averaging, decentralized gossip averaging, and beyond) without re-implementing the simulation plumbing (network graphs, dataset partitioning, checkpointing, metrics) for every project.

## Why dEngine?

- **Config-driven experiments.** Every experiment — graph topology, dataset, partitioning, client behavior, aggregation rule — is declared in YAML and validated with [Pydantic](https://docs.pydantic.dev/) via [OmegaConf](https://omegaconf.readthedocs.io/). Swap a topology or a training strategy by pointing to a different config, no code changes required.
- **Pluggable scenarios.** Ships with reference implementations of **centralized**, **federated (FedAvg)**, **decentralized gossip averaging**, etc., plus an event-driven scenario API (`SyncEngine`, `Depends`, custom `Event`s) for building your own.
- **Realistic communication graphs.** Static and dynamic topologies out of the box (Erdős–Rényi, Barabási–Albert, complete graph, star, or your own via [NetworkX](https://networkx.org/)), including support for time-varying contact graphs.
- **Flexible data partitioning.** Built-in IID and non-IID partitioning strategies (including hub/non-hub variants) for splitting a dataset across clients.
- **Batteries included.** Reference datasets (MNIST, EMNIST, CIFAR-10/100) and models (MLP, small CNNs, ResNet), plus callbacks for checkpointing, logging, and metrics/analysis utilities (confusion matrices, partitioning plots, experiment reports).
- **Composable via `BUILTINS`.** Mix and match built-in configs (`dengine.BUILTINS.CORE...`) with your own custom config files and CLI overrides.

## Installation

dEngine requires **Python ≥ 3.13**.

```bash
pip install git+https://github.com/DecentralizedLearning/dEngine.git
```

This installs the `dengine` package along with its dependencies (PyTorch, torchvision, NetworkX, OmegaConf, Pydantic, scikit-learn, pandas, matplotlib, and others — see `pyproject.toml`).

## Quick start

The [`examples/`](./examples) directory contains runnable projects. The simplest is `examples/MNIST`, which trains a small CNN on MNIST over a Barabási–Albert graph using decentralized averaging.

A minimal experiment config looks like this:

```yaml
scenario:
  target: This_Will_Be_Overwritten
  arguments:
    max_communication_rounds: 200

graph:
  target: CustomNetworking

dataset:
  train:
    target: custom_mnist_train
  test:
    target: custom_mnist_test

partitioning:
  target: custom_partitioning
  arguments:
    validation_percentage: .2

client:
  target: CustomClient
  training_engine:
    target: CustomTrainingEngine
    arguments:
      epochs: 5
      lr: 0.001
      optimizer: SGD
      training_batch_size: 128
      validation_batch_size: 128
  local_model:
    target: CustomNet

callbacks:
- target: CustomCallback
  arguments:
    every_n_local_epochs: 5
```

You can also compose an experiment entirely from dEngine's built-in configs in code:

```python
import dengine
from dengine import load_experiment_from_yamls
from dengine.bin.simulation import run_simulation, SimulationArguments, VerbosityLevel
from dengine.config.utils import convert_to_nested_dict

args = SimulationArguments(
    gpus=[0],
    dataset_directory="datasets/",
    output_directory="logs/",
    dump_stdout=False,
    resume_checkpoints=False,
    sanity_check=False,
    seed=123,
    torch_num_threads=1,
    verbosity=VerbosityLevel.info,
)

cfg = load_experiment_from_yamls(
    [
        dengine.BUILTINS.CORE.GRAPH.BA_SMALL,
        dengine.BUILTINS.CORE.PARTITIONING.IID,
        dengine.BUILTINS.CORE.DATASETS.MNIST,
        dengine.BUILTINS.CORE.SCENARIOS.DECENTRALIZED_HOMOGENOUS,
    ],
    experiments_directory_root=str(args.output_directory),
    overrides=convert_to_nested_dict({"client.training_engine.arguments.epochs": 10}),
    seed=args.seed,
)

run_simulation(args, cfg)
```

Once installed, dEngine also exposes a console entry point:

```bash
dengine-simulate --config <your_config.yml> --gpu 0 --output_directory logs/
```

Run `dengine-simulate --help` (or `python main.py --help` from an example) to see all CLI options, including `--sanity_check` to validate a config without running it, `--seed`, `--dataset_directory`, and `--resume_checkpoints`.

More examples:

- [`examples/MNIST`](./examples/MNIST): minimal example of decentralized training on MNIST.
- [`examples/HelloWorld`](./examples/HelloWorld): implements a complex use case involving advanced customization.
- [`examples/Scaffold`](./examples/Scaffold): the SCAFFOLD federated optimization algorithm.
- [`examples/GAN`](./examples/GAN): decentralized training of a generative model.
- [`examples/events`](./examples/events/): advanced events API and distributed runtime

## Project structure

```
dEngine/
├── dengine/                  # The core library
│   ├── bin/                  # Simulation entry point & CLI argument parsing
│   ├── config/                # Experiment config loading, schemas, and BUILTINS
│   ├── graph/                 # Network topology abstractions (static & dynamic)
│   ├── dataset/                # Built-in datasets (MNIST, EMNIST, CIFAR-10/100)
│   ├── partitioning/           # IID / non-IID data partitioning strategies
│   ├── models/                 # Reference model architectures
│   ├── scenarios/               # Centralized, federated, decentralized & event-driven engines
│   ├── training_strategies/     # Local update strategies
│   ├── callbacks/                # Checkpointing, logging, and hooks
│   ├── analysis/                  # Post-hoc metrics, plots, and experiment reports
│   └── interfaces.py              # Core protocols/interfaces used across the framework
├── configs/                   # Built-in YAML configs (graphs, datasets, partitioning, scenarios)
├── examples/                   # Runnable example projects
├── tests/                       # Test suite
└── bin/                          # Auxiliary scripts (dataset normalization, schedule generation, etc.)
```

## Built-in configs

dEngine ships with ready-to-use configs under [`configs/core`](./configs/core), addressable through `dengine.BUILTINS`:

| Category | Options |
|---|---|
| **Graph** | Barabási–Albert (small/medium), Erdős–Rényi (small/medium), complete graph, star, centralized |
| **Dataset** | MNIST (full/reduced), CIFAR-10 (standard/fast), CIFAR-100 |
| **Partitioning** | IID, non-IID, IID balanced excluding hub, IID/non-IID hub mix |
| **Scenario** | Centralized, Federated, Decentralized (homogeneous), Decentralized (heterogeneous) |

## Related projects

- [`dengine-uciml`](https://github.com/DecentralizedLearning/dengine-uciml) — extends dEngine with support for the UCI Machine Learning datasets.
- [`notebooks`](https://github.com/DecentralizedLearning/notebooks) — notebooks for visualizing and analyzing dEngine experiment results.

## License

dEngine is released under the [MIT License](./LICENSE).

## Citing
As of today you can either cite one of these two papers:
* Sabella, Samuele, et al. "[Operating Regimes of Decentralized Learning Under Mobility and Bandwidth Constraints.](https://arxiv.org/abs/2606.28342)" arXiv preprint arXiv:2606.28342 (2026).
* Sabella, Samuele, et al. "[The Built-In Robustness of Decentralized Federated Averaging to Bad Data.](https://arxiv.org/pdf/2502.18097?)" 2025 International Joint Conference on Neural Networks (IJCNN). IEEE, 2025.

> **Note**:
> Full source code will be released soon as we finalize upcoming extensions

## Pull Request & Commit Guidelines
* **Linear History is mandatory**. Rebase!
* **One commit per file:** Keep commits atomic by limiting each commit to a single modified file.
* **Write clear commit titles:** Commit titles are important for tracking changes and generating release notes. Use a descriptive, imperative summary.

## AI-Assisted Contributions
Time is sacred and human learning/understanding is the final goal.

Given that nowadays LLMs have lowered the requirements of writing software, a few guidelines are needed to steer the project correctly. Specifically, while writing thousands of lines has become extremely cheap, reviewing them is extremely costly. Due to a lack of time/energy, I feel the need to filter contributions before starting a code review. Contributions should come from people who have real interest in the project.

Throughout these few lines, I will refer to **"blind contributions"** as PRs for the sake of contributing, which lack an underlying motivation or interest in the project.

* Blind contributions are not welcome.
* Low-hanging fruits, issues which are marked as entry level, are there to let newcomers understand the project and learn better how to interact with or extend the codebase.
* Before contributing to the project, ask yourself:
  - Do I have or am I building sufficient background to understand and appreciate this work?
  - Do I like working in decentralized learning?
  - Have I used, or do I plan to use, the repository for my own work?

  If the answers to these questions are positive, then please contact me personally via email (samuele.sabella@iit.cnr.it). The project is still basically a solo-developer project and sharing the experience in the field is extremely valuable.
* In case you do not agree with the policy, you are free to fork this repo and manage it as you like since it is distributed with an MIT license.

