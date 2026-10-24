# MNIST example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DecentralizedLearning/dEngine/blob/main/examples/MNIST/dEngine-MNIST.ipynb)

This example compares centralized, federated and decentralized training on MNIST. It builds **8 experiments**: 4 scenarios × 2 architectures.

| Scenario      | Topology                         | Clients | Aggregation  | Comm. rounds |
|---------------|----------------------------------|---------|--------------|--------------|
| `Centralized` | —                                | 1       | —            | —            |
| `FedAvg`      | Star                             | 50      | FedAvg       | 20           |
| `BA,DecAvg`   | Barabási–Albert (`m=1`)          | 50      | DecAvg       | 200          |
| `ER,DecAvg`   | Erdős–Rényi (`p=0.2`)            | 50      | DecAvg       | 200          |

Each scenario is run with two models: `CNNMnist` and `WideResNetClassifier` (depth 10, widen factor 5, dropout 0.3).

All experiments use an IID balanced partitioning of MNIST (20% validation split) and share the callbacks in [`configs/core.yml`](configs/core.yml) (loss and confusion-matrix dumps). The centralized run uses [`configs/centralized_callbacks.yml`](configs/centralized_callbacks.yml) instead, which dumps the confusion matrix at the end of every epoch.

Experiment names follow the pattern `dataset,[topology,]algorithm,model`, e.g. `mnist,BA,DecAvg,CNNMnist`.

## Running

Run all commands from this directory (`examples/MNIST`), since the configs are loaded through relative paths.

### 1. Check the configurations

```bash
python main.py --sanity_check
```

This validates every experiment without training and dumps the resolved configs to `experiments_configs.json`:

```
🟢 Configurations are fine. Dumped the following configs to experiments_configs.json.
Found 8 configs:
- mnist,Centralized,CNNMnist
- mnist,BA,DecAvg,CNNMnist
- mnist,ER,DecAvg,CNNMnist
- mnist,FedAvg,CNNMnist
- mnist,Centralized,WideResNetClassifier
- mnist,BA,DecAvg,WideResNetClassifier
- mnist,ER,DecAvg,WideResNetClassifier
- mnist,FedAvg,WideResNetClassifier
Ready to run...
```

### 2. Run a single experiment

```bash
python main.py --experiment_name_filter=mnist,BA,DecAvg,CNNMnist
```

`--experiment_name_filter` is a regular expression and can be repeated; every experiment matching at least one pattern is selected. The script lists the matches and waits for `ENTER` before starting. Some examples:

```bash
python main.py --experiment_name_filter=CNNMnist         # the 8 CNN experiments
python main.py --experiment_name_filter=FedAvg \
               --experiment_name_filter=Centralized      # FedAvg + Centralized, both models
```

### 3. Run several experiments in parallel

```bash
python main.py --concurrent_runs 8
```

Each experiment runs in its own process; `--concurrent_runs` sets how many run at the same time (default 1). Note the flag uses a hyphen.

### 4. Run clients in parallel (free-threaded Python)

> [!IMPORTANT]
> This requires a free-threaded (GIL-free) build of Python, e.g. `python3.13t` or later.

```bash
PYTHON_GIL=0 python main.py --experiment_name_filter=mnist,BA,DecAvg,CNNMnist --num_parallel_clients 50
```

Within a single experiment, `--num_parallel_clients` trains up to that many clients concurrently on separate threads.

## Useful options

| Option                         | Default      | Description                                                  |
|--------------------------------|--------------|--------------------------------------------------------------|
| `--sanity_check`               | off          | Validate and dump configs, no training.                      |
| `--experiment_name_filter`     | all          | Regex on experiment names (repeatable).                      |
| `--concurrent_runs N`          | `1`          | Number of experiments run in parallel.                       |
| `--debug`                      | off          | Skip training and testing; useful to test the pipeline fast. |
| `--gpu`                        | `0`          | GPU ids, comma separated; `-1` for CPU.                      |
| `--seed`                       | random       | Random seed (also used to generate the graphs).              |
| `--dataset_directory`          | `./datasets` | Where MNIST is downloaded.                                   |
| `--output_directory`           | `./logs`     | Where logs, metrics and checkpoints are written.             |
| `--resume_checkpoints`         | off          | Load existing checkpoints before training.                   |
| `--verbosity`                  | `silent`     | `silent`, `debug` or `info`.                                 |

Run `python main.py --help` for the full list.

## Customizing the experiments

Experiments are defined in [`main.py`](main.py) by composing built-in configs (`BUILTINS.CORE...`) with the local YAML files, then applying dot-notation overrides. To change a hyperparameter, edit the override dictionaries at the top of the file:

- `DECENTRALIZED_CONFIG_OVERRIDES` — BA and ER experiments
- `FEDERATED_CONFIGS_OVERRIDES` — FedAvg experiment
- `CENTRALIZED_CONFIG_OVERRIDES` — centralized experiment
- `CNN_MNIST`, `WIDE_RESNET` — model architectures

For example, to train decentralized clients with Adam for 100 rounds:

```python
DECENTRALIZED_CONFIG_OVERRIDES = {
    ...
    "scenario.arguments.max_communication_rounds": 100,
    "client.training_engine.arguments.optimizer": "Adam",
    "client.training_engine.arguments.lr": 0.0003,
}
```

Run `--sanity_check` again afterwards and inspect `experiments_configs.json` to confirm the resolved values.

The [notebook](dEngine-MNIST.ipynb) walks through the same example interactively.
