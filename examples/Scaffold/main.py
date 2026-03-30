import sys
from typing import Tuple
from pathlib import Path
import os
import re
from argparse import ArgumentParser
import signal
import torch.multiprocessing as mp
import json

from tqdm import tqdm
from pydantic import ValidationError

from dengine.config import load_experiment_from_yamls
from dengine.bin.args_parser import cli_argument_parser
from dengine.scenarios.federated import FederatedClient, VanillaFederatedScenario
from dengine.scenarios.decdiff import DecDiffClient
from dengine.scenarios.decentralized import DecAvgClient
from dengine.config.builtins import BUILTINS
from dengine.bin import SimulationArguments
from dengine.config import ExperimentConfiguration
from dengine.bin.simulation import load_engine
from dengine.config.utils import convert_to_nested_dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Scaffold.scaffold import *  # noqa


# ..... ..... ..... ..... ..... ..... ..... ..... #
# EXPERIMENTS
# ..... ..... ..... ..... ..... ..... ..... ..... #
DECENTRALIZED_CONFIGS = [BUILTINS.CORE.SCENARIOS.DECENTRALIZED_HOMOGENOUS, Path("configs/core_decentralized.yml")]
FEDERATED_CONFIGS = [BUILTINS.CORE.SCENARIOS.FEDERATED, Path("configs/core_decentralized.yml")]
SCAFFOLD_CONFIGS = [Path("configs/scaffold.yml"), Path("configs/core_decentralized.yml")]

MNIST_BA_CONFIGS = [BUILTINS.CORE.GRAPH.BA_MEDIUM, BUILTINS.CORE.DATASETS.MNIST, BUILTINS.CORE.PARTITIONING.IID]
MNIST_ER_CONFIGS = [BUILTINS.CORE.GRAPH.ER_MEDIUM, BUILTINS.CORE.DATASETS.MNIST, BUILTINS.CORE.PARTITIONING.IID]

CONFIG_OVERRIDES = {
    "client.training_engine.arguments.patience": 100,
    "client.training_engine.arguments.epochs": 5,
    "client.training_engine.arguments.validation_batch_size": 32,
    "client.training_engine.arguments.training_batch_size": 32,
    "scenario.arguments.max_communication_rounds": 200,
}
MNIST_CONFIG_OVERRIDES = {
    **CONFIG_OVERRIDES,
    "partitioning.arguments.validation_percentage": 0.2
}

DECAVG_CONFIG_OVERRIDES = {
    "client.target": DecAvgClient.__name__,
    "client.arguments.use_weighted_avg": True
}
FEDERATED_CONFIG_OVERRIDES = {
    "client.target": FederatedClient.__name__,
    "scenario.target": VanillaFederatedScenario.__name__
}
DECDIFF_CONFIG_OVERRIDES = {
    "client.target": DecDiffClient.__name__,
    "client.arguments.use_weighted_avg": True
}
SCAFFOLD_CONFIGS_OVERRIDES = {
    "client.target": "ScaffoldClient",
    "client.training_engine.target": "ScaffoldLocalUpdate"
}


def load_mnist_configurations(simulation_args: SimulationArguments):
    return [
        load_experiment_from_yamls(
            files=[*DECENTRALIZED_CONFIGS, *MNIST_BA_CONFIGS],
            overrides=convert_to_nested_dict({
                "name": "mnist,BA,DecAvg,balanced",
                **MNIST_CONFIG_OVERRIDES,
                **DECAVG_CONFIG_OVERRIDES
            }),
            experiments_directory_root=str(simulation_args.output_directory),
            seed=simulation_args.seed,
        ),
        load_experiment_from_yamls(
            files=[*DECENTRALIZED_CONFIGS, *MNIST_BA_CONFIGS],
            overrides=convert_to_nested_dict({
                "name": "mnist,BA,DecDiff,balanced",
                **MNIST_CONFIG_OVERRIDES,
                **DECDIFF_CONFIG_OVERRIDES,
            }),
            experiments_directory_root=str(simulation_args.output_directory),
            seed=simulation_args.seed,
        ),
        load_experiment_from_yamls(
            files=[*DECENTRALIZED_CONFIGS, *MNIST_ER_CONFIGS],
            overrides=convert_to_nested_dict({
                "name": "mnist,ER,DecAvg,balanced",
                **MNIST_CONFIG_OVERRIDES,
                **DECAVG_CONFIG_OVERRIDES
            }),
            experiments_directory_root=str(simulation_args.output_directory),
            seed=simulation_args.seed,
        ),
        load_experiment_from_yamls(
            files=[*DECENTRALIZED_CONFIGS, *MNIST_ER_CONFIGS],
            overrides=convert_to_nested_dict({
                "name": "mnist,ER,DecDiff,balanced",
                **MNIST_CONFIG_OVERRIDES,
                **DECDIFF_CONFIG_OVERRIDES,
            }),
            experiments_directory_root=str(simulation_args.output_directory),
            seed=simulation_args.seed,
        ),
        load_experiment_from_yamls(
            files=[*FEDERATED_CONFIGS, *MNIST_BA_CONFIGS],
            overrides=convert_to_nested_dict({
                "name": "mnist,FedAvg,balanced",
                **MNIST_CONFIG_OVERRIDES,
                **FEDERATED_CONFIG_OVERRIDES,
            }),
            experiments_directory_root=str(simulation_args.output_directory),
            seed=simulation_args.seed,
        ),
        load_experiment_from_yamls(
            files=[*SCAFFOLD_CONFIGS, *MNIST_BA_CONFIGS],
            overrides=convert_to_nested_dict({
                "name": "mnist,Scaffold,balanced",
                **MNIST_CONFIG_OVERRIDES,
                **SCAFFOLD_CONFIGS_OVERRIDES
            }),
            experiments_directory_root=str(simulation_args.output_directory),
            seed=simulation_args.seed,
        ),
    ]


# ..... ..... ..... ..... ..... ..... ..... ..... #
# MAIN
# ..... ..... ..... ..... ..... ..... ..... ..... #
def _worker_init():
    """Ignore SIGINT in workers — let the parent process handle shutdown."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _run_config(
    args_tuple: Tuple[SimulationArguments, ExperimentConfiguration]
):
    simulation_args, cfg = args_tuple
    try:
        loaded_engine = load_engine(simulation_args, cfg, verbose=False)
        loaded_engine.run()
    except Exception as e:
        raise RuntimeError(f"Config {cfg} failed: {e}") from e


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--concurrent-runs",
        type=int,
        default=1,
        help="Number of configurations to run in parallel (default: 1)."
    )
    parser.add_argument(
        "--experiment_name_filter",
        type=str,
        default=None,
        help="Filter experiments names with a regex."
    )
    simulation_args, _ = cli_argument_parser(parser)
    extra_arguments = parser.parse_args()

    try:
        configurations = [
            *load_mnist_configurations(simulation_args),
        ]
    except ValidationError as e:
        print("\n❌ Failed to parse the configuration due to the following validation errors: ")
        for err in e.errors():
            loc = ".".join(str(part) for part in err['loc'])
            print(f" - {loc}: {err['msg']} (type: {err.get('type', 'unknown')})")
        return

    if extra_arguments.experiment_name_filter:
        regex_pattern = re.compile(extra_arguments.experiment_name_filter)
        configurations = [cfg for cfg in configurations if regex_pattern.search(cfg.name)]
        print("Found the following configs:")
        print("- " + "\n- ".join([cfg.name for cfg in configurations]))
        try:
            input("Type [ENTER] to continue...")
        except KeyboardInterrupt:
            return

    if simulation_args.sanity_check:
        json_outfname = "experiments_configs.json"
        print(f'🟢 Configurations are fine. Dumped all config to {json_outfname}. Ready to run...')
        all_configs = [cfg.model_dump() for cfg in configurations]
        with open(json_outfname, "w") as f:
            json.dump(all_configs, f, indent=2)
        return

    pool_work_items = [(simulation_args, cfg) for cfg in configurations]
    try:
        with mp.Pool(processes=extra_arguments.concurrent_runs, initializer=_worker_init) as pool:
            with tqdm(total=len(configurations), desc="Running configs", unit="cfg") as pbar:
                for _ in pool.imap_unordered(_run_config, pool_work_items):
                    pbar.update()
    except KeyboardInterrupt:
        print("\nInterrupted — terminating workers.")
        pool.terminate()
        pool.join()


if __name__ == "__main__":
    main()
