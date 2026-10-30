from typing import Tuple
from pathlib import Path
import re
from argparse import ArgumentParser
import signal
import torch.multiprocessing as mp
import json
from typing import List, Dict, Any

from tqdm import tqdm
from pydantic import ValidationError

from dengine.bin.args_parser import cli_argument_parser
from dengine.config.builtins import BUILTINS
from dengine.bin import SimulationArguments
from dengine.config import ExperimentConfiguration
from dengine.config.utils import convert_to_nested_dict
from dengine import load_experiment_from_yamls
from dengine.bin.simulation import load_engine
from dengine.scenarios.decentralized import DecAvgClient


# ..... ..... ..... ..... ..... ..... ..... ..... #
# SCENARIOS
# ..... ..... ..... ..... ..... ..... ..... ..... #
DECENTRALIZED_BA_CONFIGS_FILES = [
    BUILTINS.CORE.SCENARIOS.DECENTRALIZED_HETEROGENOUS,
    BUILTINS.CORE.GRAPH.BA_MEDIUM,
    BUILTINS.CORE.DATASETS.MNIST,
    BUILTINS.CORE.PARTITIONING.IID,
    Path("configs/core.yml")
]

DECENTRALIZED_CONFIG_OVERRIDES = {
    "client.target": DecAvgClient,
    # Aggregation and Scenario ..... ..... #
    "scenario.arguments.common_init": False,
    "client.arguments.include_myself": True,
    "client.arguments.use_weighted_avg": True,
    "client.training_engine.arguments.patience": 0,
    "client.training_engine.arguments.epochs": 8,
    "client.training_engine.arguments.validation_batch_size": 512,
    "client.training_engine.arguments.training_batch_size": 32,
    "partitioning.arguments.validation_percentage": 0.2,
    "scenario.arguments.max_communication_rounds": 600
}


def load_decentralized(
    simulation_args: SimulationArguments,
    config_files: List[Path] = [],
    additional_config_overrides: Dict[str, Any] = {},
    debug: bool = False,
):
    debug_argument = {
        "client.arguments.debug_skip_training": True,
        "client.arguments.debug_skip_test": True
    } if debug else {}

    configs = []

    ba_config = load_experiment_from_yamls(
        files=[*config_files, *DECENTRALIZED_BA_CONFIGS_FILES],
        overrides=convert_to_nested_dict({
            "graph.arguments.seed": simulation_args.seed,
            **debug_argument,
            **DECENTRALIZED_CONFIG_OVERRIDES,
            **additional_config_overrides
        }),
        experiments_directory_root=str(simulation_args.output_directory.absolute()),
        seed=simulation_args.seed,
    )
    available_per_label_samples = ba_config.partitioning.arguments['max_per_label_samples']
    available_per_label_samples = int(available_per_label_samples * .8)
    ba_config.name = (
        f"{ba_config.dataset.train.target},"
        "BA,DecAvg,"
        f"{ba_config.client.local_model.target},"
        f"per_label_samples={available_per_label_samples}"
    )
    configs.append(ba_config)

    return configs


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
        "--concurrent_runs",
        type=int,
        default=1,
        help="Number of configurations to run in parallel (default: 1)."
    )
    parser.add_argument(
        "--num_parallel_clients",
        type=int,
        default=1,
        help="Number of clients to run in parallel in each simulation (default: 1)."
    )
    parser.add_argument(
        "--experiment_name_filter",
        type=str,
        action="append",
        default=None,
        help="Filter experiments names with a regex."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Sets the argument client.arguments.debug_skip_training=true"
    )
    simulation_args, _ = cli_argument_parser(parser)
    extra_arguments = parser.parse_args()

    configurations = []
    try:
        for max_per_label_samples in [80, 60, 40, 30, 20, 4]:
            max_per_label_samples = int(max_per_label_samples / .8)
            configurations += [
                *load_decentralized(
                    simulation_args,
                    additional_config_overrides={
                        "scenario.arguments.num_parallel_clients": extra_arguments.num_parallel_clients,
                        "partitioning.arguments.max_per_label_samples": max_per_label_samples
                    },
                    debug=extra_arguments.debug
                ),
            ]
    except ValidationError as e:
        print("\n❌ Failed to parse the configuration due to the following validation errors: ")
        for err in e.errors():
            loc = ".".join(str(part) for part in err['loc'])
            print(f" - {loc}: {err['msg']} (type: {err.get('type', 'unknown')})")
        return

    if extra_arguments.experiment_name_filter:
        selected_experiments = set()
        for pattern in extra_arguments.experiment_name_filter:
            regex_pattern = re.compile(pattern)
            selected_experiments.update(
                [cfg.name for cfg in configurations if regex_pattern.search(cfg.name)]
            )
        configurations = [cfg for cfg in configurations if cfg.name in selected_experiments]
        print(f"Found {len(configurations)} configs:")
        print("- " + "\n- ".join([cfg.name for cfg in configurations]))
        try:
            input("Type [ENTER] to continue...")
        except KeyboardInterrupt:
            return

    if simulation_args.sanity_check:
        json_outfname = "experiments_configs.json"
        print(f'🟢 Configurations are fine. Dumped the following configs to {json_outfname}.')
        if not extra_arguments.experiment_name_filter:
            print(f"Found {len(configurations)} configs:")
            print("- " + "\n- ".join([cfg.name for cfg in configurations]))
        print("Ready to run...")
        all_configs = [cfg.model_dump() for cfg in configurations]
        with open(json_outfname, "w") as f:
            json.dump(all_configs, f, indent=2)
        return

    if len(configurations) == 1:
        try:
            loaded_engine = load_engine(simulation_args, configurations[0], verbose=False)
            loaded_engine.run()
            return
        except KeyboardInterrupt:
            return
        except Exception as e:
            raise RuntimeError(f"Config {configurations[0]} failed: {e}") from e

    pool_work_items = [(simulation_args, cfg) for cfg in configurations]
    try:
        with mp.Pool(processes=extra_arguments.concurrent_runs, initializer=_worker_init, maxtasksperchild=1) as pool:
            with tqdm(total=len(configurations), desc="Running configs", unit="cfg") as pbar:
                for _ in pool.imap_unordered(_run_config, pool_work_items):
                    pbar.update()
    except KeyboardInterrupt:
        print("\nInterrupted — terminating workers.")
        pool.terminate()
        pool.join()


if __name__ == "__main__":
    main()
