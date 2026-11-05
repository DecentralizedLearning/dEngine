from dataclasses import dataclass
import sys
from typing import Sequence, List
from datetime import timedelta
from pathlib import Path
from argparse import ArgumentParser

from pydantic import ValidationError

from dengine import (
    Event,
    LocalTraining,
    Synchronization,
    ParallelEvent,
    Depends,
)
from dengine.scenarios.decentralized import DecAvgClient
from dengine.scenarios.event_api.sync_engine import SyncEngine, get_all_clients
from dengine import BUILTINS, load_experiment_from_yamls
from dengine.bin.args_parser import cli_argument_parser
from dengine.bin.simulation import load_engine, VerbosityLevel


@dataclass
class ClientConfig:
    displacement_seconds: float
    training_time_seconds: float


def create_engine(
    local_update_every: timedelta,
    max_num_workers: int,
    verbosity: VerbosityLevel,
):
    kwargs = {}
    if verbosity == VerbosityLevel.info:
        kwargs["hide_scenario_progress"] = False
        kwargs["verbose"] = False
    elif verbosity == VerbosityLevel.debug:
        kwargs["hide_scenario_progress"] = True
        kwargs["verbose"] = True
    else:
        kwargs["hide_scenario_progress"] = True
        kwargs["verbose"] = False
    engine = SyncEngine[DecAvgClient](raise_for_unknown_event=False, **kwargs)

    @engine.start()
    def entrypoint(
        clients: Sequence[DecAvgClient] = Depends(get_all_clients)
    ):
        training_events = []
        for _client in clients:
            e = LocalTraining(timestamp=(engine.timestamp), client=_client.UUID)
            training_events.append(e)
        return [
            ParallelEvent(
                events=training_events,
                tag="PARALLEL_LOCAL_TRAINING",
                num_workers=min(max_num_workers, len(clients))
            )
        ]

    @engine.parallel_event(stage="start", tag="PARALLEL_LOCAL_TRAINING")
    def schedule_synch_and_training(
        event: LocalTraining,
        clients: Sequence[DecAvgClient] = Depends(get_all_clients)
    ):
        synch_events: List[Event] = []
        training_events: List[Event] = []
        for _client in clients:
            training_events.append(
                LocalTraining(timestamp=(engine.timestamp) + local_update_every, client=_client.UUID)
            )
            synch_events.append(
                Synchronization(timestamp=event.timestamp + timedelta(microseconds=1), src=_client.UUID, destinations=None),
            )
        return [
            ParallelEvent(
                timestamp=training_events[0].timestamp,
                events=training_events,
                tag="PARALLEL_LOCAL_TRAINING",
                num_workers=min(max_num_workers, len(clients))
            ),
            ParallelEvent(
                timestamp=synch_events[0].timestamp,
                events=synch_events,
                tag="PARALLEL_SYNCHRONIZATION",
                num_workers=min(max_num_workers, len(clients))
            )
        ]

    return engine


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--local_update_every",
        type=float,
        default=40,
        help="Simulated time, in seconds, between consecutive local training rounds "
             "of each client (default: %(default)s).",
    )
    parser.add_argument(
        "--num_parallel_clients",
        type=int,
        default=sys.maxsize,
        help="Maximum number of clients trained in parallel (worker processes). "
             "Default: no limit, all clients train at once.",
    )
    simulation_args, overrides = cli_argument_parser(parser)
    args, _ = parser.parse_known_args()

    engine = create_engine(
        local_update_every=timedelta(seconds=args.local_update_every),
        max_num_workers=args.num_parallel_clients,
        verbosity=simulation_args.verbosity
    )
    try:
        cfg = load_experiment_from_yamls(
            files=[
                BUILTINS.CORE.DATASETS.MNIST,
                Path("dynamic_config.yml"),
                *simulation_args.configs,
            ],
            overrides=overrides,
            experiments_directory_root=str(simulation_args.output_directory.absolute()),
            seed=simulation_args.seed,
        )
        loaded_engine = load_engine(simulation_args, cfg, engine, verbose=False)

        if simulation_args.sanity_check:
            print(f'🟢 Configuration {cfg.name} is fine, ready to run...')
            return
    except ValidationError as e:
        print("\n❌ Failed to parse the configuration due to the following validation errors: ")
        for err in e.errors():
            loc = ".".join(str(part) for part in err['loc'])
            print(f" - {loc}: {err['msg']} (type: {err.get('type', 'unknown')})")
        return

    try:
        loaded_engine.run()
    except KeyboardInterrupt:
        print('Exit')


if __name__ == "__main__":
    main()
