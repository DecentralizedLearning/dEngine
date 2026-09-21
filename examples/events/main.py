from dataclasses import dataclass
import gc
from tqdm import tqdm
from typing import Sequence
from random import randint
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from argparse import ArgumentParser

from pydantic import ValidationError

from dengine import (
    Event,
    LocalTraining,
    Synchronization,
    DynamicGraph,
    Depends,
)
from dengine.scenarios.decentralized import DecAvgClient, VanillaDecentralizedMessage
from dengine.utils.utils import model_on_device_context
from dengine.scenarios.event_api.sync_engine import (
    SyncEngine,
    get_graph,
    get_all_clients,
    get_training_client,
    get_src,
    get_destinations,
)
from dengine import BUILTINS, load_experiment_from_yamls
from dengine.bin.args_parser import cli_argument_parser
from dengine.bin.simulation import load_engine


@dataclass
class ClientConfig:
    displacement_seconds: float
    training_time_seconds: float


def create_engine(
    test_timedelta: timedelta,
):
    engine = SyncEngine[DecAvgClient](
        synchronization_mode="manual",
        testing_mode="manual",
        raise_for_unknown_event=False
    )

    @engine.start()
    def entrypoint(
        event: Event,
        graph: DynamicGraph = Depends(get_graph),
        clients: Sequence[DecAvgClient] = Depends(get_all_clients)
    ):
        new_events = [
            Event(timestamp=engine.timestamp + test_timedelta, tag="test")
        ]
        for ith_paiv in clients:
            delta = randint(0, 10)
            e = LocalTraining(
                timestamp=(engine.timestamp + timedelta(seconds=delta)),
                client=ith_paiv.UUID
            )
            new_events.append(e)
        return new_events

    @engine.local_training("start")
    def schedule_synch_and_training(
        event: LocalTraining,
        client: DecAvgClient = Depends(get_training_client)
    ):
        delta = randint(0, 10)
        return [
            Synchronization(timestamp=event.timestamp + timedelta(microseconds=1), src=client.UUID, destinations=None),
            LocalTraining(timestamp=event.timestamp + timedelta(seconds=delta), client=client.UUID)
        ]

    @engine.synchronization("start")
    def update_buffers(
        event: Synchronization,
        trained_paiv: DecAvgClient = Depends(get_src),
        destinations: Sequence[DecAvgClient] = Depends(get_destinations),
    ):
        if len(destinations) == 0:
            return
        paiv_checkpoint = deepcopy(trained_paiv)
        for dst in destinations:
            msg = VanillaDecentralizedMessage(
                time=event.timestamp.timestamp(),
                source_client=paiv_checkpoint,
            )
            dst.message_buffer.put(msg)

    @engine.user_event(tag="test")
    def test_clients(
        clients: Sequence[DecAvgClient] = Depends(get_all_clients),
    ):
        for p in tqdm(clients, desc="Running tests", leave=False):
            with model_on_device_context(p.model, engine._device):
                p.test(engine.timestamp.timestamp(), engine.test_data)
        gc.collect()
        return [
            Event(timestamp=engine.timestamp + test_timedelta, tag="test")
        ]

    return engine


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--test_every_n_seconds',
        type=float,
        default=40
    )
    simulation_args, overrides = cli_argument_parser(parser)

    args, _ = parser.parse_known_args()

    engine = create_engine(
        timedelta(seconds=args.test_every_n_seconds),
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
