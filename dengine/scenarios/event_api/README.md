# Event Engine

> [!WARNING]
> **This API currently lacks examples**

The built-in `centralized`, `federated`, and `decentralized` scenarios cover the common training loops, but some experiments need custom timing logic — asynchronous clients, network-aware synchronization, custom test schedules, and so on. For these cases dEngine ships an **event-driven scenario API** for building your own scenario as a set of timestamped events and handlers, instead of a fixed loop.

The API is modeled after [FastAPI's dependency-injection system](https://ponderinglion.dev/posts/demystifying-fastapis-dependency-injection/): handlers are plain functions decorated onto an engine, and their arguments are resolved automatically from the current event, the engine instance, or other declared dependencies.

### Core concepts

- **`Event`** (`dengine.Event`) — a Pydantic model with a `timestamp`, a `uuid`, and a `tag`. Events are ordered by `timestamp` and consumed from a priority queue, earliest first.
- **Built-in events**:
  - `StartEvent` — automatically emitted once when `engine.run()` begins.
  - `LocalTraining(client=...)` — a client should perform a local training step.
  - `Synchronization(src=..., destinations=...)` — a client should exchange its model with some destinations (or, if `destinations=None`, its graph neighbors).
  - You can also define fully custom events by subclassing `Event` with your own `tag`, or emit ad-hoc ones with `Event(timestamp=..., tag="my_tag")` and handle them with `@engine.user_event(tag="my_tag")`.
- **Handlers** — functions registered on the engine with a decorator. Whatever a handler returns (a single `Event` or a sequence of them) is automatically scheduled on the event queue.
- **Dependency injection (`Depends`)** — handler and dependency-function parameters are resolved automatically:
  - a parameter annotated as an `Event` subclass receives the event currently being processed;
  - a parameter annotated as the engine's class receives the engine instance;
  - a parameter whose default is `Depends(some_function)` receives the return value of `some_function`, which can itself request the event, the engine, or further `Depends(...)` — dependencies compose recursively.
### The engine lifecycle

`ScenarioEventEngine` (`dengine.scenarios.event_api.scenario.ScenarioEventEngine`) drives the loop in `run()`:

1. A `StartEvent` is added to the queue.
2. The earliest event is popped and `engine.timestamp` is advanced to its timestamp.
3. Every handler registered for that event's tag/stage `"start"` runs, in registration order (handlers registered without a tag run for *every* event).
4. The engine's own `consume_event(event)` runs — this is where a concrete engine (e.g. `SyncEngine`) implements its default behavior, such as actually stepping a client's local training or dispatching a synchronization.
5. Every handler registered for `"end"` runs.
6. The loop repeats until the queue is empty or `max_communication_rounds` is reached.
Handlers are registered with these decorators:

| Decorator | Fires on | Extra filters |
|---|---|---|
| `@engine.start()` | the initial `StartEvent` | — |
| `@engine.local_training(stage, ...)` | `LocalTraining` events | `client`, `uuids`, `timestamp` |
| `@engine.synchronization(stage, ...)` | `Synchronization` events | `src`, `destination`, `uuids`, `timestamp` |
| `@engine.user_event(tag, stage, ...)` | any custom event with a matching `tag` (or every event if `tag=None`) | `client`, `uuids`, `timestamp` |

`stage` is `"start"` or `"end"` (see the lifecycle above). The extra filters let a handler react only to a specific client, a specific `src`/`destination`, a specific timestamp, or a specific event `uuid`; omitting them matches every event with that tag.

### `SyncEngine`

`dengine.SyncEngine` is the ready-to-use, synchronous implementation of `ScenarioEventEngine`. It manages the client dictionary, the priority queue, and the default handling of `LocalTraining`/`Synchronization` events, so you only need to add the handlers that customize *when* those events happen.

```python
SyncEngine[MyClientType](
    synchronization_mode="auto" | "manual",  # "manual" disables the default synchronization behavior
    testing_mode="on_local_training_end" | "manual",  # "manual" disables automatic testing after each local training step
    raise_for_unknown_event=True,  # raise if an event with no matching handler/consume_event branch is popped
)
```

It also exposes ready-made dependencies you can plug into `Depends(...)` (from `dengine.scenarios.event_api.sync_engine`):

| Dependency | Returns |
|---|---|
| `get_graph(engine)` | the scenario's `Graph`/`DynamicGraph` |
| `get_all_clients(engine)` | every client in the scenario |
| `get_training_client(engine, event)` | the client targeted by a `LocalTraining` event |
| `get_src(engine, event)` | the source client of a `Synchronization` event |
| `get_destinations(engine, event, graph=Depends(get_graph))` | the destination clients of a `Synchronization` event (its graph neighbors if `destinations=None`) |
| `get_contact_time(event, graph=Depends(get_graph), destinations=Depends(get_destinations))` | per-destination contact time, for dynamic graphs |

### Writing a custom scenario

A common pattern is a factory function that builds and configures the engine, registering handlers as closures so they can share state (e.g. a physical-layer simulator):

```python
from datetime import timedelta
from typing import Sequence

from dengine import Event, LocalTraining, Synchronization, DynamicGraph, Depends
from dengine.scenarios.event_api.sync_engine import (
    SyncEngine,
    get_all_clients,
    get_training_client,
    get_src,
    get_destinations,
)


def create_engine(test_every: timedelta):
    engine = SyncEngine[MyClient](synchronization_mode="manual")

    @engine.start()
    def entrypoint(clients: Sequence[MyClient] = Depends(get_all_clients)):
        # Kick off local training for every client and schedule the first test
        return [
            Event(timestamp=engine.timestamp + test_every, tag="test"),
            *[
                LocalTraining(timestamp=engine.timestamp + timedelta(seconds=1), client=c.UUID)
                for c in clients
            ],
        ]

    @engine.local_training("start")
    def schedule_next_round(
        event: LocalTraining,
        client: MyClient = Depends(get_training_client),
    ):
        # Trigger a synchronization right after training, then schedule the next round
        return [
            Synchronization(timestamp=event.timestamp, src=client.UUID, destinations=None),
            LocalTraining(timestamp=event.timestamp + timedelta(seconds=10), client=client.UUID),
        ]

    @engine.synchronization("start")
    def broadcast(
        event: Synchronization,
        src: MyClient = Depends(get_src),
        destinations: Sequence[MyClient] = Depends(get_destinations),
    ):
        for dst in destinations:
            ...  # push a message onto dst.message_buffer

    @engine.user_event(tag="test")
    def run_tests(clients: Sequence[MyClient] = Depends(get_all_clients)):
        for c in clients:
            c.test(engine.timestamp.timestamp(), engine.test_data)
        return [Event(timestamp=engine.timestamp + test_every, tag="test")]

    return engine
```

Because `SyncEngine` is a registered scenario, this engine still plugs into the regular loading pipeline: build it, then pass it into `load_engine`/`run_simulation` instead of letting them instantiate a scenario from the config's `scenario.target`:

```python
from dengine import load_experiment_from_yamls
from dengine.bin.simulation import load_engine
from dengine.bin.args_parser import cli_argument_parser

engine = create_engine(test_every=timedelta(seconds=40))

simulation_args, overrides = cli_argument_parser()
cfg = load_experiment_from_yamls(
    files=simulation_args.configs,
    overrides=overrides,
    experiments_directory_root=str(simulation_args.output_directory),
    seed=simulation_args.seed,
)
loaded_engine = load_engine(simulation_args, cfg, engine)
loaded_engine.run()
```

`load_engine` calls `engine.load(...)` with the graph/dataset/partitioning/client config built from the YAML files, and forwards any keys under the config's `scenario.arguments` as extra keyword arguments to `load()` (e.g. `max_communication_rounds`). Everything else about the config format (graph, dataset, partitioning, client, callbacks) is unchanged — only the scenario becomes code instead of a `scenario.target` string.
