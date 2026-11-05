# Event API Examples

Examples built on dEngine's event-driven simulation API. For the API itself see [`event_api`](../../dengine/scenarios/event_api/).

| File | What it shows |
| --- | --- |
| [`main.py`](main.py) | Standalone simulation with manual synchronization and periodic testing |
| [`main_parallel.py`](main_parallel.py) | Standalone simulation where clients train and synchronize in parallel via `ParallelEvent` |
| [`runtime.py`](runtime.py) | Custom engine factory for the HTTP service |
| [`dynamic_config.yml`](dynamic_config.yml) | Experiment config shared by the standalone scripts (MNIST, 10 `DecAvgClient`s on a complete graph) |
| [`config.json`](config.json) | Client configuration sent to the HTTP service |

---

## Standalone Simulation Scripts

If you do not need HTTP orchestration or REST APIs, run experiments directly from the command line with `SyncEngine`. The scripts load dataset profiles and experiment configurations from YAML files without starting an HTTP daemon.

Run them from this directory, since `dynamic_config.yml` is resolved relative to the working directory.

### Common options

Both scripts accept the standard simulator options:

| Option | Default | Description |
| --- | --- | --- |
| `--config FILE [FILE ...]` | — | Extra YAML files merged on top of `dynamic_config.yml`. |
| `--verbosity {silent,debug,info}` | `silent` | Output level (see [Verbosity](#verbosity)). |
| `--output_directory PATH` | `./logs` | Where experiment outputs are written. |
| `--dataset_directory PATH` | `./datasets` | Where datasets are stored. |
| `--gpu ID` | `0` | GPU to use. |
| `--seed N` | `-1` (random) | Random seed. |
| `--sanity_check` | off | Validate the configuration and exit without running. |

Any remaining `key=value` arguments are applied as config overrides, e.g. `graph.arguments.n=20`.

### `main.py`: manual synchronization and periodic testing

Each client trains at random intervals (0–10 virtual seconds) and, right after each training round, sends its model to its neighbours through a custom `Synchronization` handler. All clients are tested together on a fixed schedule through a user-defined `"test"` event.

```bash
python main.py --test_every_n_seconds 40
```

| Option | Default | Description |
| --- | --- | --- |
| `--test_every_n_seconds` | `40` | Virtual seconds between test rounds. |

### `main_parallel.py`: parallel training and synchronization

All clients train in lockstep. Every round, the script groups the `LocalTraining` events of all clients into a single `ParallelEvent`, and does the same for their `Synchronization` events, so the engine runs them concurrently instead of one by one.

```bash
python main_parallel.py --local_update_every 40 --num_parallel_clients 4 --verbosity info
```

| Option | Default | Description |
| --- | --- | --- |
| `--local_update_every` | `40` | Virtual seconds between consecutive training rounds of each client. |
| `--num_parallel_clients` | no limit | Maximum number of clients processed concurrently. Capped at the number of clients. |

Synchronization and testing use the engine defaults (`synchronization_mode="auto"`, `testing_mode="on_local_training_end"`), so each client is tested at the end of its own training step.

#### How it works

The start handler schedules one `ParallelEvent` wrapping a `LocalTraining` event per client:

```python
@engine.start()
def entrypoint(clients: Sequence[DecAvgClient] = Depends(get_all_clients)):
    training_events = [
        LocalTraining(timestamp=engine.timestamp, client=c.UUID) for c in clients
    ]
    return [
        ParallelEvent(
            events=training_events,
            tag="PARALLEL_LOCAL_TRAINING",
            num_workers=min(max_num_workers, len(clients)),
        )
    ]
```

A handler registered with `@engine.parallel_event` on that tag then schedules the next round: a parallel synchronization one microsecond later, and a parallel training `local_update_every` seconds later.

```python
@engine.parallel_event(stage="start", tag="PARALLEL_LOCAL_TRAINING")
def schedule_synch_and_training(event, clients = Depends(get_all_clients)):
    ...
    return [
        ParallelEvent(timestamp=..., events=training_events, tag="PARALLEL_LOCAL_TRAINING", num_workers=...),
        ParallelEvent(timestamp=..., events=synch_events, tag="PARALLEL_SYNCHRONIZATION", num_workers=...),
    ]
```


### Verbosity
When `SyncEngine` consumes a `ParallelEvent`, it splits the inner events round-robin into `num_workers` shards and runs each shard on a thread pool. On CUDA, each shard runs on its own CUDA stream so kernels from different clients can overlap. With `num_workers=1` the events run serially.

`SyncEngine` now accepts `verbose` and `hide_scenario_progress`, and forwards any other keyword arguments to `ScenarioEventEngine`. `main_parallel.py` maps `--verbosity` onto them:

| `--verbosity` | Behaviour |
| --- | --- |
| `silent` | No output. |
| `info` | A `Virtual time` progress bar showing elapsed virtual seconds (up to `max_communication_rounds`) and the tag of the event being processed. |
| `debug` | Per-event logs with handler names and timings; progress bar hidden. |

---

## HTTP / Multi-Node Distributed Service

Run the simulation as an HTTP daemon capable of binding to network interfaces, accepting events via REST endpoints, and interconnecting multiple engines across nodes.

### Setup

Create the required directories:

```bash
mkdir -p logs datasets
```

Start the backend:

```bash
dengine-serve
```

The server uses `DistributedEngine` by default.
To use a different engine factory, set `DENGINE_ENGINE_FACTORY`:

```bash
DENGINE_ENGINE_FACTORY="runtime:create_engine" dengine-serve
```

The API is available at:

http://localhost:8000

Interactive API documentation is available at:

http://localhost:8000/docs

### Configuration

The server can be configured through environment variables (or via an env file called `.env.local`):

| Environment Variable | Default | Description |
| --- | --- | --- |
| `HOST` | `0.0.0.0` | Host address the server binds to. |
| `PORT` | `8000` | Port the server listens on. |
| `DENGINE_RELOAD_ON_CHANGE` | `false` | Reload the server when source files change. |
| `OUTPUT_DIRECTORY` | `logs/` | Directory where server output and logs are stored. |
| `DATASETS_DIRECTORY` | `datasets/` | Directory used for datasets. |
| `DENGINE_ENGINE_FACTORY` | `DistributedEngine` | Import path of the engine factory to use. |
| `DENGINE_API_BASE_URL` | `http://localhost:8000` | Base URL used to redirect model download presigned URLs. |
| `DENGINE_JWT_SECRET` | `password` | Secret key used to sign presigned JWT download URLs. |

### Adding a Client

To add a client, send its configuration to the `/simulation/load` endpoint:

```bash
curl -X POST 'http://localhost:8000/simulation/load' \
  -H 'Content-Type: application/json' \
  -d @config.json
```

The `config.json` file should contain the configuration required to load the client.

### Running a Connected Engine

You can run another engine that connects to and has access to the first engine.

For example, if the first engine is running on port `8000`, start the connected engine on port `8001` with:

```bash
DENGINE_RELOAD_ON_CHANGE=True \
PEERS_API_BASE_URL=http://127.0.0.1:8000/ \
DENGINE_PORT=8001 \
DENGINE_ENGINE_FACTORY="runtime:create_engine" \
dengine-serve
```

Here:

* `PEERS_API_BASE_URL` specifies the API endpoint of the existing engine.
* `DENGINE_PORT=8001` runs the new engine on port `8001`, avoiding a conflict with the first engine.
* `DENGINE_ENGINE_FACTORY="runtime:create_engine"` uses the custom runtime engine.
* `DENGINE_RELOAD_ON_CHANGE=True` enables reloading when source files change.

The first engine remains available at:

```text
http://localhost:8000
```

while the connected engine is available at:

```text
http://localhost:8001
```

### Sending Events

Make sure the custom runtime is running before sending events.

Send a new event with:

```bash
curl -X POST 'http://localhost:8000/simulation/event/new' \
  -H 'Content-Type: application/json' \
  -d '[{"tag": "say_hello"}]'
```

A successful request returns `200 OK` and logs the received event:

```text
INFO:     127.0.0.1:40888 - "POST /simulation/event/new HTTP/1.1" 200 OK
Event e9951059-e317-42a8-8063-773ee799a7ef received
```

The event ID is generated by the server and will be different for each request.
