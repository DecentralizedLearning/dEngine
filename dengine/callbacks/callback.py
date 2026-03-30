from typing import List, Sequence, Optional

from dengine.config import instantiate_configuration_module, DynamicModuleConfigBase
from dengine.interfaces import ClientCallbackInterface, ClientInterface, ScenarioEngineInterface, GenericMessage
from dengine.dataset import SupervisedDataset
from dengine.config import ExperimentConfiguration

from .decorators import BUILTIN_CALLBACKS


class PeriodicCallback(ClientCallbackInterface[GenericMessage]):
    def __init__(
        self,
        client: ClientInterface,
        scenario: ScenarioEngineInterface,
        test_data: SupervisedDataset,
        experiment_configuration: ExperimentConfiguration,
        # Additional args
        every_n_local_trainings: Optional[int] = None,
        every_n_tests: Optional[int] = None,
        every_n_local_epochs: Optional[int] = None,
        every_n_batch_steps: Optional[int] = None,
    ):
        self.client = client
        self.scenario = scenario
        self.test_data = test_data
        self.experiment_configuration = experiment_configuration

        self._every_n_local_trainings = every_n_local_trainings
        self._every_n_tests = every_n_tests
        self._every_n_local_epochs = every_n_local_epochs
        self._every_n_batch_steps = every_n_batch_steps

    def _skip_local_training(self, value: float) -> bool:
        if not self._every_n_local_trainings:
            return False
        return (value > 0) and (value % self._every_n_local_trainings != 0)

    def _skip_test(self, value: float) -> bool:
        if not self._every_n_tests:
            return False
        return (value > 0) and (value % self._every_n_tests != 0)

    def _skip_epoch(self, value: float) -> bool:
        if not self._every_n_local_epochs:
            return False
        return (value > 0) and (value % self._every_n_local_epochs != 0)

    def _skip_batch_step(self, value: int) -> bool:
        if not self._every_n_batch_steps:
            return False
        return (value > 0) and (value % self._every_n_batch_steps != 0)

    def on_synchronization_start(self, current_time: float):
        if not self._skip_local_training(current_time):
            self.synchronization_start(current_time)

    def on_synchronization_end(self, current_time: float, messages: Sequence[GenericMessage]):
        if not self._skip_local_training(current_time):
            self.synchronization_end(current_time, messages)

    def on_aggregation_start(self, current_time: float):
        if not self._skip_local_training(current_time):
            self.aggregation_start(current_time)

    def on_aggregation_end(self, current_time: float):
        if not self._skip_local_training(current_time):
            self.aggregation_end(current_time)

    def on_training_batch_start(self, step: int, *args, **kwargs):
        if not self._skip_batch_step(step):
            self.training_batch_start(step, *args, **kwargs)

    def on_training_batch_end(self, step: int, *args, **kwargs):
        if not self._skip_batch_step(step):
            self.training_batch_end(step, *args, **kwargs)

    def on_training_epoch_start(self, epoch: int):
        if not self._skip_epoch(epoch):
            self.training_epoch_start(epoch)

    def on_training_epoch_end(self, epoch: int, **kwargs):
        if not self._skip_epoch(epoch):
            self.training_epoch_end(epoch, **kwargs)

    def on_test_inference_start(self, current_time: float):
        if not self._skip_test(current_time):
            self.test_inference_start(current_time)

    def on_test_inference_end(self, current_time: float, **kwargs):
        if not self._skip_test(current_time):
            self.test_inference_end(current_time, **kwargs)

    def on_local_training_start(self, current_time: float):
        if not self._skip_local_training(current_time):
            self.local_training_start(current_time)

    def on_local_training_end(self, current_time: float, **kwargs):
        if not self._skip_local_training(current_time):
            self.local_training_end(current_time, **kwargs)

    # ..... #
    # Redefinitions
    # ..... #
    def synchronization_start(self, current_time: float):
        ...

    def synchronization_end(self, current_time: float, messages: Sequence[GenericMessage]):
        ...

    def aggregation_start(self, current_time: float):
        ...

    def aggregation_end(self, current_time: float):
        ...

    def local_training_start(self, current_time: float):
        ...

    def local_training_end(self, current_time: float, **kwargs):
        ...

    def training_batch_start(self, step: int, *args):
        ...

    def training_batch_end(self, step: int, *args, **kwargs):
        ...

    def training_epoch_start(self, epoch: int):
        ...

    def training_epoch_end(self, epoch: int, **kwargs):
        ...

    def test_inference_start(self, current_time: float):
        ...

    def test_inference_end(self, current_time: float, **kwargs):
        ...


class CallbackList(PeriodicCallback):
    def __init__(self, callback: List[PeriodicCallback]):
        self._callbacks = callback

    def on_synchronization_start(self, *args, **kwargs):
        for cbk in self._callbacks:
            cbk.on_synchronization_start(*args, **kwargs)

    def on_synchronization_end(self, *args, **kwargs):
        for cbk in self._callbacks:
            cbk.on_synchronization_end(*args, **kwargs)

    def on_aggregation_start(self, *args, **kwargs):
        for cbk in self._callbacks:
            cbk.on_aggregation_start(*args, **kwargs)

    def on_aggregation_end(self, *args, **kwargs):
        for cbk in self._callbacks:
            cbk.on_aggregation_end(*args, **kwargs)

    def on_local_training_start(self, *args, **kwargs):
        for cbk in self._callbacks:
            cbk.on_local_training_start(*args, **kwargs)

    def on_local_training_end(self, *args, **kwargs):
        for cbk in self._callbacks:
            cbk.on_local_training_end(*args, **kwargs)

    def on_training_batch_start(self, *args, **kwargs):
        for cbk in self._callbacks:
            cbk.on_training_batch_start(*args, **kwargs)

    def on_training_batch_end(self, *args, **kwargs):
        for cbk in self._callbacks:
            cbk.on_training_batch_end(*args, **kwargs)

    def on_training_epoch_start(self, *args, **kwargs):
        for cbk in self._callbacks:
            cbk.on_training_epoch_start(*args, **kwargs)

    def on_training_epoch_end(self, *args, **kwargs):
        for cbk in self._callbacks:
            cbk.on_training_epoch_end(*args, **kwargs)

    def on_test_inference_start(self, *args, **kwargs):
        for cbk in self._callbacks:
            cbk.on_test_inference_start(*args, **kwargs)

    def on_test_inference_end(self, *args, **kwargs):
        for cbk in self._callbacks:
            cbk.on_test_inference_end(*args, **kwargs)


class DummyCallback(PeriodicCallback):
    def __init__(self, *args, **kwargs):
        self._every_n_local_trainings = None
        self._every_n_tests = None
        self._every_n_local_epochs = None
        self._every_n_batch_steps = None


def callback_factory(
    client: ClientInterface,
    scenario: ScenarioEngineInterface,
    configuration: List[DynamicModuleConfigBase],
    **kwargs
) -> CallbackList:
    res = []
    for cfg in configuration:
        cb = instantiate_configuration_module(
            cfg,
            superclass=PeriodicCallback,
            allowed_cls=BUILTIN_CALLBACKS,
            client=client,
            scenario=scenario,
            **kwargs
        )
        res.append(cb)
    return CallbackList(res)
