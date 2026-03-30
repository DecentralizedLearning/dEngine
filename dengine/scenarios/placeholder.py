from dengine.config.configuration import ClientModuleConfig
from dengine.dataset.dataset import SupervisedDataset
from dengine.graph.graph import Graph
from dengine.partitioning import TYPE_DATASET_PARTITIONING

from .decorators import register_scenario
from .scenario import AbstractScenarioEngine


@register_scenario()
class ScenarioPlaceholder(AbstractScenarioEngine):
    def __init__(
        self,
        *args,
        graph: Graph,
        training_data: SupervisedDataset,
        data_partitions: TYPE_DATASET_PARTITIONING,
        test_data: SupervisedDataset,
        client_configuration: ClientModuleConfig,
        common_init: bool = False,
        **kwargs
    ):
        self._graph = graph
        self.training_data = training_data
        self.test_data = test_data
        self.partitioning = data_partitions
        self._clients = self.init_clients(
            client_configuration,
            training_data,
            data_partitions,
            common_init,
        )

    def run(self):
        pass
