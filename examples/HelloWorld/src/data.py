from dengine.partitioning import TYPE_DATASET_PARTITIONING
from dengine.partitioning.iid import random_iid
from dengine.partitioning.decorators import register_partitioning
from dengine.graph import Graph

from dengine.dataset import SupervisedDataset
from dengine.dataset.mnist import load_mnist
from dengine.dataset.decorators import register_dataset


@register_dataset()
def custom_mnist_test(
    *args, **kwargs
) -> SupervisedDataset:
    return load_mnist(*args, train=False, **kwargs)


@register_dataset()
def custom_mnist_train(
    *args, **kwargs
) -> SupervisedDataset:
    return load_mnist(*args, train=True, **kwargs)


@register_partitioning()
def custom_partitioning(
    dataset: SupervisedDataset,
    test: SupervisedDataset,
    partitions: TYPE_DATASET_PARTITIONING,
    graph: Graph,
    validation_percentage: float
) -> TYPE_DATASET_PARTITIONING:
    return random_iid(
        dataset=dataset,
        graph=graph,
        partitions=partitions,
        test=test,
        minimum_training_samples_per_class=10,
        validation_percentage=validation_percentage
    )
