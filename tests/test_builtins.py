"""Configuration-only regression checks for issue #42; never load an engine."""
from pathlib import Path
import unittest

from dengine import load_experiment_from_yamls
from dengine.callbacks.local import (
    LossDumpCallback, ConfusionMatrixDumpOnCommRoundEndCallback,
    ConfusionMatrixDumpOnAggregationEnd,
)
from dengine.config.builtins import BUILTINS
from dengine.config.utils import convert_to_nested_dict
from dengine.models.classifier import CNNMnist
from dengine.scenarios.decentralized import DecAvgClient


def flatten_builtin_paths() -> list[Path]:
    def leaves(namespace):
        for name, value in vars(namespace).items():
            if name.startswith('_'):
                continue
            if isinstance(value, type):
                yield from leaves(value)
            else:
                yield value

    return list(leaves(BUILTINS))


class BuiltinsTests(unittest.TestCase):
    EXPERIMENT_ROOT = Path('logs')
    OVERRIDES = convert_to_nested_dict({
        'name': 'mnist,ER,DecAvg',
        'graph.arguments.seed': 123,
        'client.local_model.target': CNNMnist,
        'client.target': DecAvgClient,
        'scenario.arguments.common_init': True,
        'client.arguments.use_weighted_avg': True,
        'scenario.arguments.max_communication_rounds': 2,
        'callbacks': [
            {'target': LossDumpCallback},
            {'target': ConfusionMatrixDumpOnCommRoundEndCallback},
            {'target': ConfusionMatrixDumpOnAggregationEnd},
        ],
    })

    def test_all_generated_paths_exist(self):
        paths = flatten_builtin_paths()
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path):
                self.assertIsInstance(path, Path)
                self.assertTrue(path.is_file())

    def test_builtin_config_stems_are_unique(self):
        stems = [path.stem for path in flatten_builtin_paths()]
        self.assertEqual(len(stems), len(set(stems)))

    def load_experiment_from_builtins(self, files):
        return load_experiment_from_yamls(
            files=files,
            overrides=self.OVERRIDES,
            experiments_directory_root=str(self.EXPERIMENT_ROOT),
            seed=123,
        )

    def test_issue_composition(self):
        config = self.load_experiment_from_builtins([
            BUILTINS.CORE.SCENARIOS.DECENTRALIZED_HOMOGENOUS,
            BUILTINS.CORE.GRAPH.ER_SMALL,
            BUILTINS.CORE.DATASETS.MNIST,
            BUILTINS.CORE.PARTITIONING.IID,
        ])
        expected = {
            'EXPERIMENT_VERSION': '1', 'target': None,
            'experiments_directory_root': str(self.EXPERIMENT_ROOT),
            'name': 'mnist,ER,DecAvg', 'seed': 123,
            'dataset': {
                'train': {'target': 'mnist',
                          'arguments': {'train': True}},
                'test': {'target': 'mnist',
                         'arguments': {'train': False}},
                'postprocessing': None,
            },
            'partitioning': {
                'target': 'random_iid_balanced',
                'arguments': {'validation_percentage': 0.2}},
            'graph': {
                'target': 'NXGraph',
                'arguments': {'nx_class': 'erdos_renyi_graph',
                              'n': 10, 'p': 0.7, 'seed': 123}},
            'scenario': {
                'target': 'VanillaDecentralizedSynch',
                'arguments': {'common_init': True,
                              'max_communication_rounds': 2}},
            'client': {
                'target': 'DecAvgClient',
                'arguments': {'use_weighted_avg': True},
                'training_engine': {
                    'target': 'TrainingEngine',
                    'arguments': {
                        'optimizer': 'SGD', 'epochs': 5, 'lr': 0.001,
                        'sgd_momentum': 0.9, 'training_batch_size': 32,
                        'validation_batch_size': 128}},
                'local_model': {'target': 'CNNMnist', 'arguments': None}},
            'callbacks': [
                {'target': 'LossDumpCallback', 'arguments': None},
                {'target': 'ConfusionMatrixDumpOnCommRoundEndCallback',
                 'arguments': None},
                {'target': 'ConfusionMatrixDumpOnAggregationEnd',
                 'arguments': None}],
        }
        self.assertEqual(config.model_dump(), expected)


if __name__ == '__main__':
    unittest.main()
