import unittest

import torch

from dengine.models.utils import (
    model_wise_weighted_average,
    unsafe_element_wise_weighted_average,
    unsafe_layer_wise_weighted_average,
)


class AggregationTests(unittest.TestCase):
    def test_model_weights_are_not_normalized(self):
        models = [{'w': torch.tensor([2, 4])}, {'w': torch.tensor([6, 8])}]
        result = model_wise_weighted_average(models, [1, 3])
        torch.testing.assert_close(result['w'], torch.tensor([20., 28.]))
        self.assertEqual(result['w'].device.type, 'cpu')
        result['w'].zero_()
        self.assertEqual(models[0]['w'].tolist(), [2, 4])

    def test_model_missing_layer_raises(self):
        with self.assertRaises(KeyError):
            model_wise_weighted_average([{'w': torch.tensor([1.])}, {}], [.5, .5])

    def test_layer_weights_renormalize_over_present_models(self):
        reference = {
            'w': torch.tensor([[99., 99.]]),
            'w2': torch.tensor([[99., 99.]]),
            'keep': torch.tensor([7.]),
        }
        models = [
            {'w': torch.tensor([[2., 4.]]), 'w2': torch.tensor([[2., 4.]])},
            {'w2': torch.tensor([[4., 8.]])},
            {'w': torch.tensor([[6., 8.]]), 'w2': torch.tensor([[6., 12.]])},
        ]
        # Layer w uses only the first and third models with weights 1/4 and 3/4.
        # Layer w2 uses all three models with weights 1/8, 4/8, and 3/8.
        # Without explicit weights, each available model contributes equally.
        cases = [
            ([1, 4, 3], [[5., 7.]], [[4.5, 9.]]),
            (None, [[4., 6.]], [[4., 8.]]),
        ]
        for weights, expected_w, expected_w2 in cases:
            with self.subTest(weights=weights):
                result = unsafe_layer_wise_weighted_average(models, reference, weights)
                torch.testing.assert_close(result['w'], torch.tensor(expected_w))
                torch.testing.assert_close(result['w2'], torch.tensor(expected_w2))
                torch.testing.assert_close(result['keep'], reference['keep'])
                result['keep'].zero_()
                self.assertEqual(reference['keep'].item(), 7.)

    def test_element_nan_exclusion_and_reference_fallback(self):
        nan = float('nan')
        reference = {'w': torch.tensor([[90., 91., 92., 93.]]), 'keep': torch.tensor([7.])}
        models = [
            {'w': torch.tensor([[2., nan, 4., nan]])},
            {},
            {'w': torch.tensor([[6., 8., nan, nan]])},
        ]
        result = unsafe_element_wise_weighted_average(models, reference, [1, 100, 3])
        torch.testing.assert_close(result['w'], torch.tensor([[5., 8., 4., 93.]]))
        torch.testing.assert_close(result['keep'], reference['keep'])
        self.assertTrue(torch.isnan(models[0]['w'][0, 1]))
        result['keep'].zero_()
        self.assertEqual(reference['keep'].item(), 7.)
        self.assertEqual(reference['w'].tolist(), [[90., 91., 92., 93.]])

    def test_element_requires_explicit_weights(self):
        with self.assertRaises(NotImplementedError):
            unsafe_element_wise_weighted_average([{'w': torch.tensor([1.])}], {'w': torch.tensor([2.])})

    def test_empty_partial_models_preserve_reference(self):
        reference = {'w': torch.tensor([7.])}
        for aggregate in (unsafe_layer_wise_weighted_average, unsafe_element_wise_weighted_average):
            with self.subTest(aggregate=aggregate.__name__):
                result = aggregate([], reference, [])
                torch.testing.assert_close(result['w'], reference['w'])
                result['w'].zero_()
                self.assertEqual(reference['w'].item(), 7.)


if __name__ == '__main__':
    unittest.main()
