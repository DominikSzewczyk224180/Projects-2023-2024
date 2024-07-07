import unittest
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from typing import Generator, Tuple

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_evaluation import convert_masks, calculate_metrics, evaluate_model, main


class TestFunctions(unittest.TestCase):

    def setUp(self):
        # Create a simple model for testing
        self.model = Sequential([
            Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
            Flatten(),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        self.model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=[BinaryAccuracy()])

        # Dummy data for testing
        self.dummy_images = np.random.rand(8, 256, 256, 3)
        self.dummy_masks = np.random.randint(0, 2, size=(8, 256, 256, 1))

    def test_convert_masks_binary(self):
        predictions = np.random.rand(8, 256, 256, 1)
        binary_predictions, masks = convert_masks(predictions, self.dummy_masks, binary=True)
        self.assertEqual(binary_predictions.shape, predictions.shape)
        self.assertEqual(masks.shape, self.dummy_masks.shape)

    # def test_convert_masks_multiclass(self):
    #     predictions = np.random.rand(8, 256, 256, 3)
    #     multiclass_predictions, masks = convert_masks(predictions, to_categorical(self.dummy_masks, 3), binary=False)
    #     self.assertEqual(multiclass_predictions.shape[:-1], predictions.shape[:-1])
    #     self.assertEqual(masks.shape[:-1], self.dummy_masks.shape[:-1])

    def test_calculate_metrics_binary(self):
        y_true = self.dummy_masks
        y_pred = np.random.rand(8, 256, 256, 1)
        metrics = calculate_metrics(y_true, y_pred, binary=True)
        self.assertTrue('f1_score' in metrics)
        self.assertTrue('iou_score' in metrics)

    # def test_calculate_metrics_multiclass(self):
    #     y_true = to_categorical(self.dummy_masks, 3)
    #     y_pred = np.random.rand(8, 256, 256, 3)
    #     metrics = calculate_metrics(y_true, y_pred, binary=False)
    #     self.assertTrue('f1_score' in metrics)
    #     self.assertTrue('iou_score' in metrics)

    # def test_evaluate_model(self):
    #     generator = ((self.dummy_images, "dummy_name", self.dummy_masks),)
    #     metrics = evaluate_model(self.model, generator)
    #     self.assertTrue(metrics)

    # def test_main(self):
    #     test_path = "dummy_test_path"
    #     metrics = main(self.model, test_path)
    #     self.assertTrue(metrics)


if __name__ == '__main__':
    unittest.main()
