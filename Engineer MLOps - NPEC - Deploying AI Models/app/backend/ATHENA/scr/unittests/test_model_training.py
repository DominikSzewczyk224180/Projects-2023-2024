import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import itertools
from tensorflow.keras.models import Model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_training import train

class TestModelSelection(unittest.TestCase):
    """Unit tests for verifying model selection and configuration for ResNet-based models.
    Author: Benjamin Graziadei
    """

    def setUp(self) -> None:
        """Prepare data for each test; this method sets up the test environment."""
        self.patch_size = 256
        self.batch_size = 8
        self.input_shape = (self.batch_size, self.patch_size, self.patch_size, 1)  # Assume 1 channel (e.g., grayscale)
        self.output_shape_binary = (self.batch_size, self.patch_size, self.patch_size, 1)  # Binary segmentation
        self.output_shape_multiclass = (self.batch_size, self.patch_size, self.patch_size, 4)  # Multiclass segmentation

        # Mock image data and mask
        self.image_data = np.random.random(self.input_shape).astype(np.float32)
        self.mask_binary = np.random.randint(0, 2, self.output_shape_binary).astype(np.float32)
        self.mask_multiclass = np.random.randint(0, 4, self.output_shape_multiclass).astype(np.float32)

    def model_test(self, depth_sel: int, mask: np.ndarray, expected_output_shape: int, expected_activation: str, expected_loss: str) -> None:
        """
        Generic test function to avoid redundancy and test model setup based on the given parameters.

        Args:
            depth_sel (int): The depth selection for the model (0 for ResNet50, 1 for ResNet101).
            mask (np.ndarray): The mask data used in training (binary or multiclass).
            expected_output_shape (int): The expected last dimension of the model's output shape.
            expected_activation (str): The expected activation function's name in the model's final layer.
            expected_loss (str): The expected loss function used in the model.
        """
        with patch('model_training.train_generator') as mock_train_gen, \
             patch('model_training.val_generator') as mock_val_gen, \
             patch('tensorflow.keras.models.Model.fit') as mock_fit:

            mock_train_gen.return_value = itertools.cycle([(self.image_data, mask)])
            mock_val_gen.return_value = itertools.cycle([(self.image_data, mask)])
            mock_fit.return_value = None  # Avoid actual training

            # Invoke training function
            model, history = train(depth_sel=depth_sel, train_path='some/train/path')

            # Verify that fit was called
            mock_fit.assert_called()

            # Check model properties
            self.assertIsInstance(model, Model)
            self.assertEqual(model.output_shape, (None, self.patch_size, self.patch_size, expected_output_shape))
            self.assertEqual(model.layers[-1].activation.__name__, expected_activation)
            self.assertEqual(model.loss, expected_loss)

    def test_resnet50_binary(self):
        """Test ResNet50 configuration and setup for binary classification."""
        self.model_test(0, self.mask_binary, 1, 'sigmoid', 'binary_crossentropy')

    def test_resnet101_binary(self):
        """Test ResNet101 configuration and setup for binary classification."""
        self.model_test(1, self.mask_binary, 1, 'sigmoid', 'binary_crossentropy')

    def test_resnet50_multiclass(self):
        """Test ResNet50 configuration and setup for multiclass classification."""
        self.model_test(0, self.mask_multiclass, 4, 'softmax', 'categorical_crossentropy')

    def test_resnet101_multiclass(self):
        """Test ResNet101 configuration and setup for multiclass classification."""
        self.model_test(1, self.mask_multiclass, 4, 'softmax', 'categorical_crossentropy')

if __name__ == '__main__':
    unittest.main()
