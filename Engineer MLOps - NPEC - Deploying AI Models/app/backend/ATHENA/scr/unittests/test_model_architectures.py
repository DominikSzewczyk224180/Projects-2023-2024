import os
import sys
import unittest
import tensorflow as tf
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_architectures import build_unet_model

class Test_model_architecture(unittest.TestCase):

    def test_model_builder(self):
        # Dummy parameters for the model
        patch_size_x = 128
        patch_size_y = 128
        n_channels = 3

        model = build_unet_model(patch_size_x, patch_size_y, n_channels)
        self.assertIsInstance(model, tf.keras.Model)

if __name__ == '__main__':
    unittest.main()
