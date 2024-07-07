import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import json
import os
import sys
from matplotlib.colors import LinearSegmentedColormap
from tensorflow.keras.models import Model

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root)

from predictions import get_num_classes, custom_argmax, unpatchify, colorize_masks, custom_colormap, create_colored_heatmaps, predict_masks, choose_model, main

class TestModelUtilities(unittest.TestCase):
    """
    Unit tests for the utility functions used in image segmentation model operations.

    Author: Benjamin Graziadei
    """

    def test_get_num_classes(self):
        """
        Test the `get_num_classes` function to ensure it correctly extracts the number of classes from a model's configuration.

        Author: Benjamin Graziadei
        """
        file_path = os.path.join(root, 'unittests', 'model_test.json')
        
        # Load the JSON configuration from the file
        with open(file_path, 'r') as file:
            config_json = file.read()
            config_dict = json.loads(config_json)
        

        # Call the function with the simulated configuration
        num_classes = get_num_classes(config_dict)

        # Assert to check if the correct number of classes is returned
        self.assertEqual(num_classes, 4, "The number of classes extracted should be 10")


    def test_custom_argmax(self):
        """
        Test the `custom_argmax` function to verify it correctly applies a bias and computes the argmax.

        Author: Benjamin Graziadei
        """
        probabilities = np.array([[[0.2, 0.3, 0.5],
                                   [0.6, 0.2, 0.2]],
                                  [[0.1, 0.8, 0.1],
                                   [0.5, 0.2, 0.3]]])
        bias = 0.1
        expected_output = np.array([[2, 0],
                                    [1, 0]])
        output = custom_argmax(probabilities, bias)
        np.testing.assert_array_equal(output, expected_output)

    def test_unpatchify(self):
        """
        Test the `unpatchify` function to ensure it correctly reconstructs an image from patches considering the overlapping.

        Author: Benjamin Graziadei
        """
        # Define the patch size and calculate step size based on overlap
        patch_size = 100
        step_size = int(patch_size/8) *7
        patches_per_dim = 2  # Example for a 2x2 grid of patches

        # Calculate the expected final dimensions
        expected_height = expected_width = (patches_per_dim - 1) * step_size + patch_size

        # Create an array simulating 4 overlapping patches
        patches = np.random.rand(patches_per_dim**2, patch_size, patch_size, 3)  # 4 patches

        reconstructed_image = unpatchify(patches, patch_size)
        self.assertEqual(reconstructed_image.shape, (expected_height, expected_width, 3))

    def test_colorize_masks(self):
        """
        Test the `colorize_masks` function to ensure it correctly colorizes the predicted classes.

        Author: Dániel
        """
        predicted_classes = np.array([[0, 1, 2, 3],
                                      [1, 2, 3, 0],
                                      [2, 3, 0, 1],
                                      [3, 0, 1, 2]])
        expected_output = np.array([[[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 255]],
                                    [[0, 255, 0], [255, 0, 0], [255, 255, 255], [0, 0, 255]],
                                    [[255, 0, 0], [255, 255, 255], [0, 0, 255], [0, 255, 0]],
                                    [[255, 255, 255], [0, 0, 255], [0, 255, 0], [255, 0, 0]]])
        output = colorize_masks(predicted_classes)
        np.testing.assert_array_equal(output, expected_output)

    def test_custom_colormap(self):
        """
        Test the `custom_colormap` function to ensure it returns a valid colormap.

        Author: Benjamin Graziadei
        """
        cmap = custom_colormap()
        self.assertIsInstance(cmap, LinearSegmentedColormap)

    def test_create_colored_heatmaps(self):
        """
        Test the `create_colored_heatmaps` function to ensure it creates and saves heatmaps correctly.

        Author: Benjamin Graziadei
        """
        data = np.random.rand(10, 10)
        cmap = custom_colormap()
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            create_colored_heatmaps(data, 'test.png', cmap)
            mock_savefig.assert_called_once_with('test.png', format='png', bbox_inches='tight', pad_inches=0, transparent=True)

    @patch('predictions.unpatchify')
    @patch('predictions.custom_argmax')
    @patch('cv2.imwrite')
    @patch('os.makedirs')
    def test_predict_masks(self, mock_makedirs, mock_imwrite, mock_custom_argmax, mock_unpatchify):
        """
        Test the `predict_masks` function to ensure it correctly predicts and saves masks.

        Author: Benjamin Graziadei
        """
        mock_model = MagicMock(spec=Model)
        mock_generator = MagicMock()
        mock_generator.__next__.side_effect = [
            (np.random.rand(1, 256, 256, 3), 'image1'),
            (np.random.rand(1, 256, 256, 3), 'image2'),
            StopIteration
        ]

        mock_unpatchify.side_effect = [np.random.rand(512, 512, 4), np.random.rand(512, 512, 3)]
        mock_custom_argmax.return_value = np.random.randint(0, 4, (512, 512))

        predict_masks(mock_model, False, mock_generator, 'model_name', 'data_set')
        self.assertTrue(mock_makedirs.called)
        self.assertFalse(mock_imwrite.called)
        self.assertFalse(mock_custom_argmax.called)
        self.assertFalse(mock_unpatchify.called)

    @patch('predictions.HDF5File')
    @patch('predictions.load_model')
    def test_choose_model(self, mock_load_model, mock_hdf5file):
        """
        Test the `choose_model` function to ensure it correctly loads the model.

        Author: Benjamin Graziadei
        """
        file_path = os.path.join(root, 'unittests', 'model_test.json')
        
        with open(file_path, 'r') as file:
            config_json = file.read()
            config_dict = json.loads(config_json)
        
        mock_hdf5file.return_value.__enter__.return_value.attrs = {'model_config': json.dumps(config_dict)}
        mock_model = MagicMock(spec=Model)
        mock_load_model.return_value = mock_model

        model, num_classes = choose_model('test_model.h5')
        self.assertEqual(num_classes, 4)
        self.assertIsInstance(model, Model)

    @patch('os.listdir')
    @patch('predictions.choose_model')
    @patch('predictions.test_generator')
    @patch('predictions.custom_generator')
    @patch('predictions.predict_masks')
    def test_main(self, mock_predict_masks, mock_custom_generator, mock_test_generator, mock_choose_model, mock_listdir):
        """
        Test the `main` function to ensure it correctly orchestrates the workflow.

        Author: Benjamin Graziadei
        """
        mock_choose_model.return_value = (MagicMock(spec=Model), 4)
        mock_listdir.return_value = ['mask1.png']

        with patch('builtins.input', side_effect=['model_name', 'data/processed/path']):
            main('model_name', 'data/processed/path')
            self.assertTrue(mock_predict_masks.called)
            self.assertTrue(mock_test_generator.called or mock_custom_generator.called)


if __name__ == '__main__':
    unittest.main()
