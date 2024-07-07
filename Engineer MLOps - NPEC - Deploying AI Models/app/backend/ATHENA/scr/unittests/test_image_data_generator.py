import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Adding the path to the script we want to test
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from image_data_generator import get_available_mask_subfolders, train_generator, test_generator, val_generator, custom_generator # type: ignore

# Assuming the functions are in a module named `data_generators`
from image_data_generator import train_generator, test_generator, val_generator, custom_generator, get_available_mask_subfolders

class TestImageDataGenerator(unittest.TestCase):
    @patch("image_data_generator.os.scandir")
    def test_get_available_mask_subfolders(self, mock_scandir):
        # Setup mock return value
        mock_dir_entry = MagicMock()
        mock_dir_entry.is_dir.return_value = True
        mock_dir_entry.name = "subfolder1"
        mock_scandir.return_value = [mock_dir_entry]

        # Call the function
        result = get_available_mask_subfolders("mock_mask_folder")

        # Check the result
        self.assertEqual(result, ["subfolder1"])
        mock_scandir.assert_called_once_with("mock_mask_folder")

    @patch("image_data_generator.ImageDataGenerator")
    @patch("image_data_generator.get_available_mask_subfolders")
    @patch("image_data_generator.os.path.join", side_effect=lambda *args: "/".join(args))
    def test_train_generator(self, mock_path_join, mock_get_available_mask_subfolders, mock_ImageDataGenerator):
        # Setup mock return values
        mock_get_available_mask_subfolders.return_value = ["subfolder1", "root1", "root2"]
        mock_image_gen = MagicMock()
        mock_mask_gen = MagicMock()
        mock_ImageDataGenerator.return_value = mock_image_gen
        mock_image_gen.flow_from_directory.return_value.next.return_value = np.zeros((1, 256, 256, 1))
        mock_mask_gen.flow_from_directory.return_value.next.return_value = np.zeros((1, 256, 256, 1))

        # Call the function and get the generator
        gen = train_generator({}, 1, "mock_train_path")

        # Generate one batch
        img, masks = next(gen)

        # Check the results
        self.assertEqual(img.shape, (1, 256, 256, 1))
        self.assertEqual(masks.shape, (1, 256, 256, 3))

    @patch("image_data_generator.ImageDataGenerator")
    @patch("image_data_generator.get_available_mask_subfolders")
    @patch("image_data_generator.os.path.join", side_effect=lambda *args: "/".join(args))
    def test_test_generator(self, mock_path_join, mock_get_available_mask_subfolders, mock_ImageDataGenerator):
        # Setup mock return values
        mock_get_available_mask_subfolders.return_value = ["subfolder1", "root1", "root2"]
        mock_image_gen = MagicMock()
        mock_mask_gen = MagicMock()
        mock_ImageDataGenerator.return_value = mock_image_gen
        mock_image_gen.flow_from_directory.return_value.next.return_value = np.zeros((1, 256, 256, 1))
        mock_image_gen.flow_from_directory.return_value.samples = 10
        mock_mask_gen.flow_from_directory.return_value.next.return_value = np.zeros((1, 256, 256, 1))
        mock_image_gen.flow_from_directory.return_value.filenames = ["testfile.jpg"]

        # Call the function and get the generator
        gen = test_generator(1, "mock_train_path")

        # Generate one batch
        img, filenames, masks = next(gen)

        # Check the results
        self.assertEqual(img.shape, (1, 256, 256, 1))
        self.assertEqual(filenames, 'testfile.jpg')
        self.assertEqual(masks.shape, (1, 256, 256, 3))

    @patch("image_data_generator.ImageDataGenerator")
    @patch("image_data_generator.get_available_mask_subfolders")
    @patch("image_data_generator.os.path.join", side_effect=lambda *args: "/".join(args))
    def test_val_generator(self, mock_path_join, mock_get_available_mask_subfolders, mock_ImageDataGenerator):
        # Setup mock return values
        mock_get_available_mask_subfolders.return_value = ["subfolder1", "root1", "root2"]
        mock_image_gen = MagicMock()
        mock_mask_gen = MagicMock()
        mock_ImageDataGenerator.return_value = mock_image_gen
        mock_image_gen.flow_from_directory.return_value.next.return_value = np.zeros((1, 256, 256, 1))
        mock_mask_gen.flow_from_directory.return_value.next.return_value = np.zeros((1, 256, 256, 1))

        # Call the function and get the generator
        gen = val_generator(1, "mock_train_path")

        # Generate one batch
        img, masks = next(gen)

        # Check the results
        self.assertEqual(img.shape, (1, 256, 256, 1))
        self.assertEqual(masks.shape, (1, 256, 256, 3))

    @patch("image_data_generator.ImageDataGenerator")
    @patch("image_data_generator.os.path.join", side_effect=lambda *args: "/".join(args))
    def test_custom_generator(self, mock_path_join, mock_ImageDataGenerator):
        # Setup mock return values
        mock_image_gen = MagicMock()
        mock_ImageDataGenerator.return_value = mock_image_gen
        mock_image_gen.flow_from_directory.return_value.next.return_value = np.zeros((1, 256, 256, 1))
        mock_image_gen.flow_from_directory.return_value.samples = 10
        mock_image_gen.flow_from_directory.return_value.filenames = ["testfile.jpg"]

        # Call the function and get the generator
        gen = custom_generator(1, "mock_train_path")

        # Generate one batch
        img, filenames = next(gen)

        # Check the results
        self.assertEqual(img.shape, (1, 256, 256, 1))
        self.assertEqual(filenames, 'testfile.jpg')


if __name__ == "__main__":
    unittest.main()
