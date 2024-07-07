# Written by Kian

# Standard library imports
import argparse
import io
import os
import sys
import unittest
from unittest.mock import MagicMock, mock_open, patch

# Third-party imports
import cv2
import numpy as np

# Add the parent directory to the sys.path to allow relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Custom imports
from process_image import crop_dimensions, padder, save_patches, process_image, main

class TestImageProcessing(unittest.TestCase):
    def setUp(self):
        # Set up a dummy grayscale image with a larger circle
        self.img = np.zeros((500, 500), dtype=np.uint8)
        cv2.circle(self.img, (250, 250), 220, (255), -1)
        self.patch_size = 256

    def test_crop_dimensions(self):
        y_min, y_max, x_min, x_max = crop_dimensions(self.img)
        # Allowing a margin of error due to morphological operations
        self.assertAlmostEqual(y_min, 30, delta=10)
        self.assertAlmostEqual(y_max, 470, delta=10)
        self.assertAlmostEqual(x_min, 30, delta=10)
        self.assertAlmostEqual(x_max, 470, delta=10)
    
    def test_padder(self):
        padded_img = padder(self.img, self.patch_size)
        step_size = int(self.patch_size / 8) * 7
        num_patches_h = (self.img.shape[0] + step_size - self.patch_size) // step_size + 1
        num_patches_w = (self.img.shape[1] + step_size - self.patch_size) // step_size + 1
        expected_shape = ((num_patches_h - 1) * step_size + self.patch_size,
                          (num_patches_w - 1) * step_size + self.patch_size)
        self.assertEqual(padded_img.shape[:2], expected_shape)

    @patch("os.makedirs")
    @patch("cv2.imwrite")
    def test_save_patches(self, mock_imwrite, mock_makedirs):
        img_patches = np.random.rand(4, self.patch_size, self.patch_size, 1).astype(np.float32)
        fpath = "raw/test_image.png"
        save_patches(fpath, img_patches)
        
        self.assertEqual(mock_makedirs.call_count, 1)
        self.assertEqual(mock_imwrite.call_count, img_patches.shape[0])


    @patch("cv2.imread", return_value=np.zeros((500, 500), dtype=np.uint8))
    @patch("process_image.save_patches")
    def test_process_image(self, mock_save_patches, mock_cv2_imread):
        img_path = "raw/test_image.png"
        process_image(img_path, self.patch_size)
        
        mock_cv2_imread.assert_called_once_with(img_path, 0)
        mock_save_patches.assert_called_once()

    @patch('process_image.process_image')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main(self, mock_parse_args, mock_process_image):
        # Set up the mock arguments
        mock_parse_args.return_value = argparse.Namespace(imagepath='test_image_path')

        # Capture the output during the main function execution
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            main()
        
        # Ensure the process_image function was called with the correct arguments
        mock_process_image.assert_called_once_with('test_image_path')


    def tearDown(self):
        # Clean up any necessary files or resources here
        pass

if __name__ == '__main__':
    unittest.main()
