import os
import sys
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loading import load_data, load_data_uri, extract_shape

path = '../data/processed/NPEC'


class Test_data_loading(unittest.TestCase):

    def test_load_data(self):

        data_loader = load_data()

        self.assertIsNotNone(data_loader)

    @patch('image_data_generator.train_generator')
    def test_extract_shape(self, mock_train_gen):
        # Mock return value: a tuple (images, masks)
        # Create a mock image (128x128 RGB)
        mock_image = np.random.randint(0, 256, (1, 128, 128, 3), dtype=np.uint8)
        mock_mask = np.random.randint(0, 256, (1, 128, 128, 1), dtype=np.uint8)

        # Mock the generator to return the mock image and mask
        mock_generator = MagicMock()
        mock_generator.__next__.return_value = (mock_image, mock_mask)
        mock_train_gen.return_value = mock_generator

        # Now test your function
        result = extract_shape(mock_train_gen())
        expected_shape = (128, 3)  # Expected shape (height, channels)

        self.assertEqual(result, expected_shape)

    @patch('data_loading.os.listdir')
    @patch('data_loading.Image.open')
    def test_load_data_uri(self, mock_open, mock_listdir):
        # Setup mock data
        mock_listdir.side_effect = [
            ['image1.png', 'image2.png'],  # image files
            ['image1_root.png', 'image2_root.png']  # mask files
        ]

        mock_image = MagicMock()
        mock_image.convert.return_value = mock_image
        mock_image.resize.return_value = mock_image
        mock_image.__array__ = MagicMock(return_value=np.zeros((256, 256)))

        mock_open.return_value = mock_image

        # Create generator
        generator = load_data_uri('image_uri', 'mask_uri', target_size=(256, 256), batch_size=2)

        # Generate one batch
        images, masks = next(generator)

        # Assert that the generator yields the correct shape
        self.assertEqual(images.shape, (2, 256, 256, 1))
        self.assertEqual(masks.shape, (2, 256, 256, 1))

        # Assert that images and masks are correctly normalized
        self.assertTrue((images == 0).all())
        self.assertTrue((masks == 0).all())

        # Assert os.listdir was called with the correct arguments
        mock_listdir.assert_any_call('image_uri')
        mock_listdir.assert_any_call('mask_uri')

        # Assert PIL.Image.open was called for each image and mask
        self.assertEqual(mock_open.call_count, 4)



if __name__ == '__main__':
    unittest.main()
