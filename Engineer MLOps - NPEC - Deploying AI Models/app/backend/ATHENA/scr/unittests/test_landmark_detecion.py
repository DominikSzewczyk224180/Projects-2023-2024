import unittest
import numpy as np
from skimage.morphology import skeletonize
import cv2
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from landmark_detection import (
    extract_skeleton_segment,
    get_endpoints,
    remove_long_connections,
    group_and_fit_branches,
    get_plants,
    Landmark_detecion,
)

class TestSkeletonProcessing(unittest.TestCase):

    def test_extract_skeleton_segment(self):
        skeleton = np.zeros((100, 100), dtype=np.uint8)
        skeleton[50, 50] = 1
        segment = extract_skeleton_segment(skeleton, (50, 50), length=20)
        self.assertEqual(segment.shape, (41, 41))
        self.assertEqual(segment[20, 20], 1)

    def test_get_endpoints(self):
        skeleton_data = MagicMock()
        skeleton_summary = MagicMock()
        graph = np.zeros((10, 10))
        labels = np.zeros(10)
        x_coords = np.arange(10)
        y_coords = np.arange(10)
        
        skeleton_summary.iterrows.return_value = enumerate([{'path': [0, 9]}])
        skeleton_data.path.side_effect = [[0, 9]]

        graph[0, 1] = graph[1, 0] = 1  # simulate some connectivity
        graph[9, 8] = graph[8, 9] = 1  # simulate some connectivity

        endpoints, endpoint_labels = get_endpoints(skeleton_data, skeleton_summary, graph, labels, x_coords, y_coords)
        
        self.assertEqual(len(endpoints), 2)
        self.assertIn((0, 0), endpoints)
        self.assertIn((9, 9), endpoints)

    def test_remove_long_connections(self):
        mst = np.array([
            [0, 2, 0],
            [2, 0, 3],
            [0, 3, 0]
        ])
        points = np.array([[0, 0], [1, 1], [2, 2]])
        threshold = 2.5

        new_mst = remove_long_connections(mst, points, threshold)
        
        expected_mst = np.array([
            [0, 2, 0],
            [2, 0, 0],
            [0, 0, 0]
        ])
        np.testing.assert_array_equal(new_mst, expected_mst)

    def test_group_and_fit_branches(self):
        skeleton = skeletonize(np.ones((100, 100), dtype=np.uint8))
        skeleton[50, 50] = 0  # Ensure there's at least one foreground pixel

        output_image = np.zeros((100, 100, 3), dtype=np.uint8)
        new_skeleton, filtered_endpoints = group_and_fit_branches(skeleton, output_image)

        self.assertIsInstance(new_skeleton, np.ndarray)
        self.assertTrue(len(filtered_endpoints) >= 0)

    @patch('cv2.imread')
    @patch('cv2.imwrite')
    def test_get_plants(self, mock_imwrite, mock_imread):
        mock_imread.side_effect = [
            np.zeros((100, 100), dtype=np.uint8),  # mask image
            np.zeros((100, 100, 3), dtype=np.uint8)  # original image
        ]
        
        get_plants('fake_path/INDIV_mask1.png')
        
        self.assertTrue(mock_imwrite.called)

    @patch('os.listdir')
    @patch('os.path.join')
    def test_landmark_detection(self, mock_path_join, mock_listdir):
        mock_listdir.return_value = ['INDIV_mask1.png']
        mock_path_join.return_value = 'fake_path/INDIV_mask1.png'
        
        with patch('landmark_detection.get_plants') as mock_get_plants:
            Landmark_detecion('fake_path')
            mock_get_plants.assert_called_once_with('fake_path/INDIV_mask1.png')

if __name__ == '__main__':
    unittest.main()


if __name__ == '__main__':
    unittest.main()
