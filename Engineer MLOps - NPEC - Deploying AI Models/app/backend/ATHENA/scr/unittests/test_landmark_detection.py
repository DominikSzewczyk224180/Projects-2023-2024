# Unit tests for landmark_detection.py by Dániel
# Import necessary modules
import os
import sys
import unittest
from typing import Any, List, Tuple
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the functions to be tested
from landmark_detection import get_plants


class TestGetPlantsFunction(unittest.TestCase):
    """
    Unit tests for the landmark detection functions defined in landmark_detection.py.
    """

    @patch("landmark_detection.clean_root_mask")
    @patch("cv2.imread")
    @patch(
        "skimage.morphology.skeletonize",
        return_value=np.ones((2806, 2806), dtype=np.uint8),
    )
    @patch("skan.Skeleton")
    @patch("skan.summarize")
    def test_get_plants(
        self,
        mock_summarize: Any,
        mock_skeleton: Any,
        mock_skeletonize: Any,
        mock_cv2_imread: Any,
        mock_clean_root_mask: Any,
    ) -> None:
        """
        Test the get_plants function to ensure it processes the image correctly and returns
        the expected plants and skeleton data.

        This test mocks the dependencies, creates a cleaned mask, and checks if the function
        returns the correct types and properties for plants and skeleton data.

        :param mock_summarize: Mock for skan.summarize
        :param mock_skeleton: Mock for skan.Skeleton
        :param mock_skeletonize: Mock for skimage.morphology.skeletonize
        :param mock_cv2_imread: Mock for cv2.imread
        :param mock_clean_root_mask: Mock for landmark_detection.clean_root_mask
        """
        # Mock the cleaned mask
        mock_clean_root_mask.return_value = np.ones((2806, 2806), dtype=np.uint8) * 255

        # Create mock skeleton data
        skeleton_data_mock = pd.DataFrame(
            {
                "skeleton-id": [1, 2, 3, 4, 5],
                "node-id-src": [1, 1, 1, 1, 1],
                "node-id-dst": [2, 2, 2, 2, 2],
                "coord-src-0": [10, 20, 30, 40, 50],
                "coord-src-1": [15, 25, 35, 45, 55],
                "coord-dst-0": [20, 30, 40, 50, 60],
                "coord-dst-1": [25, 35, 45, 55, 65],
            }
        )
        mock_summarize.return_value = skeleton_data_mock

        # Call the function
        plants, skeleton_data = get_plants("dummy_path")

        # Assertions
        self.assertTrue(
            isinstance(skeleton_data, pd.DataFrame),
            "Error: skeleton_data is not a pandas DataFrame",
        )
        self.assertTrue(
            skeleton_data["skeleton-id"].nunique() <= 5,
            "Error: skeleton_data has more than 5 unique skeleton-ids",
        )
        self.assertTrue(
            isinstance(plants, np.ndarray), "Error: plants is not a numpy array"
        )
        self.assertTrue(len(plants) <= 5, "Error: plants has more than 5 unique rows")

    @patch("landmark_detection.plt.show")
    @patch("landmark_detection.cv2.circle", wraps=cv2.circle)
    def test_visualistion_of_roots_multiple_plants(
        self, mock_cv2_circle: Any, mock_plt_show: Any
    ) -> None:
        """
        Test the visualistion_of_roots function to ensure it handles multiple plants correctly.

        This test mocks the cv2.circle and plt.show functions, creates mock data for multiple plants
        and skeleton data, and checks if the function processes the data correctly.

        :param mock_cv2_circle: Mock for cv2.circle
        :param mock_plt_show: Mock for plt.show
        """
        # Mock data
        plants: List[int] = [1, 2, 3, 4]  # Four plant IDs

        # Example skeleton data
        skeleton_data = pd.DataFrame(
            {
                "skeleton-id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
                "node-id-src": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
                "node-id-dst": [2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5],
                "coord-src-0": [
                    10,
                    20,
                    30,
                    40,
                    10,
                    20,
                    30,
                    40,
                    10,
                    20,
                    30,
                    40,
                    10,
                    20,
                    30,
                    40,
                ],
                "coord-src-1": [
                    50,
                    60,
                    70,
                    80,
                    50,
                    60,
                    70,
                    80,
                    50,
                    60,
                    70,
                    80,
                    50,
                    60,
                    70,
                    80,
                ],
                "coord-dst-0": [
                    15,
                    25,
                    35,
                    45,
                    15,
                    25,
                    35,
                    45,
                    15,
                    25,
                    35,
                    45,
                    15,
                    25,
                    35,
                    45,
                ],
                "coord-dst-1": [
                    55,
                    65,
                    75,
                    85,
                    55,
                    65,
                    75,
                    85,
                    55,
                    65,
                    75,
                    85,
                    55,
                    65,
                    75,
                    85,
                ],
            }
        )

        # Mock plant image
        plant_img = np.zeros((2806, 2806, 1), dtype=np.uint8)

        # Call the function
        code_measurements = visualistion_of_roots(plants, skeleton_data, plant_img)

        # Assertions
        self.assertTrue(isinstance(code_measurements, pd.DataFrame))
        self.assertEqual(code_measurements.shape[0], 4)
        self.assertGreater(mock_cv2_circle.call_count, 0)


if __name__ == "__main__":
    unittest.main()
