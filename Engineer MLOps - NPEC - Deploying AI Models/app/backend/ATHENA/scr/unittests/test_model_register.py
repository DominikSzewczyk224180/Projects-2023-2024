import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the function to be tested
from model_register import register_model_if_accuracy_above_threshold

class TestRegisterModel(unittest.TestCase):

    @patch('builtins.open', new_callable=mock_open, read_data='0.45')
    @patch('model_register.MLClient')
    def test_model_not_register_below_threshold(self, mock_mlclient, mock_open):
        model_path = "dummy_model_path"
        accuracy_folder = "dummy_accuracy_folder"
        threshold = 0.5

        with patch('model_register.os.path.join', return_value='dummy_accuracy_folder/accuracy.txt'):
            register_model_if_accuracy_above_threshold(model_path, accuracy_folder, threshold)
        
        # Assert that the model client was never created since accuracy is below threshold
        mock_mlclient.assert_not_called()

    @patch('builtins.open', new_callable=mock_open, read_data='0.75')
    @patch('model_register.MLClient')
    def test_register_model_if_accuracy_above_threshold(self, mock_mlclient, mock_open):
        model_path = "dummy_model_path"
        accuracy_folder = "dummy_accuracy_folder"
        threshold = 0.5

        mock_mlclient_instance = MagicMock()
        mock_mlclient.return_value = mock_mlclient_instance

        with patch('model_register.os.path.join', return_value='dummy_accuracy_folder/accuracy.txt'):
            register_model_if_accuracy_above_threshold(model_path, accuracy_folder, threshold)
        
        # Assert that the model client was called since accuracy is above threshold
        mock_mlclient.assert_called_once()
        mock_mlclient_instance.models.create_or_update.assert_called_once()

if __name__ == "__main__":
    unittest.main()
