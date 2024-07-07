import unittest
from unittest.mock import patch, MagicMock
import subprocess
import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline import main

class TestMainFunction(unittest.TestCase):
    """
    Unit tests for the main function in the command-line interface script.
    """

    @patch('argparse.ArgumentParser.parse_args')
    @patch('subprocess.run')
    @patch('model_training.train')
    @patch('model_evaluation.main')
    @patch('model_saving.model_saving')
    @patch('predictions.main')
    @patch('landmark_detection.Landmark_detecion')
    def test_process_data(self, mock_landmark_detection, mock_predictions, mock_model_saving, mock_model_evaluation, mock_train, mock_subprocess_run, mock_parse_args):
        """
        Test the data processing functionality of the main function.
        """
        mock_parse_args.return_value = argparse.Namespace(
            process=True, process_mode='custom', dir='data', train=False, train_path=None, depth=None, predict_path=None, model_name=None
        )

        with patch('builtins.print') as mocked_print:
            main()

        mock_subprocess_run.assert_called_once_with(['python', 'scr/process_data.py', '--mode', 'custom', '--dir', 'data'])
        mocked_print.assert_not_called()
        mock_train.assert_not_called()
        mock_model_evaluation.assert_not_called()
        mock_model_saving.assert_not_called()
        mock_predictions.assert_not_called()
        mock_landmark_detection.assert_not_called()

    @patch('argparse.ArgumentParser.parse_args')
    @patch('model_training.train')
    @patch('model_evaluation.main')
    @patch('model_saving.model_saving')
    def test_train_model(self, mock_model_saving, mock_model_evaluation, mock_train, mock_parse_args):
        """
        Test the model training functionality of the main function.
        """
        mock_parse_args.return_value = argparse.Namespace(
            process=False, process_mode=None, dir=None, train=True, train_path='train_data', depth=1, predict_path=None, model_name='test_model'
        )

        mock_train.return_value = (MagicMock(), MagicMock())
        mock_model_evaluation.return_value = MagicMock()

        with patch('builtins.print') as mocked_print:
            main()

        mock_train.assert_called_once_with(1, 'train_data')
        mock_model_evaluation.assert_called_once_with(mock_train.return_value[0], 'train_data')
        mock_model_saving.assert_called_once_with(mock_train.return_value[0], mock_model_evaluation.return_value, 'test_model')
        mocked_print.assert_not_called()

    @patch('argparse.ArgumentParser.parse_args')
    @patch('predictions.main')
    @patch('landmark_detection.Landmark_detecion')
    def test_predict_path(self, mock_landmark_detection, mock_predictions, mock_parse_args):
        """
        Test the prediction functionality of the main function.
        """
        mock_parse_args.return_value = argparse.Namespace(
            process=False, process_mode=None, dir=None, train=False, train_path=None, depth=None, predict_path='predict_data', model_name='test_model'
        )

        mock_predictions.return_value = 'predictions_path'

        with patch('builtins.print') as mocked_print:
            with patch('predictions.choose_model', return_value=(MagicMock(), 4)):
                main()

        mock_predictions.assert_called_once_with('test_model', 'predict_data')
        mock_landmark_detection.assert_called_once_with('predictions_path')
        mocked_print.assert_called_once_with("Predictions saved to predictions_path")

    @patch('argparse.ArgumentParser.parse_args')
    def test_train_missing_args(self, mock_parse_args):
        """
        Test the behavior when required training arguments are missing.
        """
        mock_parse_args.return_value = argparse.Namespace(
            process=False, process_mode=None, dir=None, train=True, train_path=None, depth=None, predict_path=None, model_name=None
        )

        with self.assertRaises(SystemExit):
            main()

    @patch('argparse.ArgumentParser.parse_args')
    def test_invalid_depth(self, mock_parse_args):
        """
        Test the behavior when an invalid depth value is provided.
        """
        mock_parse_args.return_value = argparse.Namespace(
            process=False, process_mode=None, dir=None, train=True, train_path='train_data', depth=2, predict_path=None, model_name='test_model'
        )

        with self.assertRaises(SystemExit):
            main()

if __name__ == "__main__":
    unittest.main()
