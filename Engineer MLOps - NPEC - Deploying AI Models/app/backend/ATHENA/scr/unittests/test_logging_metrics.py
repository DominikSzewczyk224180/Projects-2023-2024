import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logging_metrics import TrainingLogger, plotting_metrics

class TestTrainingLogger(unittest.TestCase):

    @patch('logging_metrics.logging.info')
    def test_on_train_begin(self, mock_logging_info):
        callback = TrainingLogger()
        callback.on_train_begin()

        self.assertEqual(callback.epoch_count, 0)
        mock_logging_info.assert_called_once_with('Training started.')

    @patch('logging_metrics.mlflow.log_metric')
    @patch('logging_metrics.mlflow.log_figure')
    @patch('logging_metrics.plt.figure')
    @patch('logging_metrics.plt.plot')
    @patch('logging_metrics.plt.title')
    @patch('logging_metrics.plt.xlabel')
    @patch('logging_metrics.plt.ylabel')
    @patch('logging_metrics.plt.legend')
    def test_plotting_metrics(self, mock_legend, mock_ylabel, mock_xlabel, mock_title, mock_plot, mock_figure, mock_log_figure, mock_log_metric):
        history = MagicMock()
        history.history = {
            "loss": [0.1, 0.2, 0.3],
            "accuracy": [0.9, 0.8, 0.7],
            "f1": [0.88, 0.85, 0.82],
            "iou": [0.75, 0.7, 0.65],
            "val_loss": [0.2, 0.25, 0.3],
            "val_accuracy": [0.85, 0.82, 0.8],
            "val_f1": [0.84, 0.8, 0.78],
            "val_iou": [0.7, 0.65, 0.6]
        }

        plotting_metrics(history)

        # Verify logging of metrics
        mock_log_metric.assert_any_call("train_loss", 0.3)
        mock_log_metric.assert_any_call("train_accuracy", 0.7)
        mock_log_metric.assert_any_call("train_f1", 0.82)
        mock_log_metric.assert_any_call("train_iou", 0.65)
        mock_log_metric.assert_any_call("val_loss", 0.3)
        mock_log_metric.assert_any_call("val_accuracy", 0.8)
        mock_log_metric.assert_any_call("val_f1", 0.78)
        mock_log_metric.assert_any_call("val_iou", 0.6)

        # Verify plotting
        mock_figure.assert_called_once()
        mock_plot.assert_any_call([0.1, 0.2, 0.3], label="train_loss")
        mock_plot.assert_any_call([0.2, 0.25, 0.3], label="val_loss")
        mock_plot.assert_any_call([0.9, 0.8, 0.7], label="train_accuracy")
        mock_plot.assert_any_call([0.85, 0.82, 0.8], label="val_accuracy")
        mock_plot.assert_any_call([0.88, 0.85, 0.82], label="train_f1")
        mock_plot.assert_any_call([0.84, 0.8, 0.78], label="val_f1")
        mock_plot.assert_any_call([0.75, 0.7, 0.65], label="train_iou")
        mock_plot.assert_any_call([0.7, 0.65, 0.6], label="val_iou")
        mock_title.assert_called_once_with("Metrics")
        mock_xlabel.assert_called_once_with("Epoch")
        mock_ylabel.assert_called_once_with("Loss/Accuracy/F1/iou")
        mock_legend.assert_called_once_with(loc="lower left")

        # Verify logging of figure
        mock_log_figure.assert_called_once()

    @patch('logging_metrics.logging.info')
    def test_on_epoch_end(self, mock_logging_info):
        callback = TrainingLogger()
        callback.log_epoch_end = MagicMock()

        logs = {
            'loss': 0.1,
            'accuracy': 0.9,
            'val_loss': 0.2,
            'val_accuracy': 0.85,
            'f1': 0.88,
            'val_f1': 0.84,
            'iou': 0.75,
            'val_iou': 0.7
        }

        callback.on_epoch_end(epoch=0, logs=logs)

        self.assertEqual(callback.epoch_count, 1)
        callback.log_epoch_end.assert_called_once_with(0, logs)

    @patch('logging_metrics.logging.info')
    def test_on_train_end(self, mock_logging_info):
        callback = TrainingLogger()
        callback.epoch_count = 5
        callback.on_train_end()

        mock_logging_info.assert_called_once_with('Training finished after 5 epochs.')

    @patch('logging_metrics.logging.info')
    def test_log_epoch_end(self, mock_logging_info):
        callback = TrainingLogger()

        logs = {
            'loss': 0.1,
            'accuracy': 0.9,
            'val_loss': 0.2,
            'val_accuracy': 0.85,
            'f1': 0.88,
            'val_f1': 0.84,
            'iou': 0.75,
            'val_iou': 0.7
        }

        callback.log_epoch_end(epoch=0, logs=logs)

        mock_logging_info.assert_called_once_with(
            'Epoch 1 - Loss: 0.1000, Accuracy: 0.9000, F1: 0.8800, IoU: 0.7500, '
            'Val_loss: 0.2000, Val_accuracy: 0.8500, Val_f1: 0.8400, Val_IoU: 0.7000'
        )

if __name__ == '__main__':
    unittest.main()
