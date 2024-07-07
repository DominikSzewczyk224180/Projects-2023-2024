import logging

import matplotlib.pyplot as plt
import mlflow
import tensorflow as tf

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def plotting_metrics(history):

    mlflow.log_metric("train_loss", history.history["loss"][-1])
    mlflow.log_metric("train_accuracy", history.history["accuracy"][-1])
    mlflow.log_metric("train_f1", history.history["f1"][-1])
    mlflow.log_metric("train_iou", history.history["iou"][-1])
    mlflow.log_metric("val_loss", history.history["val_loss"][-1])
    mlflow.log_metric("val_accuracy", history.history["val_accuracy"][-1])
    mlflow.log_metric("val_f1", history.history["val_f1"][-1])
    mlflow.log_metric("val_iou", history.history["val_iou"][-1])

    fig = plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.plot(history.history["f1"], label="train_f1")
    plt.plot(history.history["val_f1"], label="val_f1")
    plt.plot(history.history["iou"], label="train_iou")
    plt.plot(history.history["val_iou"], label="val_iou")
    plt.title("Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy/F1/iou")
    plt.legend(loc="lower left")
    mlflow.log_figure(fig, "metrics.png")

class TrainingLogger(tf.keras.callbacks.Callback):
    """
    Writen by Matey

    A custom callback for logging training progress.

    Methods
    -------
    on_train_begin(logs=None)
        Logs the start of training.
    on_epoch_end(epoch, logs=None)
        Logs the end of each epoch.
    on_train_end(logs=None)
        Logs the end of training.
    log_epoch_end(epoch, logs)
        Logs detailed metrics at the end of each epoch.

    Attributes
    ----------
    epoch_count : int
        The count of completed epochs.
    """

    def __init__(self):
        self.epoch_count = 0

    def on_train_begin(self, logs=None):
        """
        Logs the start of training.

        Parameters
        ----------
        logs : dict, optional
            Additional logs (default is None).
        """
        self.epoch_count = 0
        logging.info("Training started.")

    def on_epoch_end(self, epoch, logs=None):
        """
        Logs the end of each epoch and updates the epoch count.

        Parameters
        ----------
        epoch : int
            The index of the epoch.
        logs : dict, optional
            Additional logs containing metrics (default is None).
        """
        logs = logs or {}
        self.epoch_count += 1
        self.log_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        """
        Logs the end of training.

        Parameters
        ----------
        logs : dict, optional
            Additional logs (default is None).
        """
        logging.info(f"Training finished after {self.epoch_count} epochs.")

    def log_epoch_end(self, epoch, logs):
        """
        Logs detailed metrics at the end of each epoch.

        Parameters
        ----------
        epoch : int
            The index of the epoch.
        logs : dict
            Logs containing metrics.
        """
        loss = logs.get("loss")
        accuracy = logs.get("accuracy")
        val_loss = logs.get("val_loss")
        val_accuracy = logs.get("val_accuracy")
        f1 = logs.get("f1")
        val_f1 = logs.get("val_f1")
        iou = logs.get("iou")
        val_iou = logs.get("val_iou")
        logging.info(
            f"Epoch {epoch+1} - "
            f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, "
            f"F1: {f1:.4f}, IoU: {iou:.4f}, "
            f"Val_loss: {val_loss:.4f}, Val_accuracy: {val_accuracy:.4f}, "
            f"Val_f1: {val_f1:.4f}, Val_IoU: {val_iou:.4f}"
        )
