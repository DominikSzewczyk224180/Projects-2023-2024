import logging
from typing import Any, List, Optional, Tuple, Union

import keras_tuner as kt
import tensorflow as tf

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def compile_model(
    model: tf.keras.Model, optimizer: str, learning_rate: float, metrics: List
) -> tf.keras.Model:
    """
    Writen by Matey

    Compile the U-Net model with specified optimizer and learning rate.

    Parameters:
    model (Model): The U-Net model to be compiled.
    optimizer (str): Optimizer to use for model training.
    learning_rate (float): Learning rate for the optimizer.

    Returns:
    Model: Compiled U-Net model.
    """
    logging.info(f"Optimizer={optimizer}, learning_rate={learning_rate}")

    if optimizer == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=metrics)
    logging.info("Model successfully compiled")

    return model


def train_and_evaluate(
    model: tf.keras.Model,
    train_data: Any,
    val_data: Any,
    epochs: int,
    callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    batch_size: int = 32,
    optimizer: str = "adam",
    learning_rate: float = 1e-3,
    metrics: List[Union[str, tf.keras.metrics.Metric]] = ["accuracy"],
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History, List[float]]:
    """
    Writen by Matey

    This function compiles the model, then it trains and
    evaluates the given model.

    Parameters
    ----------
    model : tf.keras.Model
        The machine learning model to be trained and evaluated.
    train_data : Any
        The training data.
    val_data : Any
        The validation data.
    epochs : int
        The number of epochs to train the model.
    callbacks : list of tf.keras.callbacks.Callback, optional
        List of callback functions to apply during training (default is None).
    batch_size : int, optional
        The number of samples per batch (default is 32).
    optimizer : str, optional
        The optimizer to use for training (default is 'adam').
    learning_rate : float, optional
        The learning rate for the optimizer (default is 1e-3).

    Returns
    -------
    model : tf.keras.Model
        The trained model.
    history : tf.keras.callbacks.History
        The training history.
    evaluation : list
        The evaluation results.
    """

    model = compile_model(model, optimizer, learning_rate, metrics)

    # Calculate steps per epoch for training and validation
    steps_train = 16900 / batch_size
    steps_valid = 2197 / batch_size

    # Train the model
    history = model.fit(
        train_data,
        epochs=epochs,
        steps_per_epoch=steps_train,
        validation_data=val_data,
        validation_steps=steps_valid,
        callbacks=callbacks,
    )

    # Evaluate the model on the validation data
    evaluation = model.evaluate(val_data, steps=steps_valid)
    logging.info(
        f"Model training completed! The validation accuracy is {evaluation[1]}"
    )

    return model, history, evaluation


class MyTuner(kt.BayesianOptimization):
    def __init__(
        self,
        model,
        train_data,
        valid_data,
        epochs,
        early_stopping=None,
        metrics=["accuracy"],
        *args,
        **kwargs,
    ):
        """
        Written by Matey

        Custom tuner class for hyperparameter optimization.

        Parameters:
        model (function): Model building function.
        train_data (Any): Training data.
        valid_data (Any): Validation data.
        epochs (int): Number of epochs to train the model.
        early_stopping (tf.keras.callbacks.Callback): Early stopping callback.
        """
        super(MyTuner, self).__init__(*args, **kwargs)
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.metrics = metrics

        logging.info("Initialized MyTuner")

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        """
        Run a single trial for hyperparameter optimization.

        Parameters:
        trial (kt.HyperParameters): Hyperparameter trial object.
        """
        hp = trial.hyperparameters

        # Create the model with the specified hyperparameters

        evaluation = train_and_evaluate(
            self.model,
            self.train_data,
            self.valid_data,
            self.epochs,
            early_stopping=self.early_stopping,
            batch_size=hp.Choice("batch_size", values=[32, 64, 128, 256]),
            learning_rate=hp.Float(
                "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
            ),
            optimizer=hp.Choice("optimizer", values=["adam", "sgd"]),
            metrics=self.metrics,
        )

        # Returning the accuracy
        return evaluation[2][1]
