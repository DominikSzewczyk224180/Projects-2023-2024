import logging
import os
import sys
from typing import List, Tuple, Union

import tensorflow as tf

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def save_model(model: tf.keras.Model, folder_path: str, model_name: str) -> str:
    """
    Save the Keras model to the specified folder path.

    Parameters
    ----------
    model : tf.keras.Model
        The Keras model to be saved.
    folder_path : str
        The directory where the model will be saved.

    Returns
    -------
    str
        The file path where the model is saved.

    Author: Dominik Szewczyk
    """
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"{model_name}.h5")
    model.save(file_path)
    logging.info(f"Model saved successfully at {file_path}")
    return file_path


def get_layer_description(layer: tf.keras.layers.Layer) -> str:
    """
    Get a description of the Keras layer.

    Parameters
    ----------
    layer : tf.keras.layers.Layer
        The Keras layer to describe.

    Returns
    -------
    str
        A description of the layer.

    Author: Dominik Szewczyk
    """
    layer_type = type(layer).__name__
    if layer_type == "Conv2D":
        filters = layer.filters
        kernel_size = layer.kernel_size
        activation = layer.activation.__name__
        return (
            f"Convolutional layer: {filters} filters, "
            f"kernel size {kernel_size}, activation function '{activation}'"
        )
    elif layer_type == "MaxPooling2D":
        pool_size = layer.pool_size
        return f"MaxPooling layer: pool size {pool_size}"
    elif layer_type == "Flatten":
        return "Flatten layer: flattens the input"
    elif layer_type == "Dense":
        units = layer.units
        act = layer.activation.__name__
        return f"Dense layer: {units} units, activation function '{act}'"
    else:
        return f"Layer type: {layer_type}"


def create_model_description(model: tf.keras.Model) -> str:
    """
    Create a description of the Keras model architecture.

    Parameters
    ----------
    model : tf.keras.Model
        The Keras model to describe.

    Returns
    -------
    str
        A description of the model architecture.

    Author: Dominik Szewczyk
    """
    model_description = "Model Description:\n\n"
    model_description += "This model architecture consists of the following layers:\n\n"

    for layer in model.layers:
        layer_description = get_layer_description(layer)
        model_description += f"- {layer_description}\n"

    model_description += f"\n\nNumber of Parameters: {model.count_params()}\n"
    return model_description


def create_full_model_architecture(model: tf.keras.Model) -> str:
    """
    Get the full architecture of the Keras model in JSON format.

    Parameters
    ----------
    model : tf.keras.Model
        The Keras model to describe.

    Returns
    -------
    str
        The JSON string representation of the model architecture.

    Author: Dominik Szewczyk
    """
    return model.to_json()


def training_configuration(model: tf.keras.Model, metrics: dict) -> str:
    """
    Get the training configuration and validation metrics of the Keras model.

    Parameters
    ----------
    model : tf.keras.Model
        The Keras model to describe.

    Returns
    -------
    str
        A description of the training configuration and validation metrics.

    Author: Dominik Szewczyk
    """
    optimizer_config = model.optimizer.get_config()
    loss_function = model.loss

    training_config = "Training Configuration:\n\n"
    training_config += f"Optimizer: {optimizer_config['name']}\n"
    training_config += (
        f"Learning Rate: {float(optimizer_config['learning_rate']):.6f}\n"
    )
    training_config += f"Loss Function: {loss_function}\n"

    training_config += "\nValidation Metrics:\n"
    for metric, value in metrics.items():
        training_config += f"{metric}: {value:.4f}\n"

    return training_config


def save_selected_info(
    model: tf.keras.Model, folder_path: str, choices: List[int], metrics: dict, model_name: str
) -> None:
    """
    Save selected information about the Keras model.

    Parameters
    ----------
    model : tf.keras.Model
        The Keras model.
    folder_path : str
        The directory where the information will be saved.
    choices : List[int]
        The list of choices indicating what information to save.

    Author: Dominik Szewczyk
    """
    combined_info = ""

    if 1 in choices:
        save_model(model, folder_path, model_name)

    if 2 in choices:
        combined_info += create_model_description(model) + "\n\n"

    if 3 in choices:
        combined_info += "Model Architecture (JSON):\n\n"
        combined_info += create_full_model_architecture(model) + "\n\n"

    if 4 in choices:
        combined_info += training_configuration(model, metrics) + "\n\n"

    if combined_info:
        file_path = os.path.join(folder_path, f"{model_name}_info.txt")
        with open(file_path, "w") as file:
            file.write(combined_info)
        logging.info(f"Model information saved successfully at {file_path}")


def model_saving(model: tf.keras.Model, metrics: dict, model_name: str) -> None:
    """
    Prompt the user to save the Keras model and its related information.

    Parameters
    ----------
    model : tf.keras.Model
        The Keras model to be saved.

    Author: Dominik Szewczyk
    """
    folder_path = root_dir + "/models/"

    user_input = input(
        "Enter the number of things you want to save separated by commas "
        "(e.g., 1,2,3,4):\n"
        "1 for saving model\n"
        "2 for model description\n"
        "3 for full complex model architecture\n"
        "4 for training configuration\n"
    )

    choices = [int(choice.strip()) for choice in user_input.split(",")]

    if model is None:
        logging.error(
            "No model defined. Please load or define your model in the script."
        )
        return

    save_selected_info(model, folder_path, choices, metrics, model_name)