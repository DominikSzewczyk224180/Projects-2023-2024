import argparse
import os
import sys
import mlflow
import tensorflow as tf
import logging
from typing import Optional
from model_architectures import build_unet_model
from model_hyperparameter_tuning import MyTuner, train_and_evaluate
from model_saving import save_model
from metric import f1, iou
from data_loading import load_data, load_data_uri


def evaluate(model_path: str,
             use_uri: bool = False,
             test_images_path: Optional[str] = None,
             test_masks_path: Optional[str] = None,
             output_dir: str = "outputs") -> None:
    """
    from the self-study material - modified by Matey

    Evaluate a trained model on test data.

    Parameters
    ----------
    model_path : str
        The path to the trained model file.
    use_uri : bool, optional
        Flag to indicate whether to load data using URIs (default is False).
    test_images_path : str, optional
        The path to the test images (used if `use_uri` is True).
    test_masks_path : str, optional
        The path to the test masks (used if `use_uri` is True).
    output_dir : str, optional
        The directory to save the evaluation results (default is "outputs").

    Returns
    -------
    None

    This function performs the following steps:

    1. Configures logging.
    2. Loads the test data based on the `use_uri` flag.
    3. Loads the trained model.
    4. Evaluates the model using the test data.
    5. Logs the evaluation metrics.
    6. Saves the test accuracy to a file in the specified output directory.
    """

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load test data based on the use_uri flag
    if not use_uri:
        path = '../ATHENA/data/processed/NPEC'
        _, _, test = load_data(13, path)
    else:
        test = load_data_uri(test_images_path, test_masks_path)

    logging.info("Data is loaded")

    # Load the trained model
    model = tf.keras.models.load_model(model_path, custom_objects={'f1': f1, 'iou': iou})

    logging.info("Model is loaded")

    # Calculate steps for test evaluation
    test_steps = 2197 / 32

    # Evaluate the model on the test data
    test_loss, test_acc, test_f1, test_iou = model.evaluate(test, steps=test_steps)

    logging.info("Model is evaluated with 4 metrics - loss, accuracy, f1 and iou")

    # Print test accuracy
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    # Log metrics to MLflow
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_f1", test_f1)
    mlflow.log_metric("test_iou", test_iou)

    # Check if the output directory exists, and create if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write the accuracy to a file in the output directory
    with open(output_dir + "/accuracy.txt", "w") as f:
        f.write(str(test_acc))

    logging.info("Accuracy is saved.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate a TensorFlow model.")
    parser.add_argument("--use_uri", action="store_true", help="Use the dataset from the local environment")
    parser.add_argument("--test_images_path", type=str, help="Path to the test images.")
    parser.add_argument("--test_masks_path", type=str, help="Path to the test masks.")
    parser.add_argument("--model_path", type=str, help="Path to the saved model.")
    parser.add_argument("--accuracy_path", type=str, default="outputs", help="Path to the accuracy txt")

    args = parser.parse_args()

    evaluate(args.model_path, args.use_uri, args.test_images_path, args.test_masks_path, args.accuracy_path)
