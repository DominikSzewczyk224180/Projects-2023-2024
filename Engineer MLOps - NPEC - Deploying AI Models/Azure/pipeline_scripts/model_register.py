import argparse
import os

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.identity import ClientSecretCredential


def register_model_if_accuracy_above_threshold(
    model_path: str, accuracy_folder: str, threshold: float = 0.5
) -> None:
    """
    From the self-study material

    Register the model if its accuracy is above a specified threshold.

    Parameters
    ----------
    model_path : str
        The path to the model file.
    accuracy_folder : str
        The folder containing the accuracy file.
    threshold : float, optional
        The accuracy threshold for model registration (default is 0.5).

    Returns
    -------
    None

    This function performs the following steps:

    1. Reads the accuracy from the specified accuracy file.
    2. Compares the accuracy against the specified threshold.
    3. If the accuracy is above the threshold, registers the model to Azure ML.
    """

    print(f"Registering model if accuracy is above {threshold}.")
    print(f"Model path: {model_path}")
    print(f"Accuracy file: {accuracy_folder}")
    # Get the accuracy file
    accuracy_file = os.path.join(accuracy_folder, "accuracy.txt")
    # Load accuracy from file
    with open(accuracy_file, "r") as f:
        print(f"Reading accuracy from {accuracy_file}")
        accuracy = float(f.read().strip())

    print(f"Model accuracy: {accuracy}")

    # Only register model if accuracy is above threshold
    if accuracy > threshold:
        print("Model accuracy is above threshold, registering model.")

        # Define your Azure ML settings
        subscription_id = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
        resource_group = "buas-y2"
        workspace_name = "CV1"
        tenant_id = "0a33589b-0036-4fe8-a829-3ed0926af886"
        client_id = "a2230f31-0fda-428d-8c5c-ec79e91a49f5"
        client_secret = "Y-q8Q~H63btsUkR7dnmHrUGw2W0gMWjs0MxLKa1C"

        credential = ClientSecretCredential(tenant_id, client_id, client_secret)

        ml_client = MLClient(
            credential, subscription_id, resource_group, workspace_name
        )

        model = Model(
            path=model_path,
            type=AssetTypes.CUSTOM_MODEL,
            name="Model",
            description="Model created from pipeline",
        )

        # Register the model
        model = ml_client.models.create_or_update(model)
        print("Model registered.")
    else:
        print("Model accuracy is not above threshold, not registering model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Register a model if accuracy is above threshold."
    )
    parser.add_argument("--model", type=str, help="Saved model path.")
    parser.add_argument("--accuracy", type=str, help="Model's accuracy path.")
    args = parser.parse_args()

    register_model_if_accuracy_above_threshold(args.model, args.accuracy)
