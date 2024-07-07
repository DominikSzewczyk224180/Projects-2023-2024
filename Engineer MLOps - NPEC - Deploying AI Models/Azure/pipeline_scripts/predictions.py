import json
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np
from h5py import File as HDF5File
from tensorflow.keras.models import Model, load_model, model_from_json
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

try:
    from image_data_generator import test_generator, custom_generator
    from metric import F1IoUMetric
except ImportError:
    from .image_data_generator import test_generator, custom_generator
    from .metric import F1IoUMetric

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def colorize_masks(predicted_classes: np.ndarray) -> np.ndarray:
    """
    Written by Dániel
    
    Colorize the predicted classes into an RGB image where each mask has a different color.

    Parameters:
    - predicted_classes: A 2D array of predicted classes with values in the range [0, 3].

    Returns:
    - colorized_image: A 3D RGB image with each mask in a different color.
    """
    # Define colors for each mask (R, G, B)
    colors = [
        [0, 0, 255],    # Blue for mask 0
        [0, 255, 0],    # Green for mask 1
        [255, 0, 0],    # Red for mask 2
        [255, 255, 255]   # White for mask 3
    ]

    # Initialize an RGB image
    colorized_image = np.zeros((predicted_classes.shape[0], predicted_classes.shape[1], 3), dtype=np.uint8)

    # Assign colors to each mask
    for class_idx in range(4):
        colorized_image[predicted_classes == class_idx] = colors[class_idx]
    return colorized_image

def get_num_classes(config_dict: dict) -> int:
    """
    Inspect the .h5 file to determine the number of classes based on the output layer shape.

    Args:
        model_path (str): Path to the Keras model file (.h5).

    Returns:
        int: Number of classes in the model's output layer.

    Author: Benjamin Graziadei
    """
    model_architecture = model_from_json(json.dumps(config_dict))

    # Inspect the output layer
    output_layer = model_architecture.layers[-1]
    num_classes = output_layer.output_shape[-1]

    return num_classes


def custom_argmax(probabilities: np.ndarray, bias: float) -> np.ndarray:
    """
    Custom argmax function that reduces the probability of the background class by a bias value before performing argmax.

    Parameters:
    probabilities (np.ndarray): The probability distribution over classes for each pixel, last dimension should correspond to classes.
    bias (float): The value to subtract from the background class probabilities.

    Returns:
    np.ndarray: The class predictions after bias adjustment.

    Author: Benjamin Graziadei
    """
    # Reduce the background probability
    probabilities[..., -1] -= bias
    # Ensure no probabilities fall below zero or above one
    probabilities = np.clip(probabilities, 0, 1)
    return np.argmax(probabilities, axis=-1)


def unpatchify(data: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Reconstruct the image from its patches.
    Parameters:
    data (np.ndarray): Array of image patches.
    patch_size (int): Size of each patch.
    Returns:
    np.ndarray: Reconstructed image.
    
    """
    num_patches, patch_height, patch_width, channels = data.shape
    step_size = int(patch_size / 8) * 7

    # Estimate patches per dimension
    patches_per_dim = int(np.sqrt(num_patches))
    # Dynamically calculate image size
    img_height = (patches_per_dim - 1) * step_size + patch_size
    img_width = (patches_per_dim - 1) * step_size + patch_size

    # Initialize an empty array for the unpatchified image and a count array
    reconstructed_img = np.zeros((img_height, img_width, channels), dtype=np.float32)
    count = np.zeros((img_height, img_width, channels), dtype=np.float32)

    for x in range(patches_per_dim):
        for y in range(patches_per_dim):
            x_start = x * step_size
            y_start = y * step_size
            x_end = x_start + patch_size
            y_end = y_start + patch_size
            patch_index = y * patches_per_dim + x
            # Accumulate the sum of probabilities and count contributions
            reconstructed_img[y_start:y_end, x_start:x_end, :] += data[patch_index]
            count[y_start:y_end, x_start:x_end, :] += 1
    # Avoid division by zero
    count[count == 0] = 1
    # Average the overlapping regions by dividing accumulated sums by counts
    reconstructed_img /= count

    return reconstructed_img

def custom_colormap():
    # Colors as (R, G, B, A), where A is alpha for transparency
    colors = [(1, 0, 0, 0),   # fully transparent at the lowest
              (1, 0, 0, 1),   # red at low
              (1, 1, 0, 1),   # yellow at middle
              (0, 1, 0, 1)]   # green at high
    # Position of each color in the gradient [0, 1]
    nodes = [0.0, 0.33, 0.5, 1.0]
    cmap = LinearSegmentedColormap.from_list("custom_colormap", list(zip(nodes, colors)), N=256)
    return cmap

def create_colored_heatmaps(data, filename, cmap):
    # Calculate the dimensions of the data
    height, width = data.shape

    # Create a figure with the same resolution as the data
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    
    # Normalize the data to ensure transparency effects
    norm = plt.Normalize(vmin=np.min(data[data>0]), vmax=np.max(data))  # Ignore zero for vmin to maintain transparency

    # Display the image without axes and colorbar
    ax.imshow(data, interpolation='nearest', cmap=cmap, norm=norm)
    ax.axis('off')  # Hide the axes

    # Save the image directly with the given size
    plt.savefig(filename, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)

def predict_masks(model: Model, binary: bool, generator, model_name, data_set, patch_size: int = 256) -> None:
    """
    Generate and display predictions for a batch of images, handling each image in patches.

    Parameters:
    model: The machine learning model to use for prediction.
    binary (bool): True if the task is binary segmentation, else False.
    patch_size (int): The size of the patches to use.

    Returns:
    None

    Author: Benjamin Graziadei
    """

    path = os.path.join(root_dir, os.path.normpath(f"data/predictions/{data_set}_{model_name}"))
    os.makedirs(path, exist_ok=True)
    try:
        while True:
            predictions = []
            images = []
            for _ in range(13):
                image, name, *rest = next(generator)
                preds = model.predict(image)
                predictions.append(preds)
                images.append(image)


            prediction_array = np.concatenate(predictions, axis=0)
            image_array = np.concatenate(images, axis=0)

            predicted_mask = unpatchify(prediction_array, patch_size)
            full_image = unpatchify(image_array, patch_size)

            if not binary:
                predicted_classes = custom_argmax(predicted_mask, bias=0.5)
            else:
                predicted_classes = (predicted_mask > 0.5).astype(np.int32)

            for i in range(predicted_mask.shape[-1] - 1):
                custom_cmap = custom_colormap()
                create_colored_heatmaps(predicted_mask[...,i], os.path.join(path, f"PROB_mask{i+1}_" + name + ".png"), custom_cmap)
                # Save the image - must use PNG to support alpha channel
            cv2.imwrite(os.path.join(path, "ORG_" + name + ".png"), full_image * 255)

            root_mask = (predicted_classes == 0).astype(np.uint8) * 255
            shoot_mask = (predicted_classes == 1).astype(np.uint8) * 255
            seed_mask = (predicted_classes == 2).astype(np.uint8) * 255

            cv2.imwrite(os.path.join(path, f"INDIV_mask1_{name}.png"), root_mask)
            cv2.imwrite(os.path.join(path, f"INDIV_mask2_{name}.png"), shoot_mask)
            cv2.imwrite(os.path.join(path, f"INDIV_mask3_{name}.png"), seed_mask)
            
            colorized_image = colorize_masks(predicted_classes)

            cv2.imwrite(os.path.join(path, f"COMB_{name}.png"), colorized_image)
            print(f"Saved predicted mask {name}.")
    except StopIteration as e:
        return path


def choose_model(model_name) -> Tuple:
    """
    Interactively choose and load a model from available .h5 files based on user input.

    Returns:
    Tuple: A tuple containing the loaded model and its type (binary or multiclass).

    Author: Benjamin Graziadei
    """
    model_path = root_dir + "/models/" + model_name
    

    with HDF5File(model_path, "r") as f:
        model_config = f.attrs.get("model_config")
        if model_config is None:
            raise ValueError("model_config attribute not found in the .h5 file.")

        # Decode if necessary
        if isinstance(model_config, bytes):
            model_config = model_config.decode("utf-8")

        # Parse JSON to get model configuration
        config_dict = json.loads(model_config)
    
    num_classes = get_num_classes(config_dict)

    model = load_model(
        model_path,
        custom_objects={
            "F1IoUMetric": lambda **kwargs: F1IoUMetric(
                num_classes=num_classes, **kwargs
            )
        },
    )
    return model, num_classes


def main(model_name, path) -> None:
    print(f"Model name: {model_name}")
    print(f"Path: {path}")

    model, num_classes = choose_model(model_name + ".h5")
    
    if any('mask' in d for d in os.listdir(os.path.join(root_dir, 'data/processed', path))):
        print('Predicting masks on test set')
        generator = test_generator(13, path)
        return predict_masks(model, num_classes == 1, generator, model_name, path.split('\\')[-1])
    elif path:
        print('Predicting masks on images')
        generator = custom_generator(13, path)
        return predict_masks(model, num_classes == 1, generator, model_name, path.split('\\')[-1])