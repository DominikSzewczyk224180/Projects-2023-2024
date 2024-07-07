import os
from typing import Any, Generator, Tuple

import numpy as np
from image_data_generator import (test_generator, train_generator,
                                       val_generator)
from PIL import Image


def load_data_uri(
    image_uri: str,
    mask_uri: str,
    target_size: Tuple[int, int] = (256, 256),
    batch_size: int = 32,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Writen by Matey

    Custom data generator that yields batches of images and masks.

    Parameters
    ----------
    image_uri : str
        URI of the directory containing the images.
    mask_uri : str
        URI of the directory containing the masks.
    target_size : tuple
        Desired image size after resizing. Default is (256, 256).
    batch_size : int
        Number of images to yield per batch. Default is 32.

    Yields
    ------
    tuple
        Batch of preprocessed images and masks.
    """

    image_filenames = [f for f in os.listdir(image_uri) if f.endswith(".png")]
    mask_filenames = [
        f
        for f in os.listdir(mask_uri)
        if f.endswith(".png") and "root" in f and "occluded_root" not in f
    ]

    # Ensure the mask filenames list is sorted
    mask_filenames.sort()

    # Ensure only those images with corresponding masks are used
    paired_filenames = [
        (img, msk)
        for img, msk in zip(image_filenames, mask_filenames)
        if img.split(".")[0] in msk
    ]

    while True:
        for start in range(0, len(paired_filenames), batch_size):
            end = min(start + batch_size, len(paired_filenames))
            batch_images = []
            batch_masks = []

            for image_filename, mask_filename in paired_filenames[start:end]:
                image_path = os.path.join(image_uri, image_filename)
                mask_path = os.path.join(mask_uri, mask_filename)

                # Load and preprocess image

                # Convert to grayscale
                image = Image.open(image_path).convert("L")
                image = image.resize(target_size)  # Resize image
                image = np.array(image) / 255.0  # Normalize pixel values
                batch_images.append(
                    np.expand_dims(image, axis=-1)
                )  # Add channel dimension

                # Load and preprocess mask
                # Convert to grayscale
                mask = Image.open(mask_path).convert("L")
                mask = mask.resize(target_size)  # Resize mask
                mask = np.array(mask) / 255.0  # Normalize pixel values
                batch_masks.append(
                    np.expand_dims(mask, axis=-1)
                )  # Add channel dimension

            # Print the path of at least one image in this batch
            if batch_images:
                print(f"Processed batch includes image: {image_path}")

            yield np.array(batch_images), np.array(batch_masks)


def load_data() -> Tuple[Any, Any, Any]:
    """
    Written by Matey

    Load the data.

    Args:
        path: The path to the folder for the data.

    Returns:
        data: train and validation data

    """

    path = "../../data/processed/NPEC"

    train = train_generator(dict(), 8, path)
    val = val_generator(8, path)
    test = test_generator(8, path)

    return train, val, test


def extract_shape(generator: Any) -> Tuple[int, int]:
    """
    Written by Matey

    Extracts the shape of a single image from a generator.

    Args:
        generator (Data generator): The generated data.

    Returns:
        Integers: The size of the image and the channel
    """

    image, mask = next(generator)

    shape = image[0].shape

    return shape[0], shape[2]
