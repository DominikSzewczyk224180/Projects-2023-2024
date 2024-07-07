import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from typing import List, Tuple, Generator

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

patch_dir_base = root_dir + "/data/processed"

def get_available_mask_subfolders(mask_folder: str) -> List[str]:
    """
    Checks and returns the available mask subfolders within the specified mask folder.

    Parameters:
    - mask_folder (str): The directory path of the mask folder.

    Returns:
    - List[str]: A list of available mask subfolders.

    Author: Benjamin Graziadei
    """
    try:
        subfolders = [f.name for f in os.scandir(mask_folder) if f.is_dir()]
        print(f"Detected mask subfolders: {subfolders}")
        return subfolders
    except FileNotFoundError:
        print("Mask folder not found. Please check the path.")
        return []

def train_generator(
    data_gen_args: dict,
    batch_size: int,
    train_path: str,
    image_folder: str = "train_images",
    mask_folder: str = "train_masks",
    target_size: Tuple[int, int] = (256, 256),
    seed: int = 6,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Custom data generator to load images and their corresponding multiple masks.

    Parameters:
    - data_gen_args (dict): Dictionary of data augmentation arguments for ImageDataGenerator.
    - batch_size (int): Batch size for the generator.
    - train_path (str): Base directory path containing image and mask folders.
    - image_folder (str): Name of the folder containing images. Default is "train_images".
    - mask_folder (str): Name of the folder containing masks. Default is "train_masks".
    - target_size (Tuple[int, int]): Target size for resizing images and masks. Default is (256, 256).
    - seed (int): Random seed for shuffling and transformations. Default is 6.

    Returns:
    - Generator[Tuple[np.ndarray, np.ndarray], None, None]: A generator yielding batches of images and their corresponding masks.

    Author: Benjamin Graziadei
    """
    train_path = os.path.join(patch_dir_base, train_path)
    print("Train path: ", train_path)

    mask_subfolders = get_available_mask_subfolders(os.path.join(train_path, mask_folder))
    image_datagen = ImageDataGenerator(rescale=1.0 / 255, **data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode="grayscale",
        target_size=target_size,
        batch_size=batch_size,
        seed=seed,
    )

    mask_generators = [
        mask_datagen.flow_from_directory(
            os.path.join(train_path, mask_folder),
            classes=[subfolder],
            class_mode=None,
            color_mode="grayscale",
            target_size=target_size,
            batch_size=batch_size,
            seed=seed,
        )
        for subfolder in mask_subfolders
    ]

    while True:
        root_indices = [
            index for index, folder in enumerate(mask_subfolders) if "root" in folder
        ]

        img = image_generator.next()
        masks = [mask_generator.next() for mask_generator in mask_generators]
        masks = np.concatenate(masks, axis=-1)  # Concatenate masks along the last dimension

        # Check if there are exactly two such indices and combine the corresponding elements
        if len(root_indices) == 2:
            combined_mask = masks[..., root_indices[0]] + masks[..., root_indices[1]]
            combined_mask = np.clip(combined_mask, 0, 1)

        total_mask_coverage = np.sum(masks, axis=-1)
        # Clip the result to ensure it remains a binary mask (this should not be necessary because our labels don't overlap but it's more robust)
        total_mask_coverage = np.clip(total_mask_coverage, 0, 1)
        # Create the background mask by subtracting the total mask coverage from 1
        background_mask = 1 - total_mask_coverage

        try:
            # Replace the first mask in root_indices with the combined mask
            masks[..., root_indices[0]] = combined_mask
            for i in range(1, len(root_indices)):
                masks = np.delete(masks, root_indices[i], axis=-1)
        except Exception as e:
            continue
        finally:
            if masks.shape[-1] == 1:
                yield_mask = masks
            else:
                yield_mask = np.concatenate([masks, background_mask[..., None]], axis=-1)

            yield img, yield_mask

def test_generator(
    batch_size: int,
    train_path: str,
    image_folder: str = "test_images",
    mask_folder: str = "test_masks",
    target_size: Tuple[int, int] = (256, 256),
    seed: int = 6,
) -> Generator[Tuple[np.ndarray, str, np.ndarray], None, None]:
    """
    Custom data generator to load images and their corresponding multiple masks.

    Parameters:
    - batch_size (int): Batch size for the generator.
    - train_path (str): Base directory path containing image and mask folders.
    - image_folder (str): Name of the folder containing images. Default is "test_images".
    - mask_folder (str): Name of the folder containing masks. Default is "test_masks".
    - target_size (Tuple[int, int]): Target size for resizing images and masks. Default is (256, 256).
    - seed (int): Random seed for shuffling and transformations. Default is 6.

    Returns:
    - Generator[Tuple[np.ndarray, str, np.ndarray], None, None]: A generator yielding batches of images, their filenames, and corresponding masks.

    Author: Benjamin Graziadei
    """
    train_path = os.path.join(patch_dir_base, train_path)
    mask_subfolders = get_available_mask_subfolders(os.path.join(train_path, mask_folder))
    image_datagen = ImageDataGenerator(rescale=1.0 / 255)
    mask_datagen = ImageDataGenerator()

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode="grayscale",
        target_size=target_size,
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
    )

    mask_generators = [
        mask_datagen.flow_from_directory(
            os.path.join(train_path, mask_folder),
            classes=[subfolder],
            class_mode=None,
            color_mode="grayscale",
            target_size=target_size,
            batch_size=batch_size,
            seed=seed,
            shuffle=False,
        )
        for subfolder in mask_subfolders
    ]

    num_samples = image_generator.samples
    steps_per_epoch = np.ceil(num_samples / batch_size)

    for _ in range(int(steps_per_epoch)):
        root_indices = [
            index for index, folder in enumerate(mask_subfolders) if "root" in folder
        ]

        img = image_generator.next()
        img_file_names = image_generator.filenames[_ * batch_size : (_ + 1) * batch_size]
        img_file_names = os.path.basename(img_file_names[0]).rsplit("_", 2)[0]

        masks = [mask_generator.next() for mask_generator in mask_generators]
        masks = np.concatenate(masks, axis=-1)  # Concatenate masks along the last dimension

        # Check if there are exactly two such indices and combine the corresponding elements
        if len(root_indices) == 2:
            combined_mask = masks[..., root_indices[0]] + masks[..., root_indices[1]]
            combined_mask = np.clip(combined_mask, 0, 1)

        total_mask_coverage = np.sum(masks, axis=-1)
        # Clip the result to ensure it remains a binary mask (this should not be necessary because our labels don't overlap but it's more robust)
        total_mask_coverage = np.clip(total_mask_coverage, 0, 1)
        # Create the background mask by subtracting the total mask coverage from 1
        background_mask = 1 - total_mask_coverage

        try:
            # Replace the first mask in root_indices with the combined mask
            masks[..., root_indices[0]] = combined_mask
            for i in range(1, len(root_indices)):
                masks = np.delete(masks, root_indices[i], axis=-1)
        except Exception as e:
            continue
        finally:
            if masks.shape[-1] == 1:
                yield_mask = masks
            else:
                yield_mask = np.concatenate([masks, background_mask[..., None]], axis=-1)

            yield img, img_file_names, yield_mask

def val_generator(
    batch_size: int,
    train_path: str,
    image_folder: str = "val_images",
    mask_folder: str = "val_masks",
    target_size: Tuple[int, int] = (256, 256),
    seed: int = 6,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Custom data generator to load images and their corresponding multiple masks.

    Parameters:
    - batch_size (int): Batch size for the generator.
    - train_path (str): Base directory path containing image and mask folders.
    - image_folder (str): Name of the folder containing images. Default is "val_images".
    - mask_folder (str): Name of the folder containing masks. Default is "val_masks".
    - target_size (Tuple[int, int]): Target size for resizing images and masks. Default is (256, 256).
    - seed (int): Random seed for shuffling and transformations. Default is 6.

    Returns:
    - Generator[Tuple[np.ndarray, np.ndarray], None, None]: A generator yielding batches of images and their corresponding masks.

    Author: Benjamin Graziadei
    """
    train_path = os.path.join(patch_dir_base, train_path)
    mask_subfolders = get_available_mask_subfolders(os.path.join(train_path, mask_folder))
    image_datagen = ImageDataGenerator(rescale=1.0 / 255)
    mask_datagen = ImageDataGenerator()

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode="grayscale",
        target_size=target_size,
        batch_size=batch_size,
        seed=seed,
    )

    mask_generators = [
        mask_datagen.flow_from_directory(
            os.path.join(train_path, mask_folder),
            classes=[subfolder],
            class_mode=None,
            color_mode="grayscale",
            target_size=target_size,
            batch_size=batch_size,
            seed=seed,
        )
        for subfolder in mask_subfolders
    ]

    while True:
        root_indices = [
            index for index, folder in enumerate(mask_subfolders) if "root" in folder
        ]

        img = image_generator.next()
        masks = [mask_generator.next() for mask_generator in mask_generators]
        masks = np.concatenate(masks, axis=-1)  # Concatenate masks along the last dimension

        # Check if there are exactly two such indices and combine the corresponding elements
        if len(root_indices) == 2:
            combined_mask = masks[..., root_indices[0]] + masks[..., root_indices[1]]
            combined_mask = np.clip(combined_mask, 0, 1)

        try:
            # Replace the first mask in root_indices with the combined mask
            masks[..., root_indices[0]] = combined_mask
            for i in range(1, len(root_indices)):
                masks = np.delete(masks, root_indices[i], axis=-1)
        except Exception as e:
            continue
        finally:
            if masks.shape[-1] == 1:
                yield_mask = masks
            else:
                total_mask_coverage = np.sum(masks, axis=-1)
                # Clip the result to ensure it remains a binary mask (this should not be necessary because our labels don't overlap but it's more robust)
                total_mask_coverage = np.clip(total_mask_coverage, 0, 1)
                # Create the background mask by subtracting the total mask coverage from 1
                background_mask = 1 - total_mask_coverage
                yield_mask = np.concatenate([masks, background_mask[..., None]], axis=-1)

            yield img, yield_mask

def custom_generator(
    batch_size: int,
    train_path: str,
    image_folder: str = "images",
    target_size: Tuple[int, int] = (256, 256),
) -> Generator[Tuple[np.ndarray, str], None, None]:
    """
    Custom data generator to load images.

    Parameters:
    - batch_size (int): Batch size for the generator.
    - train_path (str): Base directory path containing image folders.
    - image_folder (str): Name of the folder containing images. Default is "images".
    - target_size (Tuple[int, int]): Target size for resizing images. Default is (256, 256).

    Returns:
    - Generator[Tuple[np.ndarray, str], None, None]: A generator yielding batches of images and their filenames.

    Author: Benjamin Graziadei
    """
    image_path = os.path.join(patch_dir_base, train_path)
    image_datagen = ImageDataGenerator(rescale=1.0 / 255)

    image_generator = image_datagen.flow_from_directory(
        image_path,
        classes=[image_folder],
        class_mode=None,
        color_mode="grayscale",
        target_size=target_size,
        batch_size=batch_size,
        shuffle=False
    )
    num_samples = image_generator.samples
    steps_per_epoch = np.ceil(num_samples / batch_size)

    for _ in range(int(steps_per_epoch)):
        img = image_generator.next()
        img_file_names = image_generator.filenames[_ * batch_size : (_ + 1) * batch_size]
        img_file_names = os.path.basename(img_file_names[0]).rsplit("_", 2)[0]
        yield img, img_file_names
