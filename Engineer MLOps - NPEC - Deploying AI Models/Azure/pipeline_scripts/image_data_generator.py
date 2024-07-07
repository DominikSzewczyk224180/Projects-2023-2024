import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)


patch_dir_base = root_dir + "/data/processed"


def get_available_mask_subfolders(mask_folder: str) -> list:
    """
    Checks and returns the available mask subfolders within the specified mask folder.

    Parameters:
    - mask_folder (str): The directory path of the mask folder.

    Returns:
    - list: A list of available mask subfolders.

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
    data_gen_args,
    batch_size,
    train_path,
    image_folder="train_images",
    mask_folder="train_masks",
    target_size=(256, 256),
    seed=6,
):
    """
    Custom data generator to load images and their corresponding multiple masks.

    Parameters:
    - data_gen_args: Dictionary of data augmentation arguments for ImageDataGenerator.
    - batch_size: Batch size for the generator.
    - train_path: Base directory path containing image and mask folders.
    - image_folder: Name of the folder containing images.
    - mask_folder: Name of the folder containing masks.
    - target_size: Target size for resizing images and masks.
    - seed: Random seed for shuffling and transformations.

    Returns:
    - generator: A generator yielding batches of images and their corresponding masks.

    Author: Benjamin Graziadei
    """
    train_path = os.path.join(patch_dir_base, train_path)
    print("Train path: ", train_path)

    mask_subfolders = get_available_mask_subfolders(
        os.path.join(train_path, mask_folder)
    )
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
        masks = np.concatenate(
            masks, axis=-1
        )  # Concatenate masks along the last dimension

        # Check if there are exactly two such indices and combine the corresponding elements
        if len(root_indices) == 2:
            combined_mask = masks[..., root_indices[0]] + masks[..., root_indices[1]]
            combined_mask = np.clip(combined_mask, 0, 1)

        """if np.sum(masks[0,..., root_indices[0]]) > 0:
            plt.subplot(1, 3, 1)
            plt.imshow(masks[0,..., root_indices[0]], cmap='gray')
            plt.title('Root 1')
            plt.subplot(1, 3, 2)
            plt.imshow(masks[0,..., root_indices[1]], cmap='gray')
            plt.title('Root 2')
            plt.subplot(1, 3, 3)
            plt.imshow(combined_mask[0], cmap='gray')
            plt.title('Combined')
            plt.show()"""

        total_mask_coverage = np.sum(masks, axis=-1)
        # Clip the result to ensure it remains a binary mask (this should not be necessary because our labels dont overlap but its more rubust)
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
                yield_mask = np.concatenate(
                    [masks, background_mask[..., None]], axis=-1
                )

            """for j in range(yield_mask.shape[-1]):
                plt.subplot(1, yield_mask.shape[-1], j + 1)
                plt.imshow(yield_mask[0, :, :, j], cmap='gray')
                plt.title(f'Mask {j}')
            plt.show()"""
            yield img, yield_mask


def test_generator(
    batch_size,
    train_path,
    image_folder="test_images",
    mask_folder="test_masks",
    target_size=(256, 256),
    seed=6,
):
    """
    Custom data generator to load images and their corresponding multiple masks.

    Parameters:
    - batch_size: Batch size for the generator.
    - train_path: Base directory path containing image and mask folders.
    - image_folder: Name of the folder containing images.
    - mask_folder: Name of the folder containing masks.
    - target_size: Target size for resizing images and masks.
    - seed: Random seed for shuffling and transformations.

    Returns:
    - generator: A generator yielding batches of images and their corresponding masks.
    
    Author: Benjamin Graziadei
    """
    train_path = os.path.join(patch_dir_base, train_path)
    mask_subfolders = get_available_mask_subfolders(
        os.path.join(train_path, mask_folder)
    )
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
            index
            for index, folder in enumerate(mask_subfolders)
            if "root" in folder
        ]

        img = image_generator.next()
        img_file_names = image_generator.filenames[
            _ * batch_size : (_ + 1) * batch_size
        ]
        img_file_names = os.path.basename(img_file_names[0]).rsplit("_", 2)[0]

        masks = [mask_generator.next() for mask_generator in mask_generators]
        masks = np.concatenate(
            masks, axis=-1
        )  # Concatenate masks along the last dimension

        # Check if there are exactly two such indices and combine the corresponding elements
        if len(root_indices) == 2:
            combined_mask = (
                masks[..., root_indices[0]] + masks[..., root_indices[1]]
            )
            combined_mask = np.clip(combined_mask, 0, 1)

        """if np.sum(masks[0,..., root_indices[0]]) > 0:
            plt.subplot(1, 3, 1)
            plt.imshow(masks[0,..., root_indices[0]], cmap='gray')
            plt.title('Root 1')
            plt.subplot(1, 3, 2)
            plt.imshow(masks[0,..., root_indices[1]], cmap='gray')
            plt.title('Root 2')
            plt.subplot(1, 3, 3)
            plt.imshow(combined_mask[0], cmap='gray')
            plt.title('Combined')
            plt.show()"""

        total_mask_coverage = np.sum(masks, axis=-1)
        # Clip the result to ensure it remains a binary mask (this should not be necessary because our labels dont overlap but its more rubust)
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
                yield_mask = np.concatenate(
                    [masks, background_mask[..., None]], axis=-1
                )

            """for j in range(yield_mask.shape[-1]):
                plt.subplot(1, yield_mask.shape[-1], j + 1)
                plt.imshow(yield_mask[0, :, :, j], cmap='gray')
                plt.title(f'Mask {j}')
            plt.show()"""
            yield img, img_file_names, yield_mask


def val_generator(
    batch_size,
    train_path,
    image_folder="val_images",
    mask_folder="val_masks",
    target_size=(256, 256),
    seed=6,
):
    """
    Custom data generator to load images and their corresponding multiple masks.

    Parameters:
    - batch_size: Batch size for the generator.
    - train_path: Base directory path containing image and mask folders.
    - image_folder: Name of the folder containing images.
    - mask_folder: Name of the folder containing masks.
    - target_size: Target size for resizing images and masks.
    - seed: Random seed for shuffling and transformations.

    Returns:
    - generator: A generator yielding batches of images and their corresponding masks.

    Author: Benjamin Graziadei
    """
    train_path = os.path.join(patch_dir_base, train_path)
    mask_subfolders = get_available_mask_subfolders(
        os.path.join(train_path, mask_folder)
    )
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
        masks = np.concatenate(
            masks, axis=-1
        )  # Concatenate masks along the last dimension

        # Check if there are exactly two such indices and combine the corresponding elements
        if len(root_indices) == 2:
            combined_mask = masks[..., root_indices[0]] + masks[..., root_indices[1]]
            combined_mask = np.clip(combined_mask, 0, 1)

        """if np.sum(masks[0,..., root_indices[0]]) > 0:
            plt.subplot(1, 3, 1)
            plt.imshow(masks[0,..., root_indices[0]], cmap='gray')
            plt.title('Root 1')
            plt.subplot(1, 3, 2)
            plt.imshow(masks[0,..., root_indices[1]], cmap='gray')
            plt.title('Root 2')
            plt.subplot(1, 3, 3)
            plt.imshow(combined_mask[0], cmap='gray')
            plt.title('Combined')
            plt.show()"""

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
                # Clip the result to ensure it remains a binary mask (this should not be necessary because our labels dont overlap but its more rubust)
                total_mask_coverage = np.clip(total_mask_coverage, 0, 1)
                # Create the background mask by subtracting the total mask coverage from 1
                background_mask = 1 - total_mask_coverage
                yield_mask = np.concatenate(
                    [masks, background_mask[..., None]], axis=-1
                )

            """for j in range(yield_mask.shape[-1]):
                plt.subplot(1, yield_mask.shape[-1], j + 1)
                plt.imshow(yield_mask[0, :, :, j], cmap='gray')
                plt.title(f'Mask {j}')
            plt.show()"""
            yield img, yield_mask

def custom_generator(
    batch_size,
    train_path,
    image_folder="images",
    target_size=(256, 256),
):
    """
    Custom data generator to load images and their corresponding multiple masks.

    Parameters:
    - data_gen_args: Dictionary of data augmentation arguments for ImageDataGenerator.
    - batch_size: Batch size for the generator.
    - train_path: Base directory path containing image and mask folders.
    - image_folder: Name of the folder containing images.
    - mask_folder: Name of the folder containing masks.
    - target_size: Target size for resizing images and masks.
    - seed: Random seed for shuffling and transformations.

    Returns:
    - generator: A generator yielding batches of images and their corresponding masks.

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
        img_file_names = image_generator.filenames[
            _ * batch_size : (_ + 1) * batch_size
        ]
        img_file_names = os.path.basename(img_file_names[0]).rsplit("_", 2)[0]
        yield img, img_file_names