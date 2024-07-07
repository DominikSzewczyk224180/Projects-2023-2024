# Made by Kian van Holst

# Standard library imports
import os
import sys
import time
import glob
import shutil
import logging
import argparse
from datetime import datetime
from typing import Tuple, Optional, List

# Third-party imports
import cv2
import numpy as np
from patchify import patchify
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import inquirer
from azureml.core import Workspace, Datastore, Dataset
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.data.dataset_factory import DataPath

# Custom imports
from logger import logger

def set_working_directory() -> None:
    """
    Written by Kian
    
    Set the working directory to the root directory of the current script.
    
    This function sets the working directory to be three levels up from the directory
    containing the current script and logs the new working directory.
    
    Returns:
        None
    """
    # Get the absolute path of the current script
    current_script_path = os.path.abspath(sys.argv[0])
    # Determine the root directory three levels up
    root_directory = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
    # Change the working directory
    os.chdir(root_directory)
    # Log the new working directory
    logger.info("Working directory set to: %s", os.getcwd())


def select_directory(start_path: str = 'data/raw', mode: str = 'custom') -> str:
    """
    Written by Kian
    
    Select a directory based on the given mode and data directory.
    
    Args:
        start_path (str): The directory path containing raw data directories. Defaults to 'data/raw'.
        mode (str): The mode of the script, either 'custom' or 'default'. Defaults to 'custom'.
    
    Returns:
        str: The path of the selected directory.
    """
    # Get the absolute path of the start directory
    current_path = os.path.abspath(start_path)
    # List all directories in the start directory
    entries = [entry for entry in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, entry))]
    # Sort the entries alphabetically
    entries.sort()
    
    # Automatically select the first directory in default mode
    if mode == 'default':
        if entries:
            selected_dir = entries[0]
            current_path = os.path.join(current_path, selected_dir)
            logger.info(f"Default mode: Automatically selected {current_path}")
        else:
            logger.warning(f"No directories found in {current_path}")
        return current_path
    
    # Prompt the user to select a directory in custom mode
    while True:
        if not entries:
            logger.warning(f"No directories found in {current_path}")
            return current_path

        questions = [
            inquirer.List(
                'directory',
                message="Select the directory containing the files you want to process:",
                choices=entries,
                default=entries[0]
            ),
        ]

        answers = inquirer.prompt(questions)
        selected_dir = answers['directory']
        current_path = os.path.join(current_path, selected_dir)
        logger.info(f"You have selected: {current_path}")
        return current_path

def check_directory_and_handle_contents(chosen_directory: str, mode: str, store: str) -> Tuple[bool, Optional[str]]:
    """
    Written by Kian

    Check if the destination directory and its corresponding masks directory are populated and prompt the user for an action.
    
    Args:
        chosen_directory (str): The path to the chosen directory.
        mode (str): The mode of operation, either 'default' or 'custom'.
        store (str): The storage mode, either 'local' or 'cloud'.
    
    Returns:
        Tuple[bool, Optional[str], str]: A tuple containing a boolean indicating success, an optional string representing the user action, and the path to the patched directory.
    """
    # Determine the paths based on the storage mode
    if store == 'cloud':
        patched_dir = chosen_directory.replace('raw', 'processed/temp')
        print(patched_dir)
        img_dir = os.path.join(patched_dir, 'images')
        masks_dir = os.path.join(patched_dir, 'masks')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        img_files = os.listdir(img_dir)
        mask_files = os.listdir(masks_dir)
    else:    
        patched_dir = chosen_directory.replace('raw', 'processed')
        masks_dir = patched_dir.replace('images', 'masks')
        os.makedirs(patched_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        logger.info('Destination directories created.')
        img_files = os.listdir(patched_dir)
        mask_files = os.listdir(masks_dir)

    # Check if the directories are already populated
    if img_files or mask_files:
        logger.warning("Destination directory '%s' is already populated and contains processed files.", (patched_dir))
        
        # Automatically delete contents in default mode
        if mode == 'default':
            user_input = 'd'
        else:
            questions = [
                inquirer.List(
                    'action',
                    message="Choose an option:",
                    choices=[
                        ('Add new files', 'a'),
                        ('Delete directory contents (recommended)', 'd'),
                        ('Cancel operation', 'c'),
                    ],
                ),
            ]

            answers = inquirer.prompt(questions)
            user_input = answers['action']

        # Handle the user's choice
        if user_input == 'd':
            logger.info('Deleting directory contents...')
            try:
                shutil.rmtree(patched_dir)
                logger.info("All contents in the destination directory have been deleted.")
            except Exception as e:
                logger.error("Error deleting contents: %s", e)
                return False, None, patched_dir
            return True, 'd', patched_dir
        elif user_input == 'a':
            logger.info("You chose to add non-existing files only (not available in early-access).")
            return True, 'a', patched_dir
        elif user_input == 'c':
            logger.info("Operation canceled by the user.")
            return False, 'c', patched_dir
    else:
        logger.info("Destination directory is empty, continuing with the process.")
        return True, 'a', patched_dir


def get_image_paths(img_dir: str, extensions: List[str] = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']) -> List[str]:
    """
    Written by Kian

    Get a list of image paths from a directory based on specified file extensions.
    
    Args:
        img_dir (str): The directory to search for images.
        extensions (List[str]): A list of file extensions to search for. Defaults to common image formats.
    
    Returns:
        List[str]: A list of image paths matching the specified extensions.
    """
    img_paths = []
    # Search for images with the specified extensions
    for ext in extensions:
        found_paths = glob.glob(os.path.join(img_dir, ext))
        img_paths.extend(found_paths)
        if found_paths:
            logger.info("Found %d images with extension %s", len(found_paths), ext)
    logger.info("Total images found: %d", len(img_paths))
    return img_paths


def get_masks(fpath: str, mask_extensions: Optional[List[str]] = None) -> List[str]:
    """
    Written by Kian

    Get a list of mask files corresponding to a given image file.
    
    Args:
        fpath (str): The file path of the image.
        mask_extensions (Optional[List[str]]): A list of mask extensions to look for. Defaults to None.
    
    Returns:
        List[str]: A list of mask file names matching the specified extensions or all masks if no extensions are specified.
    """
    image_file = os.path.basename(fpath)[:-4]
    img_dir = os.path.dirname(fpath)
    masks_dir = img_dir.replace('images', 'masks')
    files = os.listdir(masks_dir)

    # Return all masks if no extensions are specified
    if mask_extensions is None:
        return [file for file in files if file.startswith(image_file)]
    
    # Return masks matching the specified extensions
    matching_masks_names = [
        f"{image_file}_{mask_extension}" for mask_extension in mask_extensions
        if f"{image_file}_{mask_extension}" in files
    ]

    return matching_masks_names

def get_image_paths_and_masks(img_dir: str, mode: str) -> Tuple[List[str], List[str]]:
    """
    Written by Kian

    Get a list of image paths and corresponding mask extensions from a directory.
    
    Args:
        img_dir (str): The directory to search for images.
        mode (str): The mode of operation, either 'default' or 'custom'.
    
    Returns:
        Tuple[List[str], List[str]]: A tuple containing a list of image paths and a list of mask extensions.
    """
    img_paths = get_image_paths(img_dir)
    if not img_paths:
        logger.warning("No images found in the directory: %s", img_dir)
        return [], []
    
    # Get mask extensions from the first image
    sample_img_path = img_paths[0]
    matching_masks_names = get_masks(sample_img_path)
    sample_img_base = os.path.splitext(os.path.basename(sample_img_path))[0]
    mask_extensions = [mask.replace(sample_img_base, '') for mask in matching_masks_names]
    logger.info(f"The following masks have been found for {sample_img_base}: {mask_extensions}.")

    # Prompt the user to continue or cancel in non-default mode
    if mode != 'default':
        questions = [
            inquirer.List(
                'action',
                message="Do you wish to continue?",
                choices=[
                    ('Continue', 'continue'),
                    ('Cancel', 'cancel'),
                ],
            ),
        ]
        answers = inquirer.prompt(questions)
        response = answers['action']
        
        if response == 'cancel':
            logger.warning("Please check the naming convention of the provided data to ensure it is what the system expects /ref to documentation.")
            sys.exit()
            
    return img_paths, mask_extensions


def select_masks(mask_extensions: List[str], mode: str) -> List[str]:
    """
    Written by Kian

    Select which masks to process based on the given mode.
    
    Args:
        mask_extensions (List[str]): A list of mask extensions to choose from.
        mode (str): The mode of operation, either 'default' or 'custom'.
    
    Returns:
        List[str]: A list of selected mask extensions.
    """
    # Automatically select all masks in default mode
    if mode == 'default':
        return mask_extensions
    
    # Prompt the user to select masks in custom mode
    questions = [
        inquirer.Checkbox(
            'masks',
            message="Select the masks you want to process:",
            choices=mask_extensions,
            default=mask_extensions,
        ),
    ]
    answers = inquirer.prompt(questions)
    selected_masks = answers['masks']
    return selected_masks

def validate_image_paths(img_paths: List[str], selected_masks: List[str], img_dir: str, mode: str) -> List[str]:
    """
    Written by Kian

    Validate image paths by checking for the presence of corresponding mask files.
    
    Args:
        img_paths (List[str]): A list of image file paths to validate.
        selected_masks (List[str]): A list of selected mask extensions to check for.
        img_dir (str): The directory containing the images.
        mode (str): The mode of operation, either 'default' or 'interactive'.
    
    Returns:
        List[str]: A list of valid image paths that have all corresponding mask files.
    """
    skipped_messages = []
    masks_dir = img_dir.replace('images', 'masks')
    masks_dir_files = os.listdir(masks_dir)

    valid_img_paths = img_paths.copy()
    
    # Check each image for the presence of corresponding masks
    for img_path in img_paths:
        image_file = os.path.splitext(os.path.basename(img_path))[0]
        missing_files = [mask_extension for mask_extension in selected_masks if f"{image_file}{mask_extension}" not in masks_dir_files]
        if missing_files:
            message = f"Missing masks for '{image_file}': {missing_files}."
            skipped_messages.append(message)
            valid_img_paths.remove(img_path)

    # Log and prompt the user if there are missing masks
    if skipped_messages:
        for message in skipped_messages:
            logger.warning(message)
            time.sleep(0.2)

        if mode != 'default':
            questions = [
                inquirer.List(
                    'action',
                    message="There are missing masks. Press enter to continue or (c) to cancel:",
                    choices=[
                        ('Continue', 'continue'),
                        ('Cancel', 'cancel'),
                    ],
                ),
            ]

            answers = inquirer.prompt(questions)
            response = answers['action']

            if response == 'cancel':
                logger.warning("Please check the naming convention of the provided data to ensure it is what the system expects /ref to documentation.")
                sys.exit()
    else:
        logger.info("No missing masks detected.")
    
    return valid_img_paths


def split_data(img_paths: List[str], train_size: float = 0.8, val_size: float = 0.1, random_state: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Written by Kian

    Split the image paths into training, validation, and test sets.
    
    Args:
        img_paths (List[str]): A list of image file paths to split.
        train_size (float): Proportion of the dataset to include in the training set. Defaults to 0.8.
        val_size (float): Proportion of the dataset to include in the validation set. Defaults to 0.1.
        random_state (int): Random seed for reproducibility. Defaults to 42.
    
    Returns:
        Tuple[List[str], List[str], List[str]]: A tuple containing the training, test, and validation image paths.
    """
    logger.info("Starting data split with train_size=%s and val_size=%s", train_size, val_size)
    
    # Split the data into training and temporary sets
    train_img_paths, temp_img_paths = train_test_split(img_paths, train_size=train_size, random_state=random_state)
    # Calculate the proportion of the validation set within the remaining data
    test_size = val_size / (1 - train_size)
    test_size = round(test_size, 6)
    
    # Split the temporary set into validation and test sets
    val_img_paths, test_img_paths = train_test_split(temp_img_paths, test_size=test_size, random_state=random_state)
    
    logger.info("Data split completed. Training set: %d, Testing set: %d, Validation set: %d", len(train_img_paths), len(test_img_paths), len(val_img_paths))
    
    return train_img_paths, test_img_paths, val_img_paths


def create_directories(base_dir: str, set_types: List[str], mask_extensions: List[str]) -> None:
    """
    Written by Kian
    
    Create directories for the given set types and mask extensions.
    
    Args:
        base_dir (str): The base directory where the directories will be created.
        set_types (List[str]): A list of set types (e.g., ['train', 'test', 'val']).
        mask_extensions (List[str]): A list of mask extensions to create corresponding directories for.
    
    Returns:
        None
    """
    # Create directories for each set type and mask extension
    if set_types:
        for set_type in set_types:
            img_dir = os.path.join(base_dir, f'{set_type}_images')
            mask_dir = os.path.join(base_dir, f'{set_type}_masks')
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)
            
            for mask_extension in mask_extensions:
                os.makedirs(os.path.join(mask_dir, mask_extension[1:].replace('_mask.tif', '')), exist_ok=True)
            
            logger.info("Created directories: %s and %s", img_dir, mask_dir)
    else:
        img_dir = os.path.join(base_dir, 'images')
        mask_dir = os.path.join(base_dir, 'masks')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        
        for mask_extension in mask_extensions:
            os.makedirs(os.path.join(mask_dir, mask_extension[1:].replace('_mask.tif', '')), exist_ok=True)
        logger.info("Created directories: %s and %s", img_dir, mask_dir)

def crop_dimensions(img: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Written by Kian

    Extracts the crop dimensions for the given image.
    
    Args:
        img (np.ndarray): The input image array.
    
    Returns:
        Tuple[int, int, int, int]: The coordinates (y_min, y_max, x_min, x_max) for cropping the image.
    """
    # Crop the image to a specific width
    img = img[:, :4000]
    # Apply median blur and thresholding
    im_blurred = cv2.medianBlur(img, 5)
    _, output_im = cv2.threshold(im_blurred, 50, 250, cv2.THRESH_BINARY_INV)
    kernel = np.ones((11, 11), np.uint8)
    output_im = cv2.erode(output_im, kernel, iterations=1)
    output_im = cv2.dilate(output_im, kernel, iterations=1)
    _, _, stats, _ = cv2.connectedComponentsWithStats(output_im)
    largest_label = np.argmax(stats[:, cv2.CC_STAT_AREA])

    # Get the bounding box of the largest connected component
    x, y, w, h = stats[largest_label][:4]
    side_length = max(w, h) + 10
    x_min = max(min(x + w // 2 - side_length // 2, img.shape[1] - side_length), 0)
    y_min = max(min(y + h // 2 - side_length // 2, img.shape[0] - side_length), 0)
    x_max = x_min + side_length
    y_max = y_min + side_length
    return y_min, y_max, x_min, x_max


def padder(image: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Written by Kian
    
    Pad the image to ensure it fits the specified patch size with a step of 7/8.

    Args:
        image (np.ndarray): The input image array.
        patch_size (int): The size of the patches to be created.

    Returns:
        np.ndarray: The padded image array.
    """
    step_size = int(patch_size / 8) * 7
    h, w = image.shape[:2]
    num_patches_h = (h + step_size - patch_size) // step_size + 1
    num_patches_w = (w + step_size - patch_size) // step_size + 1
    
    # Calculate the padding required to fit the patches
    padded_h = (num_patches_h - 1) * step_size + patch_size
    padded_w = (num_patches_w - 1) * step_size + patch_size
    height_padding = padded_h - h
    width_padding = padded_w - w

    # Pad the image
    padded_image = cv2.copyMakeBorder(image, 0, height_padding, 0, width_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded_image



def save_patches(fpath: str, img_patches: np.ndarray) -> None:
    """
    Written by Kian
    
    Save image patches to the corresponding file path.

    Args:
        fpath (str): The file path to save the patches.
        img_patches (np.ndarray): The array of image patches.

    Returns:
        None
    """
    patch_size = img_patches.shape[2]
    n_patches = img_patches.shape[0]
    img_patches = (img_patches.reshape(-1, patch_size, patch_size, 1) * 255).astype(np.uint8)
    img_patch_path = fpath.replace('raw', 'processed/patched')
    
    # Save each patch with a numbered file name
    for i, patch in enumerate(img_patches):
        col = (i + n_patches) // n_patches
        row = (i + n_patches) % n_patches + 1
        image_patch_path_numbered = f'{img_patch_path[:-4]}_{col:02}_{row:02}.png'
        
        if os.path.exists(image_patch_path_numbered):
            os.remove(image_patch_path_numbered)
        
        cv2.imwrite(image_patch_path_numbered, patch)

def process_images(
    set_type: str, 
    set_paths: List[str], 
    img_patched_dir: str, 
    img_dir: str, 
    selected_masks: List[str], 
    patch_size: int, 
    mode: str
) -> None:
    """
    Written by Kian
    
    Process images by cropping, normalizing, padding, and patchifying them. Save the resulting patches.

    Args:
        set_type (str): The type of the set (e.g., 'train', 'test', 'val'). Can be empty.
        set_paths (List[str]): A list of paths to the images to be processed.
        img_patched_dir (str): The directory where the patched images will be saved.
        img_dir (str): The directory containing the original images.
        selected_masks (List[str]): A list of selected mask extensions to process.
        patch_size (int): The size of the patches.
        mode (str): The mode of operation ('a' for adding new files only).

    Returns:
        None
    """

    set_type_images_dir = f"{set_type}_images" if set_type else "images"
    set_type_masks_dir = f"{set_type}_masks" if set_type else "masks"
    logger.info(f"Processing {set_type + ' ' if set_type else ''}images.")    
 

    for img_path in tqdm(set_paths, desc=f"Patching {set_type + ' ' if set_type else ''}images...", leave=False, smoothing=0.1):
        masks_dir = img_dir.replace('images', 'masks')
        image_file = os.path.splitext(os.path.basename(img_path))[0]
        matching_masks_names = [f"{image_file}{mask}" for mask in selected_masks if f"{image_file}{mask}" in os.listdir(masks_dir)]
        
        image_files = os.listdir(os.path.join(img_patched_dir, set_type_images_dir))
        if image_files and any(image_file in filename for filename in image_files) and mode == 'a':
            continue
        
        img = cv2.imread(img_path, 0)
        y_min, y_max, x_min, x_max = crop_dimensions(img)
        img_cropped = img[y_min:y_max, x_min:x_max]
        img_normalized = img_cropped / 255.0 if img_cropped.max() > 1 else img_cropped
        img_padded = padder(img_normalized, patch_size)
        img_patches = patchify(img_padded, (patch_size, patch_size), step=int((patch_size/8)*7))
        save_patches(os.path.join(img_patched_dir, set_type_images_dir, os.path.basename(img_path)), img_patches)

        for mask_name in matching_masks_names:
            mask_path = os.path.join(masks_dir, mask_name)
            mask = cv2.imread(mask_path, 0)
            mask_cropped = mask[y_min:y_max, x_min:x_max]
            mask_normalized = mask_cropped / 255.0 if mask_cropped.max() > 1 else mask_cropped
            mask_padded = padder(mask_normalized, patch_size)
            mask_patches = patchify(mask_padded, (patch_size, patch_size), step=int((patch_size/8)*7))
            for mask_extension in selected_masks:
                if mask_name == f'{image_file}{mask_extension}':
                    mask_extension_clean = mask_extension[1:].replace('_mask.tif', '')
                    save_patches(os.path.join(img_patched_dir, set_type_masks_dir, mask_extension_clean, mask_name), mask_patches)

    logger.info(f"Completed processing {set_type + ' ' if set_type else ''}images.")


def upload_directory(local_path: str, target_path: str, datastore: str) -> Dataset:
    """
    Written by Kian
    
    Upload a local directory to a datastore.

    Args:
        local_path (str): The local directory path to upload.
        target_path (str): The target directory path in the datastore.

    Returns:
        Dataset: The uploaded dataset.
    """
    dataset = Dataset.File.upload_directory(
        src_dir=local_path,
        target=DataPath(datastore, target_path),
        overwrite=True,
        show_progress=True
    )
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Select a directory containing files to process.")
    parser.add_argument('--mode', type=str, default='default', choices=['custom', 'default'],
                        help='Mode of operation: "interactive" for user prompt, "default" for automatic selection.')
    parser.add_argument('--store', type=str, default='local', choices=['local', 'cloud'],
                        help='Specify the storage mode for the application. Choose "local" to use local storage, or "cloud" to use cloud storage.')
    parser.add_argument('--dir', type=str, default=None,
                        help='Specify the directory to process. Overrides default directory selection.')
    parser.add_argument('--masks', default=None, type=str, nargs='+', help='List of mask extensions to process (e.g. "_mask1.tif _mask2.tif").')
    args = parser.parse_args()

    if args.dir:
        chosen_directory = args.dir  # Override with the user-specified directory if provided
        chosen_directory = os.path.join('data/raw', chosen_directory)

    else:
        chosen_directory = select_directory('data/raw', mode=args.mode)  # Use the default or selected directory based on mode

    img_dir = os.path.join(chosen_directory, 'images/')
    patch_size = 256
    proceed, mode, img_patched_dir = check_directory_and_handle_contents(chosen_directory, mode=args.mode, store=args.store)
    dataset = os.path.basename(img_patched_dir)

    if proceed:
        img_paths, mask_extensions = get_image_paths_and_masks(img_dir,mode=args.mode)
        if args.masks:
            selected_masks = args.masks
            mask_extensions = args.masks
        else:
            selected_masks = select_masks(mask_extensions, mode=args.mode)
            print(selected_masks)
        img_paths = validate_image_paths(img_paths, selected_masks, img_dir, mode=args.mode)
        if args.store == 'local':
            train_img_paths, test_img_paths, val_img_paths = split_data(img_paths)
            create_directories(img_patched_dir, ['train', 'test', 'val'], mask_extensions)

            for set_type, set_paths in zip(['train', 'test', 'val'], [train_img_paths, test_img_paths, val_img_paths]):
                process_images(set_type, set_paths, img_patched_dir, img_dir, selected_masks, patch_size, mode)
                
            logger.info("Processing finished.")

        if args.store == 'cloud':
            set_type = None
            create_directories(img_patched_dir, set_type, mask_extensions)

            process_images(set_type, img_paths, img_patched_dir, img_dir, selected_masks, patch_size, mode)
            # Retrieve workspace
            subscription_id = '0a94de80-6d3b-49f2-b3e9-ec5818862801'
            resource_group = 'buas-y2'
            workspace_name = 'CV1'

            # Log in using interactive Auth
            auth = InteractiveLoginAuthentication()

            # Declare workspace & datastore.
            workspace = Workspace(subscription_id=subscription_id,
                                resource_group=resource_group,
                                workspace_name=workspace_name,
                                auth=auth)

            # Get the default datastore
            datastore = Datastore.get(workspace, datastore_name='workspaceblobstore')
            dataset = upload_directory(img_patched_dir, f'processed_data/{dataset}')
            
            #for set_type, set_paths in zip(['train', 'test', 'val'], [train_img_paths, test_img_paths, val_img_paths]):
            #    None
            #shutil.rmtree(img_patched_dir)
    else:
        logger.info("No operations performed.")
if __name__ == "__main__":
    main()