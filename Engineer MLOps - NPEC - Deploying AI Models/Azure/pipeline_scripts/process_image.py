# Made by Kian van Holst
# Standard library imports
import argparse
import datetime
import os
import sys
from typing import List, Optional, Tuple

# Third-party imports
import cv2
import numpy as np
from patchify import patchify # type: ignore
from tqdm import tqdm # type: ignore

def crop_dimensions(img: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Written by Kian
    
    Extracts the crop dimensions for the given image.

    Args:
        img (np.ndarray): The input image array.

    Returns:
        Tuple[int, int, int, int]: The coordinates (y_min, y_max, x_min, x_max) for cropping the image.
    """
    img = img[:, :4000] 
    # Apply morphological operations
    im_blurred = cv2.medianBlur(img, 5) 
    _, output_im = cv2.threshold(im_blurred, 50, 250, cv2.THRESH_BINARY_INV)
    kernel = np.ones((11, 11), np.uint8)
    output_im = cv2.erode(output_im, kernel, iterations=1)
    output_im = cv2.dilate(output_im, kernel, iterations=1)
    # Retrieve stats of detected components and get the largest component
    _, _, stats, _ = cv2.connectedComponentsWithStats(output_im) 
    largest_label = np.argmax(stats[:, cv2.CC_STAT_AREA])

    # Extracts the coordinates of largest component
    x, y, w, h = stats[largest_label][:4] 
    # Adds a bit of threshold
    side_length = max(w, h) + 10 
    # This part ensures a centered crop
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
    # Step size is set to overlap a bit to catch more detail that can be cut off
    step_size = int(patch_size / 8) * 7 
    h, w = image.shape[:2] # Calculates the number of patches
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
    Written by Kian, slightly modified by Dániel
    
    Save image patches to the corresponding file path.

    Args:
        fpath (str): The file path to save the patches.
        img_patches (np.ndarray): The array of image patches.

    Returns:
        None
    """
    # Get the size of each patch
    patch_size = img_patches.shape[2]
    # Get the number of patches
    n_patches = img_patches.shape[0]
    # Reshape patches and convert to 8-bit integer type
    img_patches = (img_patches.reshape(-1, patch_size, patch_size, 1) * 255).astype(np.uint8)
    # Modify file path to 'processed' instead of 'raw'
    img_patch_path = fpath.replace('raw', 'processed')
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(img_patch_path), exist_ok=True)

    # Loop over each patch to save them individually
    for i, patch in enumerate(img_patches):
        # Calculate the column number for saving the patch
        col = (i + n_patches) // n_patches
        # Calculate the row number for saving the patch
        row = (i + n_patches) % n_patches + 1
        # Create a new file name for each patch
        image_patch_path_numbered = f'{img_patch_path[:-4]}_{row:02}_{col:02}.png'
        
        # If the file already exists, remove it
        if os.path.exists(image_patch_path_numbered):
            os.remove(image_patch_path_numbered)
        
        # Save the patch as a PNG file
        cv2.imwrite(image_patch_path_numbered, patch)
    
    # Get the folder name
    folder_path = os.path.dirname(img_patch_path)

    return folder_path

def process_image(image: str, patch_size: int = 256) -> None:
    """
    Written by Kian, slightly modified by Dániel
    
    Process a single image by cropping, normalizing, padding, and patchifying it. Save the resulting patches.

    Args:
        img_path (str): The path to the image to be processed.
        patch_size (int): The size of the patches. Defaults to 256.

    Returns:
        None
    """
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Get the cropping dimensions
    y_min, y_max, x_min, x_max = crop_dimensions(gray_image)
    # Crop the image
    img_cropped = gray_image[y_min:y_max, x_min:x_max]
    # Normalize the image if necessary
    img_normalized = img_cropped / 255.0 if img_cropped.max() > 1 else img_cropped
    # Pad the image to fit patch size
    img_padded = padder(img_normalized, patch_size)
    # Patchify the padded image with a specified step size
    img_patches = patchify(img_padded, (patch_size, patch_size), step=int((patch_size/8)*7))
    # Save the patches to the specified path
    folder_path = save_patches(img_path, img_patches)

    print(f"Saved patches to********************************************* {folder_path}")

    return folder_path


def main():
    parser = argparse.ArgumentParser(description="Process an image and save the processed image in a specific directory.")
    parser.add_argument('--imagepath', type=str, required=True, help='Path to the image to be processed.')
    args = parser.parse_args()
    
    process_image(args.imagepath)

if __name__ == "__main__":
    main()