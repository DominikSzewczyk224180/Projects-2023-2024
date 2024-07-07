# Written by Dániel

# Importing necessary libraries
from typing import List, Tuple, Dict
import sys, os
import numpy as np
import cv2
import logging
import time
import json
import argparse
from h5py import File as HDF5File
import shutil
from fastapi import UploadFile
from tensorflow.keras.models import load_model # type: ignore


# Set up logging to a file
log_filename_upload = os.path.join('logs/prediction_times_UPLOAD.log')
logging.basicConfig(filename=log_filename_upload, level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Importing functions
from process_image import process_image, crop_dimensions, padder
from metric import F1IoUMetric
from predictions import custom_argmax, custom_colormap, create_colored_heatmaps, get_num_classes, unpatchify

def original_finder(chosen_directory: str) -> List[str]:
    """
    # Written by Dániel
    
    Finds all files in the specified directory and its subdirectories that start with 'ORG_'.
    
    Parameters:
    chosen_directory (str): The directory in which to search for files.
    
    Returns:
    List[str]: A list of file paths that start with 'ORG_'.
    """
    # Initialize an empty list to store the file paths
    filepaths = []
    # Walk through the directory and its subdirectories
    for dirpath, dirnames, filenames in os.walk(chosen_directory):
        # Iterate through all files in the current directory
        for file in filenames:
            # Check if the file name starts with 'ORG_' for original files
            if file.startswith('ORG_'):
                # Construct the full file path and add it to the list
                filepaths.append(os.path.join(dirpath, file))
    
    return filepaths

def comb_prob_finder(filepaths: List[str], counter: int) -> Tuple[str, str]:
    """
    Written by Dániel

    Given a list of file paths and a counter, returns the corresponding combined mask and probability map file paths.
    
    Parameters:
    filepaths (List[str]): A list of file paths starting with 'ORG_'.
    counter (int): An index (1-based) to select a specific file path from the list.
    
    Returns:
    Tuple[str, str]: A tuple containing the file paths for the combined mask and probability map.
    """
    # Get the file path based on the counter (1-based index)
    filepath = filepaths[counter]

    # Replace 'ORG_' with 'COMB_' to get the combined mask file path
    combined_mask = filepath.replace('ORG_', 'COMB_')
    # Replace 'ORG_' with 'PROB_' to get the probability map file path
    probability_map_1 = filepath.replace('ORG_', 'PROB_mask1')
    probability_map_2 = filepath.replace('ORG_', 'PROB_mask2')
    probability_map_3 = filepath.replace('ORG_', 'PROB_mask3')

    probability_maps = [probability_map_1, probability_map_2, probability_map_3]

    return combined_mask, probability_maps

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

def pad_image(img_path: str, patch_size: int = 256) -> None:
    """
    Written by Kian, modified by Dániel
    
    Process a single image by cropping, normalizing, padding, and patchifying it. Save the resulting patches.

    Args:
        img_path (str): The path to the image to be processed.
        patch_size (int): The size of the patches. Defaults to 256.

    Returns:
        None
    """
    # Read the image in grayscale mode
    img = cv2.imread(img_path, 0)
    # Get the cropping dimensions
    y_min, y_max, x_min, x_max = crop_dimensions(img)
    # Crop the image
    img_cropped = img[y_min:y_max, x_min:x_max]

    # Pad the image to fit patch size
    img_padded = padder(img_cropped, patch_size)

    return img_padded

def load_patches_from_folder(folder: str, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Written by Dániel.

    Load image patches from a specified folder.

    Parameters:
    - folder: The directory containing the image patches.
    - target_size: The target size for resizing images.

    Returns:
    - An array of image patches.
    """
    images = []
    filenames = []
    for filename in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, target_size)
                img = img.astype('float32') / 255.0  # Normalize to [0, 1]
                img = np.expand_dims(img, axis=-1)  # Add channel dimension
                images.append(img)
                filenames.append(filename)

    for filename in filenames:
        img_path = os.path.join(folder, filename)
        #os.remove(img_path)

    return images

# Create a function to predict masks for test images
def predict_masks(model_path, data_path, patch_size: int = 256) -> None:
    """
    Written by Benni, modified by Dániel

    Predict masks for images using the provided model.
    
    Parameters:
    model: The trained Keras model for prediction.
    path (str): The path to the directory containing the test images.
    patch_size (int): Size of each patch.
    """

    # Get the model name from model path
    model_path = model_path.replace('\\', '/')
    model_name = model_path.split('/')[-1].split('.')[0]

    # Get the name of the last folder in the path
    data_path = data_path.replace('\\', '/')
    final_folder = data_path.split('/')[-1]


    saving_folder = os.path.join(root_dir, f'data/predictions/{final_folder}_{model_name}')

    # Making a directory to store the predictions
    os.makedirs(saving_folder, exist_ok=True)

    

    # Load the model from the provided path
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

    # Loading the model
    model = load_model(model_path, custom_objects={'F1IoUMetric': lambda **kwargs: F1IoUMetric(num_classes=num_classes)})

    # Loading in the files from the upload folder
    files = os.listdir(data_path)

    if len(files) % 169:
        print(f"Total number of patches should be divisible by 169. Number of files found: {len(files)}.")
        print("Still proceeding with the prediction, but predictions might be wrong.")

    
    
    for i in range(0, len(files), 169):
        subset_files = files[i:i+169]

        

        patches = []
        for filename in subset_files:

            # Construct the path to the image
            patch_path = os.path.join(data_path, filename)
            patch_path = os.path.normpath(patch_path)
            

            # Load 169 patches from the folder
            patch = cv2.imread(patch_path, cv2.IMREAD_GRAYSCALE)

            

            patches.append(patch)
            
        # Ensure we have exactly 169 patches
        if len(patches) % 169 != 0:
            print(f"Expected 169 patches but found {len(patches)}.")

        patches = np.array(patches)
        
        # Predict on the patches
        preds = model.predict(patches)
       
        # Unpatchify the predictions
        predicted_mask = unpatchify(preds, patch_size)

        patches = np.expand_dims(patches, axis=-1)

        print("patches shape:::::::::::::::: ", patches.shape)

        # Unpatchify the original patches
        ORG_image = unpatchify(patches, patch_size)

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        
        # Construct the path to save the original image
        saving_path_ORG = os.path.join(root_dir, f'data/predictions/{final_folder}_{model_name}/ORG_{filename.split(".")[0]}.png')

        # Save the original image
        cv2.imwrite(saving_path_ORG, ORG_image)

        # Save the probability maps
        for i in range(predicted_mask.shape[-1] - 1):
            custom_cmap = custom_colormap()
            
            saving_path_PROB = os.path.join(f'data/predictions/{final_folder}_{model_name}/PROB_mask{i+1}_{filename.split(".")[0]}.png')
            create_colored_heatmaps(predicted_mask[...,i], os.path.join(saving_path_PROB), custom_cmap)
            
        # Threshold the predicted mask
        predicted_classes = custom_argmax(predicted_mask, bias=0.5)

        

        root_mask = (predicted_classes == 0).astype(np.uint8) * 255
        shoot_mask = (predicted_classes == 1).astype(np.uint8) * 255
        seed_mask = (predicted_classes == 2).astype(np.uint8) * 255

        # Saving each as INDIV_ masks
        cv2.imwrite(os.path.join(root_dir, f'data/predictions/{final_folder}_{model_name}/INDIV_mask1_{filename.split(".")[0]}_{model_name}.png'), root_mask)
        cv2.imwrite(os.path.join(root_dir, f'data/predictions/{final_folder}_{model_name}/INDIV_mask2_{filename.split(".")[0]}_{model_name}.png'), shoot_mask)
        cv2.imwrite(os.path.join(root_dir, f'data/predictions/{final_folder}_{model_name}INDIV_mask3_{filename.split(".")[0]}_{model_name}.png'), seed_mask)

        # Colorize the masks
        colorized_image = colorize_masks(predicted_classes)

        # Construct the path to save the combined mask
        saving_path_COMB = os.path.join(root_dir, f'data/predictions/{final_folder}_{model_name}/COMB_{filename.split(".")[0]}.png')

        # Save the combined mask
        cv2.imwrite(saving_path_COMB, colorized_image)

        # Print statements to show the path to the saved image
        print('Data saved to:', saving_path_COMB)
        print(f"Saved predicted mask {filename}.")
    
def list_h5_files_by_type(directory: str) -> Tuple[List[str], List[str]]:
    """
    Written by Benni, modified by Dániel

    Lists all .h5 files in the given directory and classifies them as binary or multiclass based on filename.

    Args:
        directory (str): The path to the directory to list files from.

    Returns:
        tuple: A tuple containing two lists:
            - binary_h5_files: A list of binary models' filenames.
            - multiclass_h5_files: A list of multiclass models' filenames.
    """
    binary_h5_files = []
    multiclass_h5_files = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.h5'):  # Check for .h5 extension
            filename = filename.lower()

            # Classify the files based on the filename
            if filename.endswith('binary.h5'):
                binary_h5_files.append(filename)
            else:
                multiclass_h5_files.append(filename)

    return binary_h5_files, multiclass_h5_files

def upload_images(images: List[UploadFile]) -> Dict[str, List[str]]:
    """
    Written by Dániel.

    Upload and save image files to a specified directory.

    Parameters:
    - images: List of UploadFile objects representing the images to be uploaded.
    - root_dir: The root directory.

    Returns:
    - A dictionary containing lists of saved file names or an error message.
    """
    # Define the allowed file extensions
    allowed_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']

    # Define the path to the upload folder
    upload_folder_path = 'data/raw/user_upload'

    # Ensure the upload folder exists
    os.makedirs(upload_folder_path, exist_ok=True)

    # Delete the contents of the upload folder prior to saving new files
    for file in os.listdir(upload_folder_path):
        file_path = os.path.join(upload_folder_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted previous file {file_path}")
    
    # Initialize lists to store the names of saved and invalid files
    saved_files = []
    invalid_files = []

    for image in images:
        # Check if the file extension is allowed
        file_ext = os.path.splitext(image.filename)[1].lower()
        if file_ext in allowed_extensions:
            # Define the path to save the image
            file_path = os.path.join(upload_folder_path, image.filename)

            # Save the image
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)

            # Append the saved file name to the list
            saved_files.append(image.filename)
        else:
            # Collect invalid files for reporting
            invalid_files.append(image.filename)

    if invalid_files:
        # Generate error message for invalid files
        error_message = f"Unsupported file format. Supported formats are: {', '.join(allowed_extensions)}"
        logging.error(error_message)
        return {"error": error_message, "files_saved": saved_files}
    else:
        return {"files_saved": saved_files}
    
def unpatchify(data: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Written by Benhamin, modified by Dánielű

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
            patch_index = x * patches_per_dim + y
            # Accumulate the sum of probabilities and count contributions
            reconstructed_img[y_start:y_end, x_start:x_end, :] += data[patch_index]
            count[y_start:y_end, x_start:x_end, :] += 1
    # Avoid division by zero
    count[count == 0] = 1
    # Average the overlapping regions by dividing accumulated sums by counts
    reconstructed_img /= count

    return reconstructed_img

def predict_uploaded_images(model_name: str, patch_size: int = 256) -> Tuple[List[str], List[str], List[str], str]:
    """
    Written by Dániel.

    Predict on uploaded images using a specified model.

    Parameters:
    - model_name: The name of the model to be used for predictions.
    - model_type: The type of the model ('multiclass' or other).
    - patch_size: The size of the image patches.

    Returns:
    - A tuple containing lists of paths for the original, combined, and probability maps, and a status message.
    """

    start_time = time.time()  # Start the timer

    all_image_ORG_paths = []
    all_image_COMB_paths = []
    all_image_PROB_paths = []

    # Construct the path to the model from the input
    model_path = f'models/{model_name}.h5'

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

    # Loading the model
    model = load_model(model_path, custom_objects={'F1IoUMetric': lambda **kwargs: F1IoUMetric(num_classes=num_classes)})

    # Define the path to the upload folder
    upload_folder = 'data/raw/user_upload'

    # Loading in the files from the upload folder
    files = os.listdir(upload_folder)

    # Get the number of images
    num_images = len(files)

    for filename in files:
        # Construct the path to the image
        image_path = os.path.join(upload_folder, filename)
        image_path = os.path.normpath(image_path)

        # Creating lists to store the image paths
        ORG_paths = []
        COMB_paths = []
        PROB_paths = []

        # Save the padded image
        ORG_image = pad_image(image_path, patch_size)

        # Construct the path to save the original image
        saving_path_ORG = f'data/predictions/user_uploads/ORG_{filename.split(".")[0]}_{model_name}.png'

        # Save the original image
        cv2.imwrite(saving_path_ORG, ORG_image)

        # Append the path to the list
        ORG_paths.append(saving_path_ORG)

        # Print the path to the original image
        print("Original of the image saved to ", saving_path_ORG)

        # Process the image
        folder_path = process_image(image_path)

        # Load 169 patches from the folder
        patches = load_patches_from_folder(folder_path)

        # Convert the list of patches to a numpy array
        patches = np.array(patches)
        
        # Ensure we have exactly 169 patches
        if len(patches) % 169:
            print(f"Expected 169 patches but found {len(patches)}.")

        # Predict on the patches
        preds = model.predict(patches)
        
        # Unpatchify the predictions
        predicted_mask = unpatchify(preds, patch_size)

        # Save the probability maps
        for i in range(predicted_mask.shape[-1] - 1):
            custom_cmap = custom_colormap()
            
            saving_path_PROB = f'data/predictions/user_uploads/PROB_mask{i+1}_{filename.split(".")[0]}_{model_name}.png'
            create_colored_heatmaps(predicted_mask[...,i], saving_path_PROB, custom_cmap)
            PROB_paths.append(saving_path_PROB)

        

        # Construct the path to save the combined mask
        saving_path_COMB = f'data/predictions/user_uploads/COMB_{filename.split(".")[0]}_{model_name}.png'

        # Creating a list to store the image paths
        COMB_paths.append(saving_path_COMB)

        # Threshold the predicted mask
        predicted_classes = custom_argmax(predicted_mask, bias=0.5)

        # PRinting unique values of predicted_mask
        print("Unique values of predicted mask::::::::::::::: ", np.unique(predicted_classes))

        root_mask = (predicted_classes == 0).astype(np.uint8) * 255
        shoot_mask = (predicted_classes == 1).astype(np.uint8) * 255
        seed_mask = (predicted_classes == 2).astype(np.uint8) * 255

        # Saving each as INDIV_ masks
        cv2.imwrite(f'data/predictions/user_uploads/INDIV_mask1_{filename.split(".")[0]}_{model_name}.png', root_mask)
        cv2.imwrite(f'data/predictions/user_uploads/INDIV_mask2_{filename.split(".")[0]}_{model_name}.png', shoot_mask)
        cv2.imwrite(f'data/predictions/user_uploads/INDIV_mask3_{filename.split(".")[0]}_{model_name}.png', seed_mask)
        
        # Colorize the masks
        colorized_image = colorize_masks(predicted_classes)

        # Save the combined mask
        cv2.imwrite(saving_path_COMB, colorized_image)

        # Print statements to show the path to the saved image
        print('Data saved to:', saving_path_COMB)
        print(f"Saved predicted mask {filename}.")

        # Deleting the contents of the folder_path folder
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            print(file_path)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Append the lists to the main lists
        all_image_ORG_paths.append(ORG_paths)
        all_image_COMB_paths.append(COMB_paths)
        all_image_PROB_paths.append(PROB_paths)
    
    stringed_org_paths = str(all_image_ORG_paths)
    stringed_comb_paths = str(all_image_COMB_paths)
    stringed_prob_paths = str(all_image_PROB_paths)

    result = {
        "ORG_paths": stringed_org_paths,
        "COMB_paths": stringed_comb_paths,
        "PROB_paths": stringed_prob_paths,
        "output_message": "Predictions saved."
    }
    
    with open("output.json", "w") as f:
        json.dump(result, f)

    # Geting the time per image
    end_time = time.time()  # End the timer
    total_time = end_time - start_time

    if num_images != 0:
        time_per_image = total_time / num_images
    else:
        time_per_image = 0

    # Logging
    logger.info(f"Time per image: {time_per_image:.4f} seconds. Number of images: {num_images}. Total time: {total_time:.4f} seconds.")


    return all_image_ORG_paths, all_image_COMB_paths, all_image_PROB_paths, "Predictions saved."

def main():
    parser = argparse.ArgumentParser(description="Image processing script.")
    subparsers = parser.add_subparsers(dest="command")

    # Upload images command
    upload_parser = subparsers.add_parser("upload")
    upload_parser.add_argument("images", nargs="+", help="List of image file paths to upload")

    # Predict images command
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model_name", required=True, help="Name of the model to use for prediction")
    predict_parser.add_argument("--patch_size", type=int, default=256, help="Patch size for prediction")

    args = parser.parse_args()

    if args.command == "upload":
        result = upload_images(args.images)
        print(result)
    elif args.command == "predict":
        result = predict_uploaded_images(args.model_name, args.patch_size)
        print(result)

    
if __name__ == '__main__':
    image_paths = predict_uploaded_images('resnet_unet_10')