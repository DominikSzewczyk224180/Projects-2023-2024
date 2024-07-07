import asyncio
import subprocess
import os
import shutil
import sys
import logging
import io
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Dict
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Request, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
import contextlib
import json
import zipfile
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from fastapi import FastAPI, HTTPException
import asyncio
import sys
import logging
import platform

from keras.models import load_model, model_from_json
from h5py import File as HDF5File


app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DirectoryRequest(BaseModel):
    path: str

#os.chdir('../')

# parent_dir = (os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(parent_dir)
from scr.metric import F1IoUMetric

#from scr.data.data_processing import get_masks
origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",  # Add any other origins you need
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ██████╗░░█████╗░████████╗░█████╗░
# ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗
# ██║░░██║███████║░░░██║░░░███████║
# ██║░░██║██╔══██║░░░██║░░░██╔══██║
# ██████╔╝██║░░██║░░░██║░░░██║░░██║
# ╚═════╝░╚═╝░░╚═╝░░░╚═╝░░░╚═╝░░╚═╝

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), target_dir: Path = 'data/raw'):

    # Create the nested directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)

    # Save the uploaded file to a temporary location
    file_location = target_dir / file.filename
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Unpack the zip file if the uploaded file is a zip
    if file.filename.endswith('.zip'):
        try:
            with zipfile.ZipFile(file_location, 'r') as zip_ref:
                # Extract all contents, ignoring __MACOSX directories and files
                for member in zip_ref.namelist():
                    if not member.startswith('__MACOSX'):
                        zip_ref.extract(member, target_dir)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid zip file")

        # Delete the original zip file after extraction
        os.remove(file_location)


    return {"filename": file.filename}

# These functions you would normally import from you .py script
def list_directories(path: str) -> List[str]:
    """
    List directories in the given start path.

    Args:
        start_path (str): The directory path containing raw data directories. Defaults to 'data/raw'.

    Returns:
        List[str]: The list of directories.
    """

    if not os.path.exists(path):
        logger.warning(f"Path does not exist: {path}")
        return []

    entries = [entry for entry in os.listdir(path) if os.path.isdir(os.path.join(path, entry))]
    entries.sort()

    if not entries:
        logger.warning(f"No directories found in {path}")

    return entries


# These functions you would normally import from you .py script
def get_raw_data_info(path):
    """
    Counts the number of files in the 'images' and 'masks' subdirectories of the given directory.

    Parameters:
        directory (str): The path to the main directory.

    Returns:
        dict: A dictionary with the counts of files in 'images' and 'masks' subdirectories.
    """
    root_dir = os.path.join('data/raw', path)
    images_dir = os.path.join(root_dir, 'images')
    masks_dir = os.path.join(root_dir, 'masks')

    images_count = len(os.listdir(images_dir)) if os.path.exists(images_dir) else 0
    masks_count = len(os.listdir(masks_dir)) if os.path.exists(masks_dir) else 0
    sample = os.listdir(images_dir)[1]
    img_name = os.path.splitext(sample)[0]
    sample_path = os.path.join(images_dir, sample)

    classes = get_masks(sample_path)
    class_extensions = [(file.replace(img_name, '')) for file in classes]

    class_count = len(classes)

    return images_count, masks_count, class_count, class_extensions

# These functions you would normally import from you .py script
def get_masks(fpath: str, mask_extensions: Optional[List[str]] = None) -> List[str]:
    """
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

    if mask_extensions is None:
        return [file for file in files if file.startswith(image_file)]

    matching_masks_names = [
        f"{image_file}_{mask_extension}" for mask_extension in mask_extensions
        if f"{image_file}_{mask_extension}" in files
    ]

    return matching_masks_names



@app.post("/list_directories/")
def list_directories_endpoint(request: DirectoryRequest):
    try:
        directories = list_directories(path=request.path)
        if not directories:
            raise HTTPException(status_code=404, detail="No directories found.")
        return {"directories": directories}
    except Exception as e:
        logger.error(f"Error listing directories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_raw_data_info/")
def get_raw_data_info_endpoint(request: DirectoryRequest):
    try:
        images_count, masks_count, class_count, classes = get_raw_data_info(path=request.path)
        return {"images_count": images_count, "masks_count": masks_count, "class_count": class_count, "classes": classes}
    except Exception as e:
        logger.error(f"Error retrieving information: {e}")
        error = "Error reading dataset"
        return {"images_count": error, "masks_count": error, "class_count": error, "classes": error}


@app.delete("/delete_data/{datatype}/{directory}")
async def delete_data_endpoint(datatype:str, directory: str):
    dir = os.path.join('data', datatype, directory)
    print(dir)
    try:
        shutil.rmtree(dir)
        return {"message": "Directory deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{dir} {e}")

@app.post("/process_data/{directory}")
async def process_data_endpoint(directory: str, request: Request):
    try:
        data = await request.json()
    except Exception as e:
        data = {}

    masks = data.get('masks', [])

    if not isinstance(masks, list) or not all(isinstance(mask, str) for mask in masks):
        raise HTTPException(status_code=400, detail="Invalid format for masks. Expected a list of strings.")

    script_path = "scr/process_data.py"
    try:
        args = [sys.executable, script_path, '--dir', directory]
        if masks:
            args.extend(['--masks', *masks])

        # Asynchronously execute the Python script
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Processing failed: {stderr.decode()} for {script_path}")

        return {"message": "Data processed successfully", "output": stdout.decode()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/working_directory/")
async def get_working_directory():
    return {"working_directory": os.getcwd()}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_location = f"{'data/raw'}/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return JSONResponse({"info": f"file '{file.filename}' saved at '{file_location}'"})





# ███╗░░░███╗░█████╗░██████╗░███████╗██╗░░░░░  ██╗███╗░░██╗███████╗░█████╗░
# ████╗░████║██╔══██╗██╔══██╗██╔════╝██║░░░░░  ██║████╗░██║██╔════╝██╔══██╗
# ██╔████╔██║██║░░██║██║░░██║█████╗░░██║░░░░░  ██║██╔██╗██║█████╗░░██║░░██║
# ██║╚██╔╝██║██║░░██║██║░░██║██╔══╝░░██║░░░░░  ██║██║╚████║██╔══╝░░██║░░██║
# ██║░╚═╝░██║╚█████╔╝██████╔╝███████╗███████╗  ██║██║░╚███║██║░░░░░╚█████╔╝
# ╚═╝░░░░░╚═╝░╚════╝░╚═════╝░╚══════╝╚══════╝  ╚═╝╚═╝░░╚══╝╚═╝░░░░░░╚════╝░

def f1(y_true, y_pred):
    """
    Calculate the F1 score, the harmonic mean of precision and recall.

    Args:
        y_true (Tensor): True labels.
        y_pred (Tensor): Predicted labels.

    Returns:
        Tensor: F1 score.
    """
    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    f1_score = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return f1_score

def iou(y_true, y_pred):
    """
    Calculate the Intersection over Union (IoU) metric.

    Args:
        y_true (Tensor): True labels.
        y_pred (Tensor): Predicted labels.

    Returns:
        Tensor: IoU score.
    """
    def f(y_true, y_pred):
        y_pred = K.cast(y_pred > 0.5, dtype=K.floatx())
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
        total = K.sum(K.abs(y_true) + K.abs(y_pred), axis=[1, 2, 3])
        union = total - intersection
        iou = (intersection + K.epsilon()) / (union + K.epsilon())
        return iou

    return K.mean(f(y_true, y_pred), axis=0)



MODELS_DIR = "models"
def extract_classes(model_path):
    with HDF5File(model_path, "r") as f:
        model_config = f.attrs.get("model_config")
        if model_config is None:
            raise ValueError("model_config attribute not found in the .h5 file.")
 
        # Decode if necessary
        if isinstance(model_config, bytes):
            model_config = model_config.decode("utf-8")
 
        # Parse JSON to get model configuration
        config_dict = json.loads(model_config)
        model_architecture = model_from_json(json.dumps(config_dict))
 
    # Inspect the output layer
    output_layer = model_architecture.layers[-1]
    num_classes = output_layer.output_shape[-1]
 
    return num_classes

@app.get("/models", response_model=List[str])
async def list_models():
    models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.h5')]
    model_names = [os.path.splitext(m)[0] for m in models]
    return model_names

@app.get("/models/{model_name}/summary")
async def model_summary(model_name: str):
    model_path = os.path.join('models', f'{model_name}.h5')
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    num_classes = extract_classes(model_path)
    try:
        model = load_model(
            model_path,
            custom_objects={
                "F1IoUMetric": lambda **kwargs: F1IoUMetric(
                    num_classes=num_classes, **kwargs
                )
            },
        )

        # Capture the model summary as a string
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            model.summary()
        summary_string = stream.getvalue()
        
        return JSONResponse(content={"summary": summary_string})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


LOGS_DIR = 'logs'

@app.get("/models/{logs_name}/logs")
async def model_logs(model_name: str):
    logs_path = os.path.join(LOGS_DIR, f'models/{model_name}.json')
    if not os.path.exists(logs_path):
        raise HTTPException(status_code=404, detail=f"Logs not found {logs_path}")

    try:
        with open(logs_path, 'r') as file:
            logs = file.readlines()
        return {"logs": logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {str(e)}")
    



# ███╗░░░███╗░█████╗░██████╗░███████╗██╗░░░░░  ████████╗██████╗░░█████╗░██╗███╗░░██╗██╗███╗░░██╗░██████╗░
# ████╗░████║██╔══██╗██╔══██╗██╔════╝██║░░░░░  ╚══██╔══╝██╔══██╗██╔══██╗██║████╗░██║██║████╗░██║██╔════╝░
# ██╔████╔██║██║░░██║██║░░██║█████╗░░██║░░░░░  ░░░██║░░░██████╔╝███████║██║██╔██╗██║██║██╔██╗██║██║░░██╗░
# ██║╚██╔╝██║██║░░██║██║░░██║██╔══╝░░██║░░░░░  ░░░██║░░░██╔══██╗██╔══██║██║██║╚████║██║██║╚████║██║░░╚██╗
# ██║░╚═╝░██║╚█████╔╝██████╔╝███████╗███████╗  ░░░██║░░░██║░░██║██║░░██║██║██║░╚███║██║██║░╚███║╚██████╔╝
# ╚═╝░░░░░╚═╝░╚════╝░╚═════╝░╚══════╝╚══════╝  ░░░╚═╝░░░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝╚═╝░░╚══╝╚═╝╚═╝░░╚══╝░╚═════╝░
class TrainRequest(BaseModel):
    depth_sel: int
    data_dir: str



@app.post("/train")
async def train_endpoint(train_request: TrainRequest, background_tasks: BackgroundTasks):
    if train_request.depth_sel not in [0, 1]:
        raise HTTPException(status_code=400, detail="Invalid depth selection. Please choose 0 or 1.")
    path = os.path.join("data", "processed", train_request.data_dir)
    if not os.path.exists(path):
        raise HTTPException(status_code=400, detail="Data directory does not exist.")
    try:
        background_tasks.add_task(train, train_request.depth_sel, path)
        return {"message": "Training started successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in training: {e}")



# ██████╗░██████╗░███████╗██████╗░██╗░█████╗░████████╗
# ██╔══██╗██╔══██╗██╔════╝██╔══██╗██║██╔══██╗╚══██╔══╝
# ██████╔╝██████╔╝█████╗░░██║░░██║██║██║░░╚═╝░░░██║░░░
# ██╔═══╝░██╔══██╗██╔══╝░░██║░░██║██║██║░░██╗░░░██║░░░
# ██║░░░░░██║░░██║███████╗██████╔╝██║╚█████╔╝░░░██║░░░
# ╚═╝░░░░░╚═╝░░
# Define Pydantic models for input
class DirectoryPath(BaseModel):
    """Model to represent the directory path input."""
    image_folder: str

class ModelName(BaseModel):
    """Model to represent the path of the model input."""
    model_name: str

class PatchesFolder(BaseModel):
    """Model to represent the path of the patches folder input."""
    patches_folder: str

# Define Pydantic models for output

class LoadFilesResponse(BaseModel):
    originals_filepaths: List[str]
    combined_mask: str
    probability_map: str

@app.post("/predict_folder/")
async def process_data_endpoint(patches_folder: PatchesFolder, model_name: ModelName):
 
    script_path = "scr/pipeline.py"
    try:
        # Asynchronously execute the Python script
        process = await asyncio.create_subprocess_exec(
            sys.executable, script_path, '--predict_path', patches_folder.patches_folder,
            '--model_name', model_name.model_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()
 
        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Processing failed: {stderr.decode()} for {script_path}")
 
        return {"message": "Data processed successfully", "output": stdout.decode()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 

@app.post("/predict_single_image/")
async def predict_new_image(model_name: ModelName) -> Dict[str, str]:
    """
    Written by Dániel
    
    Endpoint to predict on a single image uploaded by the user.

    This endpoint accepts an image file, processes the image by saving it to a predefined 
    upload directory, and returns a JSON response with the names of the saved files.

    :param model_name: ModelName object containing the name of the model.
    :return: JSON response containing the list of filenames that were successfully saved.
    """
    script_path = "scr/API_functions_inference.py"
    print(model_name.model_name)
    try:
        # Asynchronously execute the Python script
        process = await asyncio.create_subprocess_exec(
            sys.executable, script_path, "predict", "--model_name", model_name.model_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {stderr.decode()}")


        # Read the output from the file
        with open("output.json", "r") as f:
            result = json.load(f)

        return {
            "ORG_paths": result["ORG_paths"],
            "COMB_paths": result["COMB_paths"],
            "PROB_paths": result["PROB_paths"],
            "output_message": result["output_message"]
        }

    
    except Exception as e:
        # Log the error
        logging.error(f"Error in predict_new_image: {e}")
        # Raise an HTTP exception with a 500 status code
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")


@app.post("/upload_images/")
async def upload_image_endpoint(images: List[UploadFile] = File(...)) -> Dict[str, List[str]]:
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
    
    
# ░█████╗░███╗░░██╗░█████╗░██╗░░░░░██╗░░░██╗░██████╗███████╗
# ██╔══██╗████╗░██║██╔══██╗██║░░░░░╚██╗░██╔╝██╔════╝██╔════╝
# ███████║██╔██╗██║███████║██║░░░░░░╚████╔╝░╚█████╗░█████╗░░
# ██╔══██║██║╚████║██╔══██║██║░░░░░░░╚██╔╝░░░╚═══██╗██╔══╝░░
# ██║░░██║██║░╚███║██║░░██║███████╗░░░██║░░░██████╔╝███████╗
# ╚═╝░░╚═╝╚═╝░░╚══╝╚═╝░░╚═╝╚══════╝░░░╚═╝░░░╚═════╝░╚══════╝

counter = 1



@app.get("/download-predictions/")
async def download_predictions(folder: str):
    folder_path = os.path.join('data/predictions', folder)
    if os.path.isdir(folder_path):
        zip_io = BytesIO()
        shutil.make_archive(folder_path, 'zip', folder_path)
        with open(f"{folder_path}.zip", 'rb') as f:
            zip_io.write(f.read())
        zip_io.seek(0)
        os.remove(f"{folder_path}.zip")
        return StreamingResponse(zip_io, media_type="application/x-zip-compressed", headers={"Content-Disposition": f"attachment; filename={folder}.zip"})
    else:
        raise HTTPException(status_code=404, detail="Folder not found")
# ███████╗███████╗███████╗██████╗░██████╗░░█████╗░░█████╗░██╗░░██╗  ██╗░░░░░░█████╗░░█████╗░██████╗░
# ██╔════╝██╔════╝██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔══██╗██║░██╔╝  ██║░░░░░██╔══██╗██╔══██╗██╔══██╗
# █████╗░░█████╗░░█████╗░░██║░░██║██████╦╝███████║██║░░╚═╝█████═╝░  ██║░░░░░██║░░██║██║░░██║██████╔╝
# ██╔══╝░░██╔══╝░░██╔══╝░░██║░░██║██╔══██╗██╔══██║██║░░██╗██╔═██╗░  ██║░░░░░██║░░██║██║░░██║██╔═══╝░
# ██║░░░░░███████╗███████╗██████╔╝██████╦╝██║░░██║╚█████╔╝██║░╚██╗  ███████╗╚█████╔╝╚█████╔╝██║░░░░░
# ╚═╝░░░░░╚══════╝╚══════╝╚═════╝░╚═════╝░╚═╝░░╚═╝░╚════╝░╚═╝░░╚═╝  ╚══════╝░╚════╝░░╚════╝░╚═╝░░░░░
class FeedbackRequest(BaseModel):
    dataset_model: str
    imageId: str
    feedback: str

@app.get("/list_images/{dataset_model}")
async def list_images(dataset_model: str):
    try:
        images_dir = os.path.join('data/predictions', dataset_model)
        checked_dirs = [
            os.path.join('data/predictions', dataset_model, 'checked', 'model_correct'),
            os.path.join('data/predictions', dataset_model, 'checked', 'model_incorrect'),
            os.path.join('data/predictions', dataset_model, 'checked', 'user_uploads_correct')
        ]
        images = [img for img in os.listdir(images_dir) if img.startswith('ORG_') and img.endswith('.png')]

        for checked_dir in checked_dirs:
            if os.path.exists(checked_dir):
                images += [f"checked/{os.path.basename(checked_dir)}/{img}" for img in os.listdir(checked_dir) if img.startswith('ORG_') and img.endswith('.png')]
        images = [os.path.join('../../backend/ATHENA/data/predictions',dataset_model, img) for img in images]
        if not images:
            raise HTTPException(status_code=404, detail="No images found.")
        return {"images": images}
    except Exception as e:
        logger.error(f"Error listing images: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set_feedback/")
async def set_feedback(feedback: FeedbackRequest):

    mask_suffix = {
        "root": "mask1",
        "seed": "mask2",
        "shoot": "mask3"
    }

    try:
        dataset_model = feedback.dataset_model
        image_id = feedback.imageId
        feedback_value = feedback.feedback.lower()
        #mask_type = feedback.maskType

        # if mask_type not in mask_suffix:
        #     raise HTTPException(status_code=400, detail="Invalid mask type. Choose from 'root', 'seed', 'shoot', 'occluded_root'.")

        #src_mask_paths = [os.path.join('data/predictions', dataset_model, image_id.replace('ORG_', f'PROB_{mask_suffix[mask_type]}_')) for mask_type in mask_suffix] #replace('.png', f'_{mask_suffix[mask_type]}'))
 
        if feedback_value == "correct":
            dst_dir = os.path.join('data/predictions', dataset_model, 'checked', 'model_correct')
        elif feedback_value == "incorrect":
            dst_dir = os.path.join('data/predictions', dataset_model, 'checked', 'model_incorrect')
        else:
            raise HTTPException(status_code=400, detail="Invalid feedback value. Use 'correct' or 'incorrect'.")

        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        #dst_mask_path = os.path.join(dst_dir, os.path.basename(src_mask_path))

        # if not os.path.exists(src_mask_path):
        #     raise HTTPException(status_code=404, detail="Mask not found.")
        files = os.listdir(os.path.join("data/predictions", dataset_model))

        pred_paths = [file for file in files if os.path.splitext(image_id)[0] in file]
        for pred_path in pred_paths:
            shutil.move(os.path.join("data/predictions", dataset_model, pred_path), dst_dir)

        return {"message": f"Mask moved to {dst_dir} folder."}
    except Exception as e:
        logger.error(f"Error setting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/submitFeedback/")
# async def submit_feedback(feedback: FeedbackRequest):
#     try:
#         await set_feedback(feedback)
#         return {"message": "Feedback submitted successfully."}
#     except Exception as e:
#         logger.error(f"Error submitting feedback: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


@app.post("/download_mask/")
async def download_mask(dataset_model: str = Form(...), imageId: str = Form(...)):
    try:
        base_path = os.path.join("data/predictions", dataset_model)
        subdirs = ['checked/model_correct', 'checked/model_incorrect', 'checked/user_uploads_correct']
        files = os.listdir(base_path)

        pred_paths = [os.path.join(base_path, file) for file in files if os.path.splitext(imageId)[0] in file]

        for subdir in subdirs:
            subdir_path = os.path.join(base_path, subdir)
            if os.path.exists(subdir_path):
                subdir_files = os.listdir(subdir_path)
                pred_paths.extend([os.path.join(subdir_path, file) for file in subdir_files if os.path.splitext(imageId)[0] in file])

        zip_filename = f"{imageId}_preds.zip"
        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w') as zipf:
            for file in pred_paths:
                zipf.write(file, os.path.basename(file))

        zip_buffer.seek(0)

        return StreamingResponse(zip_buffer, media_type='application/zip', headers={'Content-Disposition': f'attachment; filename={zip_filename}'})
    except Exception as e:
        logger.error(f"Error downloading mask: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/upload_corrected_mask/")
# async def upload_corrected_mask(dataset_model: str = Form(...), imageId: str = Form(...), maskType: Optional[str] = "root", correctedMask: UploadFile = File(...)):
#     mask_suffix = {
#         "root": "root_mask.tif",
#         "seed": "seed_mask.tif",
#         "shoot": "shoot_mask.tif",
#         "occluded_root": "occluded_root_mask.tif"
#     }

#     try:
#         if maskType not in mask_suffix:
#             raise HTTPException(status_code=400, detail="Invalid mask type. Choose from 'root', 'seed', 'shoot', 'occluded_root'.")

#         dst_dir = os.path.join(f'data/predictions/{dataset_model}/checked/user_uploads_correct')

#         if not os.path.exists(dst_dir):
#             os.makedirs(dst_dir)

#         corrected_mask_path = os.path.join(dst_dir, imageId.replace('org_', '').replace('.png', f'_{mask_suffix[maskType]}'))

#         with open(corrected_mask_path, "wb") as buffer:
#             shutil.copyfileobj(correctedMask.file, buffer)

#         return {"message": "Corrected mask uploaded successfully."}
#     except Exception as e:
#         logger.error(f"Error uploading corrected mask: {e}")
#         raise HTTPException(status_code=500, detail=str(e))