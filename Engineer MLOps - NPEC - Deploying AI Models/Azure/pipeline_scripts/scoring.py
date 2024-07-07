import os
import sys
import json
import tensorflow as tf
import numpy as np
import cv2
import base64
from metric import f1, iou
from  predictions import unpatchify
from  process_image import process_image

def init():
    # Define the model as a global variable to be used later in the predict function
    global model

    # Get the path where the model is saved, it is set in the environment variable AZUREML_MODEL_DIR by the deployment configuration
    base_path = os.getenv("AZUREML_MODEL_DIR")

   # If the environment variable is not set, use a local path
    if base_path is None:
        base_path = 'models'

    print(f"base_path: {base_path}")

    # add the model file name to the base_path
    model_path = os.path.join(base_path, 'U-net.h5') # local
    # model_path = os.path.join(base_path, "INPUT_model", 'model.keras') # azure
    # print the model_path to check if it is correct
    print(f"model_path: {model_path}")


    # Load the model
    model = tf.keras.models.load_model(model_path, custom_objects={'f1':f1, 'iou':iou})
    print("Model loaded successfully")



def run(img_path):
    # Process the image using the process_image function
    data = json.loads(img_path)

    # Get the base64-encoded image data, print the data to see make sure it is correct
    base64_image = data["data"]
    print(f"base64_image: {base64_image}")

    # Decode the base64 string into bytes
    image_bytes = base64.b64decode(base64_image)
    print(f"image_bytes: {image_bytes}")

    # Convert bytes to numpy array
    image_array = np.frombuffer(image_bytes, np.uint8)

    # Decode image array using OpenCV
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    img_patches = process_image(image)
    print(img_patches.shape)

    predictions= []
    for patch in img_patches:
        # Assuming model.predict() expects a batch-like input, we reshape the patch
        print(patch.shape)
        prediction = model.predict(patch)
        predictions.append(prediction)

    predictions = np.array(predictions)
    print(predictions.shape)
    predictions = predictions.reshape((13 * 13, 256, 256, 1))
    unpatched = unpatchify(predictions, 256)
    predictions_list = unpatched.tolist()

    # Return predictions in JSON format
    return json.dumps(predictions_list)

if __name__ == "__main__":

    init()
    run("sample_data.json")