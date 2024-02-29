#task 13

from stable_baselines3 import PPO
from ot2_gym_wrapper import OT2Env
import numpy as np

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
import shutil
from shutil import copy
from collections import defaultdict
import tensorflow as tf
import keras.backend as K
from patchify import patchify, unpatchify
import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam


from sim_class import Simulation

from skimage.morphology import skeletonize
from skimage.morphology import remove_small_objects


from skan import Skeleton, summarize
from skan.csr import skeleton_to_csgraph
from skan import draw


from task_13_function import padder, create_and_save_patches, save_detected_instances_with_crop


# Initialise the simulation environment
num_agents = 1
env = OT2Env( render=True)
obs, info = env.reset()



### 
# 
#
#
# Do all the CV things so that you end up with a list of goal positions
#
#
#
###


image_path

mask = cv2.imread(image_path)





def save_detected_instances_with_crop( mask, save_path):
    combined_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    cv2.line(combined_mask, (1377, 0), (1350, combined_mask.shape[0]), (0, 0, 0), thickness=5)
    cv2.line(combined_mask, (1895, 0), (2000, combined_mask.shape[0]), (0, 0, 0), thickness=5)
    cv2.line(combined_mask, (2415, 0), (2500, combined_mask.shape[0]), (0, 0, 0), thickness=5)
    cv2.line(combined_mask, (2920, 0), (3000, combined_mask.shape[0]), (0, 0, 0), thickness=5)

    _, thresholded = cv2.threshold(combined_mask, 5, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 5000]

    contours_to_use = sorted(filtered_contours, key=cv2.contourArea, reverse=True)[:len(filtered_contours)]

    instance_id = 0
    for contour in contours_to_use:
        x, y, w, h = cv2.boundingRect(contour)
        
        
        # Save the instance's mask 
        instance_mask_filename = f"instance_{instance_id}_mask.png"
        instance_mask_save_path = os.path.join(save_path, instance_mask_filename)
        cv2.imwrite(instance_mask_save_path, combined_mask[y:y+h, x:x+w])
        
        instance_id += 1


save_detected_instances_with_crop(mask,r"C:\Users\domin\Desktop\Year 2 Block B\2023-24b-fai2-adsai-DominikSzewczyk224180\2. Robotics\Y2B-2023-OT2_Twin-main\separeted_plants_1")





def process_mask_and_image(mask_path):
    root = cv2.imread(mask_path, 0)

    skeleton = skeletonize(root)
    cleaned_skeleton = remove_small_objects(skeleton, min_size=100, connectivity=2)

    simple_skeleton_branch_data = summarize(Skeleton(cleaned_skeleton))
    last_node = simple_skeleton_branch_data['node-id-dst'].max()

    return simple_skeleton_branch_data, last_node

def get_last_node_coordinates(skeleton_data, last_node):
    filtered_data = skeleton_data[skeleton_data['node-id-src'] == last_node]
    
    if filtered_data.empty:
        print("No data found for the specified node ID.")
        return None
    
    x_coord = filtered_data['image-coord-src-1'].values[0]
    y_coord = filtered_data['image-coord-src-0'].values[0]

    return x_coord, y_coord

def process_mask_folder(folder_path):
    goal_positions_all = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            mask_path = os.path.join(folder_path, filename)

            # Assuming contours_to_use is obtained from the save_detected_instances_with_crop function
            original_image = cv2.imread("path/to/your/original/image.jpg")
            mask = cv2.imread(mask_path)
            save_path = "path/to/save/instances"

            contours_to_use = save_detected_instances_with_crop(original_image, mask, save_path)

            skeleton_data, last_node = process_mask_and_image(mask_path)

            goal_positions = []
            for contour in contours_to_use:
                x, y, w, h = cv2.boundingRect(contour)
                z_coordinate = 0.080
                
                last_node_coordinates = get_last_node_coordinates(skeleton_data, last_node)
                if last_node_coordinates:
                    goal_positions.append((last_node_coordinates[0], last_node_coordinates[1], z_coordinate))

            goal_positions_all.append(goal_positions)

    return goal_positions_all

# Example usage:
folder_path = r"C:\Users\domin\Desktop\Year 2 Block B\2023-24b-fai2-adsai-DominikSzewczyk224180\2. Robotics\Y2B-2023-OT2_Twin-main\separeted_plants_1"
goal_positions = process_mask_folder(folder_path)
print(goal_positions)





# Load the trained agent
model = PPO.load(r"C:\Users\domin\Desktop\Year 2 Block B\2023-24b-fai2-adsai-DominikSzewczyk224180\2. Robotics\Y2B-2023-OT2_Twin-main\models\good\model.zip")

for goal_pos in goal_positions:
    # Set the goal position for the robot
    env.goal_position = root_pos
    # Run the control algorithm until the robot reaches the goal position
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info  = env.step(action)
        # calculate the distance between the pipette and the goal
        distance = obs[3:] - obs[:3] # goal position - pipette position
        # calculate the error between the pipette and the goal
        error = np.linalg.norm(distance)
        # Drop the inoculum if the robot is within the required error
        if error < 0.01: # 10mm is used as an example here it is too large for the real use case
            action = np.array([0, 0, 0, 1])
            obs, rewards, terminated, truncated, info  = env.step(action)
            break

        if terminated:
            obs, info = env.reset()