import cv2
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from ot2_gym_wrapper import OT2Env
import numpy as np
import pandas as pd

from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from skan import Skeleton, summarize

from task_13_function import iou, f1, padder, patchify, unpatchify, get_plants, crop_image, extract_root_coordinates, romove_objects_from_mask

from keras.models import load_model

model_root = load_model(r'C:\Users\domin\Desktop\Year 2 Block B\2023-24b-fai2-adsai-DominikSzewczyk224180\2. Robotics\Y2B-2023-OT2_Twin-main\modelroot.h5', custom_objects={'f1':f1, 'iou':iou})


# Initialise the simulation environment
num_agents = 1
env = OT2Env(render=True)
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


# Call the get_plate_image method on the instance
image_path = env.sim.get_plate_image()


patch_size = 128

 # Read the image
image = cv2.imread(image_path)
image = padder(image, patch_size)

patches = patchify(image, (patch_size, patch_size, 1), step=patch_size)
 
i = patches.shape[0]
j = patches.shape[1]
    
patches = patches.reshape(-1, patch_size, patch_size, 1)
patches.shape
 
preds = model_root.predict(patches/255)
 
preds = preds.reshape(i, j, 128, 128)
   
predicted_mask = unpatchify(preds, (image.shape[0], image.shape[1]))
   
    # Cropping the predicted mask
crop_top, crop_bottom, crop_left, crop_right = 33, 33, 11, 11
predicted_mask = predicted_mask[crop_top:-crop_bottom, crop_left:-crop_right]

def show_cropped_square(image,mask):
   
    
    th, output_im = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((9, 9), dtype="uint8")
    im_d = cv2.dilate(output_im, kernel, iterations=1)
    im_closing = cv2.erode(im_d, kernel, iterations=10)
    edges = cv2.Canny(im_closing, threshold1=100, threshold2=240)
    
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    im = cv2.drawContours(image, contours, contourIdx=-1, color=(0, 255, 0), thickness=2)
    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    x, y, w, h = cv2.boundingRect(largest_contour)

    max_side = max(w, h)
    center_x = x + w // 2
    center_y = y + h // 2
    x = center_x - max_side // 2
    y = center_y - max_side // 2
    
        
    cropped_image = image[y:y + max_side, x:x + max_side]
    cropped_mask = mask[y:y + max_side, x:x + max_side]

    return cropped_image, cropped_mask

cropped_image, cropped_mask = show_cropped_square(image, predicted_mask)

cropped_mask_normalized = (cropped_mask * 255).astype(np.uint8)
cv2.imwrite("plants.png", cropped_mask_normalized)

mask_path = r"C:\Users\domin\Desktop\Year 2 Block B\2023-24b-fai2-adsai-DominikSzewczyk224180\2. Robotics\Y2B-2023-OT2_Twin-main\plants.png"

mask = cv2.imread(mask_path)


def save_detected_instances_with_crop(mask_path, save_path):
    # Crop into 5 equal rectangles
    combined_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    total_width = combined_mask.shape[1]
    rect_width = total_width // 5

    for i in range(5):
        start_x = i * rect_width
        end_x = (i + 1) * rect_width

        instance_mask_filename = f"instance_{i}_mask.png"
        instance_mask_save_path = os.path.join(save_path, instance_mask_filename)
        cv2.imwrite(instance_mask_save_path, combined_mask[:, start_x:end_x])

save_detected_instances_with_crop(mask_path, r"C:\Users\domin\Desktop\Year 2 Block B\2023-24b-fai2-adsai-DominikSzewczyk224180\2. Robotics\Y2B-2023-OT2_Twin-main\separeted_plants_1")




def process_mask_and_image(mask_path):
    root = cv2.imread(mask_path, 0)
    
    skeleton = skeletonize(root)
    cleaned_skeleton = remove_small_objects(skeleton, min_size=200, connectivity=2)

    simple_skeleton_branch_data = summarize(Skeleton(cleaned_skeleton))
    last_node = simple_skeleton_branch_data['node-id-dst'].max()

    # print(f"File: {mask_path}, Last Node ID: {last_node}")
    # print("Skeleton Data:")
    # last_node_data = simple_skeleton_branch_data[simple_skeleton_branch_data['node-id-dst'] == last_node]
    # print(last_node_data[['image-coord-src-1', 'image-coord-src-0']])
    # print(root.shape)

    if last_node is None:
        print("No last node ID found. Skipping...")
        return None, None

    return simple_skeleton_branch_data, last_node



def get_last_node_coordinates(skeleton_data, last_node):
    # Filter the data based on the last node ID
    last_node_data = skeleton_data[skeleton_data['node-id-dst'] == last_node]

    if last_node_data.empty:
        print(f"No data found for the specified node ID {last_node}. Skipping...")
        return None

    x_coord = last_node_data['image-coord-src-1'].values[0]
    y_coord = last_node_data['image-coord-src-0'].values[0]

    return x_coord, y_coord




folder_path = r"C:\Users\domin\Desktop\Year 2 Block B\2023-24b-fai2-adsai-DominikSzewczyk224180\2. Robotics\Y2B-2023-OT2_Twin-main\separeted_plants_1"


goal_positions = []

previous_width = 0  
for filename in os.listdir(folder_path):
    if filename.endswith(".png"):  
        image_path = os.path.join(folder_path, filename)
        
        combined_mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
        skeleton_data, last_node = process_mask_and_image(image_path)
        
        
        coordinates = get_last_node_coordinates(skeleton_data, last_node)

        
        
        if coordinates:
            
            coordinates = (coordinates[0] + previous_width, coordinates[1])
            
            goal_positions.append([filename, coordinates[0], coordinates[1], 0.12])
            
            
            previous_width += combined_mask.shape[1] 



# plants, plant_img,  skeleton_data = get_plants(image,model_root)

# df = extract_root_coordinates(plants, skeleton_data, plant_img)

# y, x, l = crop_image(image)
# cropped_image = image[y:y+l, x:x+l]
            
df = pd.DataFrame(goal_positions, columns=["Plant", "Y", "X", "Z"])

for index, row in df.iterrows():
    x, y = int(row['Y']), int(row['X'])
    cv2.circle(mask, (x, y), 5, (0, 255, 0), -1)  # Draw a filled green circle

# Display the image with circles using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
plt.scatter(df['Y'], df['X'], c='red', marker='o', label='Points')
plt.legend()
plt.show()


# print(df)
#                  Plant     X     Y     Z
# 0  instance_0_mask.png   386  1216  0.08
# 1  instance_1_mask.png   942  1055  0.08
# 2  instance_2_mask.png  1471  1438  0.08
# 3  instance_3_mask.png  2025  1312  0.08
# 4  instance_4_mask.png  2579  1182  0.08



# image_width = mask.shape
# print(f"The width of the image is: {image_width} pixels")


plate_size_mm = 150
plate_size_pixels = 2840
conversion_factor = plate_size_mm / plate_size_pixels



df["X_mm"] = (df["X"] * conversion_factor)/1000
df["Y_mm"] = (df["Y"] * conversion_factor)/1000


plate_position_robot = np.array([0.10775, 0.062, 0.057])


df["X_robot"] = df["X_mm"] + plate_position_robot[0]
df["Y_robot"] = df["Y_mm"] + plate_position_robot[1]

print(df)

#                  Plant     X     Y     Z      X_mm      Y_mm   X_robot   Y_robot
# 0  instance_0_mask.png   519  1442  0.12  0.027412  0.076162  0.135162  0.164162
# 1  instance_1_mask.png   868  1824  0.12  0.045845  0.096338  0.153595  0.184338
# 2  instance_2_mask.png  1473  1573  0.12  0.077799  0.083081  0.185549  0.171081
# 3  instance_3_mask.png  1896  1440  0.12  0.100141  0.076056  0.207891  0.164056
# 4  instance_4_mask.png  2682  1566  0.12  0.141655  0.082711  0.249405  0.170711

goal_positions = df[['X_robot', 'Y_robot', 'Z']].values.tolist()

# Load the trained agent
model = PPO.load(r"C:\Users\domin\Desktop\Year 2 Block B\2023-24b-fai2-adsai-DominikSzewczyk224180\2. Robotics\Y2B-2023-OT2_Twin-main\models\new_best\model.zip")

for goal_pos in goal_positions:
    # Set the goal position for the robot
    env.goal_position = goal_pos

    print(goal_pos)

    # Run the control algorithm until the robot reaches the goal position
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info  = env.step(action)
        # calculate the distance between the pipette and the goal
        print( obs[0:3])
        distance = obs[3:] - obs[:3] # goal position - pipette position
        # calculate the error between the pipette and the goal
        error = np.linalg.norm(distance)
        print(error)
        # Drop the inoculum if the robot is within the required error
        if error < 0.009: # 10mm is used as an example here it is too large for the real use case
            action = np.array([0, 0, 0, 1])
            obs, rewards, terminated, truncated, info  = env.step(action)
            break

        if terminated:
            obs, info = env.reset()