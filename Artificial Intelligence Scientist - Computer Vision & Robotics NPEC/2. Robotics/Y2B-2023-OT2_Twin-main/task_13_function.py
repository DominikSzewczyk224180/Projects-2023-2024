import cv2
import numpy as np
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
from keras.optimizers import Adam
from skan import Skeleton, summarize
from skimage.morphology import remove_small_objects, skeletonize


# from stable_baselines3 import PPO
# from ot2_gym_wrapper import OT2Env
#????


def padder(image, patch_size):
    h = image.shape[0]
    w = image.shape[1]
    height_padding = ((h // patch_size) + 1) * patch_size - h
    width_padding = ((w // patch_size) + 1) * patch_size - w

    top_padding = int(height_padding/2)
    bottom_padding = height_padding - top_padding

    left_padding = int(width_padding/2)
    right_padding = width_padding - left_padding

    padded_image = cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded_image

def create_and_save_patches(dataset_type, patch_size, scaling_factor,mask_type):

    for image_path in glob.glob(f'Images for model/{mask_type}/{dataset_type}_images/{dataset_type}/*.png'):
        mask_suffix = f'_{mask_type}_mask.tif'
        path = f'Images for model/{mask_type}'
        mask_path = image_path.replace('images', 'masks').replace('.png', mask_suffix)

        image = cv2.imread(image_path)
        image = padder(image, patch_size)
        if scaling_factor != 1:
            image = cv2.resize(image, (0,0), fx=scaling_factor, fy=scaling_factor)
        patches = patchify(image, (patch_size, patch_size, 1), step=patch_size)
        patches = patches.reshape(-1, patch_size, patch_size, 1)

        image_patch_path = image_path.replace(path, patch_dir)
        for i, patch in enumerate(patches):
            image_patch_path_numbered = f'{image_patch_path[:-4]}_{i}.png'
            cv2.imwrite(image_patch_path_numbered, patch)

        mask_path = image_path.replace('images', 'masks').replace('.png', mask_suffix)
        mask = cv2.imread(mask_path, 0)
        mask = padder(mask, patch_size)
        if scaling_factor != 1:
            mask = cv2.resize(mask, (0,0), fx=scaling_factor, fy=scaling_factor)
        patches = patchify(mask, (patch_size, patch_size), step=patch_size)
        patches = patches.reshape(-1, patch_size, patch_size, 1)

        mask_patch_path = mask_path.replace(path, patch_dir)
        for i, patch in enumerate(patches):
            mask_patch_path_numbered = f'{mask_patch_path[:-4]}_{i}.png'
            cv2.imwrite(mask_patch_path_numbered, patch)


# Let's implement two custom metrics f1 score and iou
def f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = TP / (Positives+K.epsilon())
        return recall
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = TP / (Pred_Positives+K.epsilon())
        return precision
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        y_pred = tf.cast(y_pred>0.5,y_pred.dtype)
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
        total = K.sum(K.square(y_true),[1,2,3]) + K.sum(K.square(y_pred),[1,2,3])
        union = total - intersection
        return (intersection + K.epsilon()) / (union + K.epsilon())
    return K.mean(f(y_true, y_pred), axis=-1)


def save_detected_instances_with_crop(original_image, mask, save_path):
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
        
        # Crop corresponding regions from the original image
        instance_crop = original_image[y:y+h, x:x+w]
        
        instance_img_filename = f"instance_{instance_id}.png"
        instance_img_save_path = os.path.join(save_path, instance_img_filename)
        cv2.imwrite(instance_img_save_path, instance_crop)
        
        # Save the instance's mask 
        instance_mask_filename = f"instance_{instance_id}_mask.png"
        instance_mask_save_path = os.path.join(save_path, instance_mask_filename)
        cv2.imwrite(instance_mask_save_path, combined_mask[y:y+h, x:x+w])
        
        instance_id += 1


def get_last_node_coordinates(skeleton_data, node_id):
    # Filter data based on the provided node_id
    filtered_data = skeleton_data[skeleton_data['node-id-src'] == node_id]
    
    if filtered_data.empty:
        print("No data found for the specified node ID.")
        return None
    
    # Extract coordinates
    x_coord = filtered_data['image-coord-src-1'].values[0]
    y_coord = filtered_data['image-coord-src-0'].values[0]

    return x_coord, y_coord

def romove_objects_from_mask(mask):
    kernel = np.ones((5, 5), dtype="uint8")
    mask_d = cv2.dilate(mask, kernel, iterations=1)
    mask_closing = cv2.erode(mask_d, kernel, iterations=1)

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_closing)

    
    filtered_components = []
    for i in range(1, retval):
        centroid_x, centroid_y = centroids[i]
        if centroid_y < 1800 and centroid_y > 450:
            filtered_components.append(i)

    
    filtered_mask = np.zeros_like(mask_closing)

    for label in filtered_components:
        filtered_mask[labels == label] = 255
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(filtered_mask)

    area_values = stats[1:, cv2.CC_STAT_AREA]
    mean_area = np.mean(area_values)

    min_size = mean_area/5
    alternative = remove_small_objects(labels, min_size = min_size)

    return alternative>0

def crop_image(image):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th, output_im = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(output_im)
 
    stats = stats[1:]
    for stat in stats:
        if stat[2] > 2000:
            plants = stat
            break
   
    x, y, w, h, area = plants
 
    if w > h:
        l = h
    else:
        l = w
        
    return y, x, l

def get_plants(image,model):

    patch_size = 128
   
    y, x, l = crop_image(image)
    cropped_image = image[y:y+l, x:x+l]

    # Padder the image 
    padded_cut_image = padder(cropped_image, patch_size)
    
    patches = patchify(padded_cut_image, (patch_size, patch_size, 1), step=patch_size)
    
    i = patches.shape[0]
    j = patches.shape[1]
        
    patches = patches.reshape(-1, patch_size, patch_size, 1)
    patches.shape
    
    preds = model.predict(patches/255)
    
    preds = preds.reshape(i, j, 128, 128)
    
    predicted_mask = unpatchify(preds, (padded_cut_image.shape[0], padded_cut_image.shape[1]))

    predicted_mask = predicted_mask > 0.5
    predicted_mask = predicted_mask.astype(np.uint8) * 255

    

    # Clean the mask and convert it to BGR

    skeleton = skeletonize(predicted_mask)

    cleaned_skeleton = remove_small_objects(skeleton, min_size=1000, connectivity=2)
    # mask = romove_objects_from_mask(predicted_mask)


    # mask = mask.astype(np.uint8) * 255
    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # # Make a binary copy of the mask
    # _, binary_image = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
    # # Get the skeleton
    # skeleton = skeletonize(binary_image)
    # Get the skeleton data
    skeleton_data = summarize(Skeleton(cleaned_skeleton))
    # Sort by the skeleton ID
    skeleton_data = skeleton_data.sort_values(by='skeleton-id', ascending=True)

    # A list of all the skeleton id's
    plants = np.unique(skeleton_data['skeleton-id'])

    # Create a copy of the image
    plant_img = padded_cut_image

    return plants, plant_img,  skeleton_data


def extract_root_coordinates(plants, skeleton_data, plant_img):
    root_coordinates_list = []

    for plant_num, plant in enumerate(plants):
        # Get a new dataframe for the individual skeleton/plant data
        plant_data = skeleton_data[skeleton_data['skeleton-id'] == plant]

        # Get the start and end point of the main root
        end_main_root = plant_data['node-id-dst'].max()

        # Save the end point coordinates for the primary root
        x_end = plant_data[plant_data['node-id-dst'] == end_main_root]['coord-dst-0'].values[0]
        y_end = plant_data[plant_data['node-id-dst'] == end_main_root]['coord-dst-1'].values[0]

        # Append data to the list
        root_coordinates_list.append({
            'Plant': f'plant_{plant_num + 1}',
            'X': int(x_end),
            'Y': int(y_end),
            'Z': 0.08  # You can adjust the Z value as needed
        })

    # Create a DataFrame from the list
    root_coordinates = pd.DataFrame(root_coordinates_list)

    return root_coordinates



# Example usage:
# plants, skeleton_data, plant_img are assumed to be defined


# # U-Net model
# # Author: Sreenivas Bhattiprolu
# # This code is coming from the videos at the beginning
# from keras.models import Model
# import keras.backend as K
# from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda

# def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, lr):
# # Build the model
#     inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
#     s = inputs

#     # Contraction path
#     c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
#     c1 = Dropout(0.1)(c1)
#     c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
#     p1 = MaxPooling2D((2, 2))(c1)
    
#     c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
#     c2 = Dropout(0.1)(c2)
#     c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
#     p2 = MaxPooling2D((2, 2))(c2)
     
#     c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
#     c3 = Dropout(0.2)(c3)
#     c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
#     p3 = MaxPooling2D((2, 2))(c3)
     
#     c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
#     c4 = Dropout(0.2)(c4)
#     c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
#     p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
#     c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
#     c5 = Dropout(0.3)(c5)
#     c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
#     # Expansive path 
#     u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
#     u6 = concatenate([u6, c4])
#     c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
#     c6 = Dropout(0.2)(c6)
#     c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
#     u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
#     u7 = concatenate([u7, c3])
#     c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
#     c7 = Dropout(0.2)(c7)
#     c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
#     u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
#     u8 = concatenate([u8, c2])
#     c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
#     c8 = Dropout(0.1)(c8)
#     c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
#     u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
#     u9 = concatenate([u9, c1], axis=3)
#     c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
#     c9 = Dropout(0.1)(c9)
#     c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
#     outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
     
#     model = Model(inputs=[inputs], outputs=[outputs])
#     optimizer = Adam(lr=lr) 
#     model.compile(optimizer=optimizer,
#                   loss='binary_crossentropy',
#                   metrics=['accuracy', f1, iou])
#     model.summary()
    
#     return model


# patch_size = 128
# scaling_factor = 1

# mask_type = 'root'


# patch_dir = '2. Robotics/Y2B-2023-OT2_Twin-main/Images for model/root_patched'
# for subdir in ['train_images/train', 'train_masks/train', 'val_images/val', 'val_masks/val','test_images/test','test_masks/test']:
#     os.makedirs(os.path.join(patch_dir, subdir), exist_ok=True)

# print("Current Working Directory:", os.getcwd())

# print("Train Images:", len(os.listdir(os.path.join(patch_dir, 'train_images/train'))))
# print("Train Masks:", len(os.listdir(os.path.join(patch_dir, 'train_masks/train'))))
# print("Validation Images:", len(os.listdir(os.path.join(patch_dir, 'val_images/val'))))
# print("Validation Masks:", len(os.listdir(os.path.join(patch_dir, 'val_masks/val'))))



# create_and_save_patches('train', patch_size, scaling_factor, mask_type= mask_type)
# create_and_save_patches('val', patch_size, scaling_factor, mask_type= mask_type)

# # Training images
# train_image_datagen = ImageDataGenerator(rescale=1./255)

# train_image_generator = train_image_datagen.flow_from_directory(
#     f'{patch_dir}/train_images',
#     target_size=(patch_size, patch_size),
#     batch_size=16,
#     class_mode=None,
#     color_mode='grayscale',
#     seed=42)

# # Training masks
# train_mask_datagen = ImageDataGenerator()

# train_mask_generator = train_mask_datagen.flow_from_directory(
#     f'{patch_dir}/train_masks',
#     target_size=(patch_size, patch_size),
#     batch_size=16,
#     class_mode=None,
#     color_mode='grayscale',
#     seed=42)

# train_generator = zip(train_image_generator, train_mask_generator)

# # val images
# val_image_datagen = ImageDataGenerator(rescale=1./255)

# val_image_generator = val_image_datagen.flow_from_directory(
#     f'{patch_dir}/val_images',
#     target_size=(patch_size, patch_size),
#     batch_size=16,
#     class_mode=None,
#     color_mode='grayscale',
#     seed=42
# )

# # val masks
# val_mask_datagen = ImageDataGenerator()

# val_mask_generator = val_mask_datagen.flow_from_directory(
#     f'{patch_dir}/val_masks',
#     target_size=(patch_size, patch_size),
#     batch_size=16,
#     class_mode=None,
#     color_mode='grayscale',
#     seed=42
# )

# val_generator = zip(val_image_generator, val_mask_generator)


# model_root = simple_unet_model(patch_size, patch_size, 1, lr=0.001)


# from keras.callbacks import EarlyStopping

# cb = EarlyStopping(monitor='val_iou',
#                    patience=3   ,
#                    restore_best_weights='True',
#                    mode='max')

# h_root = model_root.fit(
#     train_generator,
#     steps_per_epoch=len(train_image_generator),
#     epochs=20,
#     validation_data = val_generator,
#     validation_steps = val_image_generator.samples//16,
#     callbacks=[cb]
#     )


# model_root.save('trained_model.h5')


