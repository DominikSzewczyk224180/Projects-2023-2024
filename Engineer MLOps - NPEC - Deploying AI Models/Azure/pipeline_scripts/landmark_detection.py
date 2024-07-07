import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial import distance_matrix
from skan import Skeleton, csr, summarize
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

def extract_skeleton_segment(skeleton, endpoint, length=20):
    x, y = endpoint
    padded_skeleton = np.pad(
        skeleton, pad_width=length, mode="constant", constant_values=0
    )
    x_pad, y_pad = x + length, y + length
    segment = padded_skeleton[
        x_pad - length : x_pad + length + 1, y_pad - length : y_pad + length + 1
    ]
    return segment

def get_endpoints(skeleton_data, skeleton_summary, graph, labels, x_coords, y_coords):
    endpoints = []
    endpoint_labels = []
    for index, row in skeleton_summary.iterrows():
        component_indices = skeleton_data.path(index)
        for node_index in [component_indices[0], component_indices[-1]]:
            degree = np.sum(graph[node_index] > 0)
            if degree == 1:
                endpoints.append((x_coords[node_index], y_coords[node_index]))
                endpoint_labels.append(labels[node_index])
    return endpoints, endpoint_labels

def remove_long_connections(mst, points, threshold):
    new_mst = np.zeros_like(mst)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if mst[i, j] > 0 and mst[i, j] < threshold:
                new_mst[i, j] = mst[i, j]
                new_mst[j, i] = mst[i, j]
    return new_mst

def group_and_fit_branches(skeleton, output_image, degree=3, distance_threshold=60, num=500):
    if not np.any(skeleton):  # Check if the skeleton is empty
        return skeleton, []

    graph, coords = csr.skeleton_to_csgraph(skeleton)
    y_coords, x_coords = coords
    skeleton_data = Skeleton(skeleton > 0)
    skeleton_summary = summarize(skeleton_data)
    n_components, labels = connected_components(graph)
    endpoints, endpoint_labels = get_endpoints(
        skeleton_data, skeleton_summary, graph, labels, x_coords, y_coords
    )

    points = np.array(endpoints)
    dist_matrix = distance_matrix(points, points)
    mst = minimum_spanning_tree(dist_matrix).toarray()
    mst = remove_long_connections(mst, points, distance_threshold)

    used_endpoints = set()
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if mst[i, j] > 0 and endpoint_labels[i] == endpoint_labels[j]:
                mst[i, j] = 0
                mst[j, i] = 0
            elif mst[i, j] > 0:
                cv2.line(
                    output_image,
                    (int(points[i][0]), int(points[i][1])),
                    (int(points[j][0]), int(points[j][1])),
                    (255, 255, 255),
                    1,
                )
                used_endpoints.add((int(points[i][0]), int(points[i][1])))
                used_endpoints.add((int(points[j][0]), int(points[j][1])))
    
    output_image = output_image[..., 0]

    skeleton = skeletonize(output_image > 0)

    labeled_skeleton = label(skeleton)
    regions = regionprops(labeled_skeleton)

    sizes = [region.area for region in regions]
    average_size = np.mean(sizes)
    threshold = 0.5 * average_size

    new_skeleton = np.zeros_like(skeleton)
    for region in regions:
        if region.area >= threshold:
            for coordinates in region.coords:
                new_skeleton[coordinates[0], coordinates[1]] = 1
                
    filtered_endpoints = []
    for ep in endpoints:
        y, x = int(ep[1]), int(ep[0])
        if new_skeleton[y, x] != 0:
            filtered_endpoints.append(ep)

    new_skeleton = new_skeleton.astype(np.uint8) * 255

    return new_skeleton, filtered_endpoints

def get_plants(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    orig_image = cv2.imread(mask_path.replace("INDIV_mask1", "ORG"))

    mask = np.array(mask, dtype=np.uint8)

    skeleton = skeletonize(np.ones((100, 100), dtype=np.uint8))
    skeleton[50, 50] = 0  # Ensure there's at least one foreground pixel

    output_image = np.zeros((100, 100, 3), dtype=np.uint8)

    skeleton, endpoints = group_and_fit_branches(skeleton, output_image)

    for endpoint in endpoints:
        x, y = endpoint
        cv2.putText(orig_image, f"({x}, {y})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(mask_path.replace("INDIV_mask1", "ORG"), orig_image)
    cv2.imwrite(mask_path, skeleton)
    return

def Landmark_detecion(path):
    for file_name in os.listdir(path):
        if file_name.startswith("INDIV_mask1"):
            get_plants(os.path.join(path, file_name))
    return print("Landmark detection completed.")
