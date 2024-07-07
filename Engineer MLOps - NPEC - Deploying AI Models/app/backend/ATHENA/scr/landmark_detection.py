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
from typing import Tuple, List

def extract_skeleton_segment(skeleton: np.ndarray, endpoint: Tuple[int, int], length: int = 20) -> np.ndarray:
    """
    Extracts a segment of the skeleton around a given endpoint.

    Parameters:
    - skeleton (np.ndarray): The skeleton image.
    - endpoint (Tuple[int, int]): The coordinates of the endpoint.
    - length (int): The length of the segment to extract. Default is 20.

    Returns:
    - np.ndarray: The extracted segment.

    Author: Benjamin Graziadei
    """
    x, y = endpoint
    padded_skeleton = np.pad(skeleton, pad_width=length, mode="constant", constant_values=0)
    x_pad, y_pad = x + length, y + length
    segment = padded_skeleton[x_pad - length : x_pad + length + 1, y_pad - length : y_pad + length + 1]
    return segment

def get_endpoints(
    skeleton_data: Skeleton, 
    skeleton_summary: np.ndarray, 
    graph: np.ndarray, 
    labels: np.ndarray, 
    x_coords: np.ndarray, 
    y_coords: np.ndarray
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Extracts endpoints from the skeleton data.

    Parameters:
    - skeleton_data (Skeleton): The skeleton data.
    - skeleton_summary (np.ndarray): Summary of the skeleton.
    - graph (np.ndarray): Graph representation of the skeleton.
    - labels (np.ndarray): Labels of the connected components.
    - x_coords (np.ndarray): X coordinates of the skeleton points.
    - y_coords (np.ndarray): Y coordinates of the skeleton points.

    Returns:
    - Tuple[List[Tuple[int, int]], List[int]]: A list of endpoints and their corresponding labels.

    Author: Benjamin Graziadei
    """
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

def remove_long_connections(mst: np.ndarray, points: np.ndarray, threshold: float) -> np.ndarray:
    """
    Removes connections in the minimum spanning tree that exceed a certain threshold.

    Parameters:
    - mst (np.ndarray): The minimum spanning tree matrix.
    - points (np.ndarray): Array of points in the skeleton.
    - threshold (float): The distance threshold for removing connections.

    Returns:
    - np.ndarray: The modified minimum spanning tree matrix.

    Author: Benjamin Graziadei
    """
    new_mst = np.zeros_like(mst)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if mst[i, j] > 0 and mst[i, j] < threshold:
                new_mst[i, j] = mst[i, j]
                new_mst[j, i] = mst[i, j]
    return new_mst

def group_and_fit_branches(
    skeleton: np.ndarray, 
    output_image: np.ndarray, 
    degree: int = 3, 
    distance_threshold: float = 60, 
    num: int = 500
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Groups and fits branches in the skeleton.

    Parameters:
    - skeleton (np.ndarray): The skeleton image.
    - output_image (np.ndarray): The output image to draw on.
    - degree (int): Degree parameter for branch fitting. Default is 3.
    - distance_threshold (float): Distance threshold for removing long connections. Default is 60.
    - num (int): Number of iterations for fitting. Default is 500.

    Returns:
    - Tuple[np.ndarray, List[Tuple[int, int]]]: The modified skeleton and a list of endpoints.

    Author: Benjamin Graziadei
    """
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

def get_plants(mask_path: str) -> None:
    """
    Processes the mask image to detect and annotate plant skeletons.

    Parameters:
    - mask_path (str): The file path to the mask image.

    Returns:
    - None

    Author: Benjamin Graziadei
    """
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

def Landmark_detecion(path: str) -> None:
    """
    Detects landmarks in mask images within a specified directory.

    Parameters:
    - path (str): The directory path containing the mask images.

    Returns:
    - None

    Author: Benjamin Graziadei
    """
    for file_name in os.listdir(path):
        if file_name.startswith("INDIV_mask1"):
            get_plants(os.path.join(path, file_name))
    return print("Landmark detection completed.")

