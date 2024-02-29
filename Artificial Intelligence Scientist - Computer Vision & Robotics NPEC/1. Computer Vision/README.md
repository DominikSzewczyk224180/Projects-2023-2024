# Computer Vision Project README

This repository contains the code and deliverables for a computer vision project comprising several tasks aimed at analyzing images of plants. Each task contributes to a comprehensive pipeline for various analyses and measurements related to plant morphology.

## Task 2: Region of Interest (ROI) Extraction

Identify and isolate the Petri dish from raw images using traditional computer vision methods. The extracted ROI will serve as the basis for subsequent analyses, allowing for focused processing on the relevant area.

## Task 3: Instance Segmentation (Traditional CV)

Perform instance segmentation to identify individual plants within the images. This involves classifying and delineating each plant at the pixel level using traditional computer vision techniques, enabling further analysis on individual plant instances.

## Task 4: Semantic Segmentation (Deep Learning)

Train a deep learning model to perform semantic segmentation on images, distinguishing between different parts of the plant such as roots, seeds, shoots, and occluded roots. This task utilizes deep learning techniques to achieve accurate pixel-level segmentation of plant organs.

## Task 5: Instance Segmentation

Apply instance segmentation to the output of the semantic segmentation model to differentiate between individual instances of plants. This task builds upon Task 4 by further refining the segmentation to identify and delineate each plant instance accurately.

## Task 6: Landmark Detection

Detect specific points of interest within the images, including the primary root tip, root-hypocotyl junction, and lateral root tips. This task involves identifying key landmarks using computer vision techniques, enabling further analysis and measurements related to plant morphology.

## Task 7: Morphometric Analysis

Quantify various morphometric measurements, such as primary root length and total lateral root length, from the images. This task involves extracting quantitative measurements from the images to analyze and compare the geometries of different plant structures.

## Task 8: Kaggle Competition

Develop a comprehensive computer vision pipeline to predict primary root lengths for a given dataset of plant images. This task integrates the skills and knowledge acquired from previous tasks to build an effective pipeline for plant morphology analysis, culminating in a Kaggle competition to evaluate the pipeline's performance.
