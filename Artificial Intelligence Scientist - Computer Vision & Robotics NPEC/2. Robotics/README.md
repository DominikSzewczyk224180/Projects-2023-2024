# Opentrons OT-2 Robot Simulation

## Environment Setup

### Dependencies
Before running the Opentrons OT-2 simulation, ensure you have the following Python packages installed:

1. **PyBullet**: The primary tool for our simulation environment.
   - Install PyBullet by running the following command in your terminal or command prompt:
     ```bash
     pip install pybullet
     ```

2. **Other Packages**:
   - [cv2](https://pypi.org/project/opencv-python/): OpenCV library for computer vision.
   - [matplotlib](https://matplotlib.org/stable/users/installing.html): Python plotting library.
   - [stable_baselines3](https://github.com/DLR-RM/stable-baselines3): Reinforcement learning library.
   - [ot2_gym_wrapper](https://github.com/your-username/ot2_gym_wrapper): Custom wrapper for the Opentrons OT-2 environment.
   - [numpy](https://numpy.org/): Fundamental package for scientific computing with Python.
   - [pandas](https://pandas.pydata.org/): Data manipulation and analysis library.
   - [scikit-image](https://scikit-image.org/): Image processing library.
   - [skan](https://github.com/jni/skan): Skeleton analysis library.
   - [keras](https://keras.io/): High-level neural networks API.

### Working Envelope of the Pipette

#### What is a Pipette?
A pipette is a virtual tool on the Opentrons OT-2, resembling a robotic arm, used for precise liquid handling.

#### What is a Working Envelope?
The working envelope is the safe and effective space where the pipette can move within the simulation environment.

#### Determining the Working Envelope
1. Visualize an imaginary cube within the robot's play area.
2. Move the pipette to each corner of this cube.
3. Record the coordinates (x, y, z) for each corner to define the working envelope.

**Working Envelope Coordinates:**
- Corner 1: [0.180, 0.130, 0.200]
- Corner 2: [-0.260, 0.130, 0.200]
- Corner 3: [-0.260, -0.260, 0.200]
- Corner 4: [0.180, -0.260, 0.200]
- Corner 5: [0.180, 0.130, 0.080]
- Corner 6: [-0.260, 0.130, 0.080]
- Corner 7: [-0.260, -0.260, 0.080]
- Corner 8: [0.180, -0.260, 0.080]


## Instructions for Environment Setup

1. Install the required dependencies listed above.
2. Clone the `ot2_gym_wrapper` repository from [https://github.com/BredaUniversityADSAI/Y2B-2023-OT2_Twin](https://github.com/BredaUniversityADSAI/Y2B-2023-OT2_Twin).
3. Set up your virtual environment and activate it.

## Model Hyperparameters

The following hyperparameters were used for training the model:

- Learning Rate: 0.001
- Batch Size: 64
- Number of Steps: 2048
- Number of Epochs: 10

These hyperparameters yielded the best performance for the trained model.

### Other Hyperparameter Configurations (Worst Performers)

While experimenting with different hyperparameter configurations, the following settings resulted in worse performance:

1. Learning Rate: 0.0001, Batch Size: 64, Number of Steps: 2048, Number of Epochs: 10
2. Learning Rate: 0.003, Batch Size: 64, Number of Steps: 1024, Number of Epochs: 10
3. Learning Rate: 0.003, Batch Size: 32, Number of Steps: 1024, Number of Epochs: 15