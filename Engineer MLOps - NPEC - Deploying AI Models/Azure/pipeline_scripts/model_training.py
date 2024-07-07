import argparse
import json
import math
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, ResNet101
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dropout,
                                     Input, Lambda, UpSampling2D, concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

try:
    from image_data_generator import train_generator, val_generator
    from metric import F1IoUMetric
    from model_architectures import resnet50, resnet101
except ImportError:
    from .image_data_generator import train_generator, val_generator
    from .metric import F1IoUMetric
    from .model_architectures import resnet50, resnet101

def train(depth_sel: int, train_path: str) -> None:
    """
    Train the U-Net model using the selected ResNet architecture as an encoder.

    Args:
        depth_sel (int): Choice between ResNet50 (0) or ResNet101 (1).
        train_path (str, optional): Custom path for training data.

    Returns:
        None: This function trains the model and saves the trained model and metrics.

    Author: Benjamin Graziadei
    """
    train_gen = train_generator(dict(horizontal_flip=True), 8, train_path)
    val_gen = val_generator(8, train_path)

    im, ma = next(train_gen)
    if ma.shape[-1] == 1:
        num_classes = 1  # Only one foreground class
        activation = "sigmoid"
        loss = "binary_crossentropy"

    else:
        num_classes = ma.shape[-1]
        activation = "softmax"
        loss = "categorical_crossentropy"

    if depth_sel == 0:
        model = resnet50(
            input_size=(256, 256, 1),
            num_classes=num_classes,
            final_activation=activation,
        )
    elif depth_sel == 1:
        model = resnet101(
            input_size=(256, 256, 1),
            num_classes=num_classes,
            final_activation=activation,
        )
    optimizer = Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer, loss=loss, metrics=[F1IoUMetric(num_classes=num_classes)]
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        verbose=1,
        patience=10,
        restore_best_weights=False,
    )
    history = model.fit(
        train_gen,
        steps_per_epoch=math.ceil(16900 / 8) / 100,
        epochs=1,
        validation_data=val_gen,
        validation_steps=math.ceil(2197 / 8) / 100,
        callbacks=[early_stopping],
        verbose=1,
    )

    return model, history