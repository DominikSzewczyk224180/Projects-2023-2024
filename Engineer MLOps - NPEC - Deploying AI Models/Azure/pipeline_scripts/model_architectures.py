import logging

import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, Dropout, Input,
                                     MaxPooling2D, concatenate, Lambda, UpSampling2D, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, ResNet101
from typing import Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def build_unet_model(
    IMG_HEIGHT: int, IMG_WIDTH: int, IMG_CHANNELS: int
) -> tf.keras.Model:
    """
    Create a U-Net model architecture with specified hyperparameters.

    Parameters:
    IMG_HEIGHT (int): Height of the input images.
    IMG_WIDTH (int): Width of the input images.
    IMG_CHANNELS (int): Number of channels in the input images.

    Returns:
    Model: U-Net model architecture.
    """

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    # Contraction path
    c1 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(
        256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(
        256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c5)

    # Expansive path
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c9)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    logging.info("Model architecture successfully built")

    return model

def resnet50(
    input_size: Tuple[int, int, int], num_classes: int, final_activation: str
) -> Model:
    """
    Build a U-Net model using the ResNet50 architecture as the encoder.

    Args:
        input_size (Tuple[int, int, int]): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes.
        final_activation (str): Activation function for the output layer, e.g., 'sigmoid' or 'softmax'.

    Returns:
        Model: A Keras Model instance with the U-Net architecture.

    Author: Benjamin Graziadei
    """
    input_tensor = Input(shape=input_size)
    x = Lambda(lambda x: tf.tile(x, [1, 1, 1, 3]))(
        input_tensor
    )  # Convert grayscale to 3-channel

    base_model = ResNet50(weights="imagenet", include_top=False, input_tensor=x)
    conv4 = base_model.get_layer("conv4_block6_out").output
    conv3 = base_model.get_layer("conv3_block4_out").output
    conv2 = base_model.get_layer("conv2_block3_out").output
    conv1 = base_model.get_layer("conv1_relu").output

    # Building the decoder part of the U-Net
    up4 = UpSampling2D((2, 2))(conv4)
    up4 = concatenate([up4, conv3])
    up4 = Conv2D(256, (3, 3), activation="relu", padding="same")(up4)
    up4 = BatchNormalization()(up4)
    up4 = Dropout(0.2)(up4)

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([up3, conv2])
    up3 = Conv2D(128, (3, 3), activation="relu", padding="same")(up3)
    up3 = BatchNormalization()(up3)
    up3 = Dropout(0.2)(up3)

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([up2, conv1])
    up2 = Conv2D(64, (3, 3), activation="relu", padding="same")(up2)
    up2 = BatchNormalization()(up2)
    up2 = Dropout(0.2)(up2)

    up1 = UpSampling2D((2, 2))(up2)
    up1 = Conv2D(32, (3, 3), activation="relu", padding="same")(up1)
    up1 = BatchNormalization()(up1)
    up1 = Dropout(0.2)(up1)

    outputs = Conv2D(num_classes, (1, 1), activation=final_activation)(up1)
    model = Model(inputs=input_tensor, outputs=outputs)
    return model


def resnet101(
    input_size: Tuple[int, int, int], num_classes: int, final_activation: str
) -> Model:
    """
    Build a U-Net model using the ResNet101 architecture as the encoder.

    Args:
        input_size (Tuple[int, int, int]): Shape of the input images.
        num_classes (int): Number of output classes.
        final_activation (str): Activation function for the output layer.

    Returns:
        Model: A Keras Model instance with the U-Net architecture using ResNet101.

    Author: Benjamin Graziadei
    """
    input_tensor = Input(shape=input_size)
    x = Lambda(lambda x: tf.tile(x, [1, 1, 1, 3]))(input_tensor)

    base_model = ResNet101(weights="imagenet", include_top=False, input_tensor=x)
    conv4 = base_model.get_layer("conv4_block23_out").output
    conv3 = base_model.get_layer("conv3_block4_out").output
    conv2 = base_model.get_layer("conv2_block3_out").output
    conv1 = base_model.get_layer("conv1_relu").output

    up4 = UpSampling2D((2, 2))(conv4)
    up4 = concatenate([up4, conv3])
    up4 = Conv2D(256, (3, 3), activation="relu", padding="same")(up4)
    up4 = BatchNormalization()(up4)
    up4 = Dropout(0.2)(up4)

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([up3, conv2])
    up3 = Conv2D(128, (3, 3), activation="relu", padding="same")(up3)
    up3 = BatchNormalization()(up3)
    up3 = Dropout(0.2)(up3)

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([up2, conv1])
    up2 = Conv2D(64, (3, 3), activation="relu", padding="same")(up2)
    up2 = BatchNormalization()(up2)
    up2 = Dropout(0.2)(up2)

    up1 = UpSampling2D((2, 2))(up2)
    up1 = Conv2D(32, (3, 3), activation="relu", padding="same")(up1)
    up1 = BatchNormalization()(up1)
    up1 = Dropout(0.2)(up1)

    outputs = Conv2D(num_classes, (1, 1), activation=final_activation)(up1)
    model = Model(inputs=input_tensor, outputs=outputs)
    return model