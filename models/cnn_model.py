"""
cnn_model.py

Defines the custom Convolutional Neural Network (CNN)
used for binary school bus classification.

Author: Ijeoma Nwosu
Course: CS898BA – Image Analysis and Computer Vision
"""

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense
)


IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
CHANNELS = 3


def build_cnn_model():

    model = Sequential(name="SchoolBusCNN")

    # -------------------------------------------------
    # Block 1
    # -------------------------------------------------

    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS),
        )
    )

    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # -------------------------------------------------
    # Block 2
    # -------------------------------------------------

    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
        )
    )

    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # -------------------------------------------------
    # Block 3
    # -------------------------------------------------

    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
        )
    )

    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # -------------------------------------------------
    # Classification Head
    # -------------------------------------------------

    model.add(Flatten())

    model.add(
        Dense(
            128,
            activation="relu",
        )
    )

    model.add(Dropout(0.50))

    model.add(
        Dense(
            1,
            activation="sigmoid",
        )
    )

    # -------------------------------------------------
    # Compile Model
    # -------------------------------------------------

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":

    cnn = build_cnn_model()

    cnn.summary()