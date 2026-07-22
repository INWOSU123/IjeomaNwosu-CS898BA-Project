"""
cnn_model.py

Defines a compact custom CNN for binary school bus classification.
The model is trained from scratch without pretrained networks.
"""

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)


def build_cnn_model(
    input_shape=(128, 128, 3),
    learning_rate=0.0005,
):
    """
    Build and compile a compact CNN.

    Parameters
    ----------
    input_shape : tuple
        Input image dimensions.

    learning_rate : float
        Adam optimizer learning rate.

    Returns
    -------
    tensorflow.keras.Model
        Compiled binary classification model.
    """

    model = Sequential(
        [
            Input(shape=input_shape),

            Conv2D(
                16,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            ),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.20),

            Conv2D(
                32,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            ),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            Conv2D(
                64,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            ),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.30),

            GlobalAveragePooling2D(),

            Dense(
                32,
                activation="relu",
            ),
            Dropout(0.40),

            Dense(
                1,
                activation="sigmoid",
            ),
        ],
        name="CompactSchoolBusCNN",
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    cnn_model = build_cnn_model()
    cnn_model.summary()