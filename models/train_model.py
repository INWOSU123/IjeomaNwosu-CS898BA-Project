"""
train_model.py

Trains a compact custom CNN for binary school bus classification.

This experiment uses:
- A smaller CNN architecture
- Data augmentation
- No class weighting
- Early stopping
- Model checkpointing

Author: Ijeoma Nwosu
Course: CS 898BA - Image Analysis and Computer Vision
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from cnn_model import build_cnn_model


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# --------------------------------------------------
# Project Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "train",
)

VALIDATION_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "validation",
)

SAVED_MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "saved_models",
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
)


# --------------------------------------------------
# Compact Model Output Paths
# --------------------------------------------------

MODEL_PATH = os.path.join(
    SAVED_MODEL_DIR,
    "school_bus_cnn_compact.keras",
)

HISTORY_PATH = os.path.join(
    OUTPUT_DIR,
    "compact_training_history.json",
)

ACCURACY_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "compact_accuracy_plot.png",
)

LOSS_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "compact_loss_plot.png",
)


# Create output directories when they do not exist.
os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Training Hyperparameters
# --------------------------------------------------

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 8


# --------------------------------------------------
# Validate Required Directories
# --------------------------------------------------

def validate_directories():
    """Confirm that the training directories exist."""

    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(
            f"Training directory was not found: {TRAIN_DIR}"
        )

    if not os.path.isdir(VALIDATION_DIR):
        raise FileNotFoundError(
            "Validation directory was not found: "
            f"{VALIDATION_DIR}"
        )


# --------------------------------------------------
# Create Data Generators
# --------------------------------------------------

def create_data_generators():
    """
    Create augmented training data and normalized validation data.
    """

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,

        # Small rotations help account for different camera angles.
        rotation_range=15,

        # Shift images horizontally and vertically.
        width_shift_range=0.10,
        height_shift_range=0.10,

        # Slightly zoom images in and out.
        zoom_range=0.10,

        # Slight brightness changes simulate lighting conditions.
        brightness_range=(0.80, 1.20),

        # School buses may face either direction.
        horizontal_flip=True,

        fill_mode="nearest",
    )

    validation_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
        seed=RANDOM_SEED,
    )

    validation_generator = (
        validation_datagen.flow_from_directory(
            VALIDATION_DIR,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="binary",
            shuffle=False,
        )
    )

    return train_generator, validation_generator


# --------------------------------------------------
# Save Training History
# --------------------------------------------------

def save_training_history(history):
    """Save training and validation metrics to a JSON file."""

    serializable_history = {
        metric: [float(value) for value in values]
        for metric, values in history.history.items()
    }

    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8",
    ) as history_file:
        json.dump(
            serializable_history,
            history_file,
            indent=4,
        )


# --------------------------------------------------
# Save Accuracy Plot
# --------------------------------------------------

def save_accuracy_plot(history):
    """Save the training and validation accuracy graph."""

    epoch_numbers = range(
        1,
        len(history.history["accuracy"]) + 1,
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        epoch_numbers,
        history.history["accuracy"],
        label="Training Accuracy",
    )

    plt.plot(
        epoch_numbers,
        history.history["val_accuracy"],
        label="Validation Accuracy",
    )

    plt.title(
        "Compact CNN: Training and Validation Accuracy"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        ACCURACY_PLOT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# --------------------------------------------------
# Save Loss Plot
# --------------------------------------------------

def save_loss_plot(history):
    """Save the training and validation loss graph."""

    epoch_numbers = range(
        1,
        len(history.history["loss"]) + 1,
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        epoch_numbers,
        history.history["loss"],
        label="Training Loss",
    )

    plt.plot(
        epoch_numbers,
        history.history["val_loss"],
        label="Validation Loss",
    )

    plt.title(
        "Compact CNN: Training and Validation Loss"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        LOSS_PLOT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# --------------------------------------------------
# Main Training Function
# --------------------------------------------------

def main():
    """Run the compact-model training process."""

    validate_directories()

    train_generator, validation_generator = (
        create_data_generators()
    )

    print("\nClass labels:")
    print(train_generator.class_indices)

    print("\nTraining image counts:")

    class_counts = np.bincount(
        train_generator.classes
    )

    for class_name, class_id in sorted(
        train_generator.class_indices.items(),
        key=lambda item: item[1],
    ):
        print(
            f"{class_name} ({class_id}): "
            f"{class_counts[class_id]}"
        )

    # Build the compact CNN defined in cnn_model.py.
    model = build_cnn_model(
        input_shape=(128, 128, 3),
        learning_rate=LEARNING_RATE,
    )

    print("\nCompact model summary:")
    model.summary()

    # Save only the model with the lowest validation loss.
    checkpoint = ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    )

    # Stop when validation loss does not improve for eight epochs.
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    print("\nStarting compact CNN training...")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Maximum epochs: {EPOCHS}")
    print("Class weighting: Disabled")

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=[
            checkpoint,
            early_stopping,
        ],
    )

    save_training_history(history)
    save_accuracy_plot(history)
    save_loss_plot(history)

    print("\nCompact-model training complete.")

    print("\nBest compact model saved to:")
    print(MODEL_PATH)

    print("\nTraining history saved to:")
    print(HISTORY_PATH)

    print("\nAccuracy plot saved to:")
    print(ACCURACY_PLOT_PATH)

    print("\nLoss plot saved to:")
    print(LOSS_PLOT_PATH)


if __name__ == "__main__":
    main()