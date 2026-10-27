"""
model.py
========
Convolutional Neural Network (CNN) architecture for Dog-Cat Image Classification.

Technical Specifications:
- Exactly THREE convolutional layers:
  1. Conv2D (32 filters, 3x3 kernel, ReLU) + MaxPooling2D (2x2)
  2. Conv2D (64 filters, 3x3 kernel, ReLU) + MaxPooling2D (2x2)
  3. Conv2D (128 filters, 3x3 kernel, ReLU) + MaxPooling2D (2x2)
- Flatten layer
- Dense layer (128 neurons, ReLU)
- Dropout layer (rate = 0.5)
- Output layer (1 neuron, Sigmoid)
- Loss: Binary Cross-Entropy
- Optimizer: Adam
- Metric: Accuracy
- Input Dimensions: 128 x 128 x 3
- Pixel Normalization: Scaled to [0, 1]
"""

import io
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_model(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    learning_rate: float = 0.001,
    include_rescaling: bool = True
) -> tf.keras.Model:
    """
    Constructs and compiles the 3-layer Convolutional Neural Network for Dog-Cat classification.

    Parameters:
        input_shape (tuple): Shape of the input images (height, width, channels). Default is (128, 128, 3).
        learning_rate (float): Learning rate for the Adam optimizer. Default is 0.001.
        include_rescaling (bool): Whether to include an internal Rescaling(1./255) layer for [0, 1] normalization.
                                  Default is True.

    Returns:
        tf.keras.Model: Compiled Keras Sequential model.
    """
    model = models.Sequential(name="DogCat_3Layer_CNN")

    # Input Layer & Rescaling to [0, 1] range
    model.add(layers.Input(shape=input_shape, name="input_image"))
    if include_rescaling:
        model.add(layers.Rescaling(1.0 / 255.0, name="rescaling_norm_0_to_1"))

    # ==========================================
    # CONVOLUTIONAL BLOCK 1 (32 Filters)
    # ==========================================
    model.add(layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="conv2d_layer_1"
    ))
    model.add(layers.MaxPooling2D(
        pool_size=(2, 2),
        name="maxpooling2d_1"
    ))

    # ==========================================
    # CONVOLUTIONAL BLOCK 2 (64 Filters)
    # ==========================================
    model.add(layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="conv2d_layer_2"
    ))
    model.add(layers.MaxPooling2D(
        pool_size=(2, 2),
        name="maxpooling2d_2"
    ))

    # ==========================================
    # CONVOLUTIONAL BLOCK 3 (128 Filters)
    # ==========================================
    model.add(layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="conv2d_layer_3"
    ))
    model.add(layers.MaxPooling2D(
        pool_size=(2, 2),
        name="maxpooling2d_3"
    ))

    # ==========================================
    # CLASSIFIER HEAD
    # ==========================================
    model.add(layers.Flatten(name="flatten"))
    model.add(layers.Dense(
        units=128,
        activation="relu",
        name="dense_128_relu"
    ))
    model.add(layers.Dropout(
        rate=0.5,
        name="dropout_0.5"
    ))
    model.add(layers.Dense(
        units=1,
        activation="sigmoid",
        name="output_sigmoid"
    ))

    # Compilation with Adam and Binary Cross-Entropy
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def get_model_summary_string(model: tf.keras.Model) -> str:
    """
    Captures model.summary() output as a string for logging and UI display.
    """
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    return stream.getvalue()


if __name__ == "__main__":
    cnn = build_cnn_model()
    print("=" * 60)
    print("Project: 3-Layer CNN for Dog-Cat Image Classification")
    print("=" * 60)
    cnn.summary()
    total_conv_layers = sum(1 for layer in cnn.layers if isinstance(layer, layers.Conv2D))
    print(f"\nVerification: Total Conv2D layers = {total_conv_layers} (Requirement: Exactly 3)")
    assert total_conv_layers == 3, "Model must have exactly 3 convolutional layers!"
    print("Architecture verified successfully.")
