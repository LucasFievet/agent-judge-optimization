"""TensorFlow/Keras model for triple MNIST classification."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_triple_mnist_model(input_shape=(28, 84, 1)) -> keras.Model:
    """
    Create a model for triple MNIST classification.
    
    The model:
    1. Splits the 84x28 input into 3 parts (each 28x28)
    2. Applies the same neural network to each part to get 10 logits per digit
    3. Applies a convolutional sum layer to combine the 3x10 logits into 27 logits (sum 0-27)
    
    Args:
        input_shape: Shape of input image (height, width, channels)
    
    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=input_shape)
    
    # Split input into 3 parts: each is 28x28
    # Input is 28x84x1, split along width dimension (axis=2)
    split1 = layers.Lambda(lambda x: x[:, :, 0:28, :])(inputs)
    split2 = layers.Lambda(lambda x: x[:, :, 28:56, :])(inputs)
    split3 = layers.Lambda(lambda x: x[:, :, 56:84, :])(inputs)
    
    # Shared neural network for each digit
    def create_digit_network():
        """Create a network that processes a single 28x28 digit image."""
        model = keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='linear'),  # 10 logits for digits 0-9
        ])
        return model
    
    digit_net = create_digit_network()
    
    # Apply the same network to each split
    logits1 = digit_net(split1)  # Shape: (batch, 10)
    logits2 = digit_net(split2)  # Shape: (batch, 10)
    logits3 = digit_net(split3)  # Shape: (batch, 10)
    
    # Stack the logits: (batch, 3, 10)
    stacked_logits = layers.Concatenate(axis=1)([logits1, logits2, logits3])
    stacked_logits = layers.Reshape((3, 10))(stacked_logits)
    
    # Apply 1D convolution to combine the 3x10 logits into 27 logits
    # This learns how to sum the three digits
    # Use a conv1d with kernel_size=3 to see all three digits at once
    conv_sum = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(stacked_logits)
    conv_sum = layers.GlobalAveragePooling1D()(conv_sum)
    conv_sum = layers.Dense(128, activation='relu')(conv_sum)
    conv_sum = layers.Dropout(0.5)(conv_sum)
    
    # Output layer: 28 classes (sums from 0 to 27)
    outputs = layers.Dense(28, activation='softmax', name='sum_prediction')(conv_sum)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='triple_mnist_model')
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    
    return model

