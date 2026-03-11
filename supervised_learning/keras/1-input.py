#!/usr/bin/env python3
"""A model that does the trick"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Build the model"""
    inputs = K.layers.Input(shape=(nx,))
    x = inputs

    for i in range(len(layers)):
        x = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)

        if i != len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    model = K.models.Model(inputs=inputs, outputs=x)

    return model
