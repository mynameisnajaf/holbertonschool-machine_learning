#!/usr/bin/env python3
"""A model that does the trick"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Build the model"""
    model = K.Sequential()

    for i in range(len(layers)):
        if i == 0:
            model.add(
                K.layers.Dense(
                    units=layers[i],
                    activation=activations[i],
                    input_shape=(nx,),
                    kernel_regularizer=K.regularizers.l2(lambtha)
                )
            )
        else:
            model.add(
                K.layers.Dense(
                    units=layers[i],
                    activation=activations[i],
                    kernel_regularizer=K.regularizers.l2(lambtha)
                )
            )

        if i != len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
