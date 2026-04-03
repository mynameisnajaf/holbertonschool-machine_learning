#!/usr/bin/env python3
"""A module that does the trick"""
import tensorflow.keras as keras


def lenet5(X):
    """LeNet5 model"""
    init = keras.initializers.he_normal()
    activation = keras.activations.relu
    conv1 = keras.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding="same",
        activation=activation,
        initializer=init,
    )(X)

    pool1 = keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=2
    )(conv1)

    conv2 = keras.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding="valid",
        activation=activation,
        initializer=init,
    )(X)

    pool2 = keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=2
    )(conv2)

    flatten = keras.layers.Flatten()(pool2)

    FC1 = keras.layers.Dense(
        units=120,
        activation=activation,
        kernel_initializer=init
    )(flatten)

    FC2 = keras.layers.Dense(
        units=84,
        activation=activation,
        kernel_initializer=init
    )(FC1)

    FC3 = keras.layers.Dense(
        units=10,
        activation="softmax",
        kernel_initializer=init
    )(FC2)

    model = keras.models.Model(inputs=X, outputs=FC3)
    adam = keras.optimizers.Adam()
    model.compile(
        optimizer=adam,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
