#!/usr/bin/env python3
"""A module that does the trick"""
from tensorflow import keras as K


def lenet5(X):
    """LeNet5 model"""
    activation = K.activations.relu
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding="same",
        activation=activation,
        kernel_initializer=K.initializers.HeNormal(
            seed=0
        )
    )(X)

    pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=2
    )(conv1)

    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding="valid",
        activation=activation,
        kernel_initializer=K.initializers.HeNormal(
            seed=0
        )
    )(pool1)

    pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=2
    )(conv2)

    flatten = K.layers.Flatten()(pool2)

    FC1 = K.layers.Dense(
        units=120,
        activation=activation,
        kernel_initializer=K.initializers.HeNormal(
            seed=0
        )
    )(flatten)

    FC2 = K.layers.Dense(
        units=84,
        activation=activation,
        kernel_initializer=K.initializers.HeNormal(
            seed=0
        )
    )(FC1)

    FC3 = K.layers.Dense(
        units=10,
        activation="softmax",
        kernel_initializer=K.initializers.HeNormal(
            seed=0
        )
    )(FC2)

    model = K.models.Model(inputs=X, outputs=FC3)
    adam = K.optimizers.Adam()
    model.compile(
        optimizer=adam,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
