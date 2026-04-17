#!/usr/bin/env python3
"""A module that does the trick"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """The identity block"""
    init = K.initializers.he_normal()
    F11, F3, F12 = filters

    conv1 = K.layers.Conv2D(filters=F11,
                            kernel_size=1,
                            activation='relu',
                            kernel_initializer=init,
                            padding='same')(A_prev)
    batch1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(batch1)

    conv2 = K.layers.Conv2D(filters=F3,
                            kernel_size=3,
                            kernel_initializer=init,
                            padding='same')(act1)
    batch2 = K.layers.BatchNormalization(axis=3)(conv2)
    act2 = K.layers.Activation('relu')(batch2)

    conv3 = K.layers.Conv2D(filters=F12,
                            kernel_size=1,
                            padding='same',
                            kernel_initializer=init)(act2)
    batch3 = K.layers.BatchNormalization(axis=3)(conv3)
    act3 = K.layers.Activation('relu')(batch3)

    add = K.layers.Add()([batch3, A_prev])
    f_act = K.layers.Activation('relu')(add)

    return f_act
