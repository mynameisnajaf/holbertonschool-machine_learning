#!/usr/bin/env python3
"""Variational Autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder

    Args:
        input_dims: integer containing dimensions of model input
        hidden_layers: list containing number of nodes for each hidden layer
        latent_dims: integer containing dimensions of latent space

    Returns:
        encoder, decoder, auto
    """

    # Encoder
    encoder_inputs = keras.Input(shape=(input_dims,))

    x = encoder_inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(
            units=nodes,
            activation='relu'
        )(x)

    z_mean = keras.layers.Dense(
        units=latent_dims,
        activation=None
    )(x)

    z_log_var = keras.layers.Dense(
        units=latent_dims,
        activation=None
    )(x)

    def sampling(args):
        """Reparameterization trick"""
        mean, log_var = args

        epsilon = keras.backend.random_normal(
            shape=keras.backend.shape(mean)
        )

        return mean + keras.backend.exp(log_var / 2) * epsilon

    z = keras.layers.Lambda(
        sampling,
        output_shape=(latent_dims,)
    )([z_mean, z_log_var])

    encoder = keras.Model(
        encoder_inputs,
        [z, z_mean, z_log_var]
    )

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))

    x = decoder_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(
            units=nodes,
            activation='relu'
        )(x)

    decoder_outputs = keras.layers.Dense(
        units=input_dims,
        activation='sigmoid'
    )(x)

    decoder = keras.Model(
        decoder_inputs,
        decoder_outputs
    )

    # Autoencoder
    z, z_mean, z_log_var = encoder(encoder_inputs)
    outputs = decoder(z)

    auto = keras.Model(
        encoder_inputs,
        outputs
    )

    # KL divergence loss
    kl_loss = -0.5 * keras.backend.sum(
        1 + z_log_var
        - keras.backend.square(z_mean)
        - keras.backend.exp(z_log_var),
        axis=-1
    )

    auto.add_loss(keras.backend.mean(kl_loss))

    auto.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    return encoder, decoder, auto
