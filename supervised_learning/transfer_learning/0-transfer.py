#!/usr/bin/env python3
"""
python script that trains
a convolutional neural network to classify
the CIFAR 10 datase
"""
from tensorflow import keras as K


def preprocess_data(X, Y):
    """Preprocess data"""
    X_p = K.applications.inception_v3.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)

    return X_p, Y_p

if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    input_K = K.layers.Input(shape=(32, 32, 3))
    input_resize = K.layers.Lambda(
        lambda x: K.backend.resize_images(
            x,
            height_factor=(224 // 32),
            width_factor=(224 // 32),
            data_format="channels_last"
        ),
        output_shape=(224, 224, 3)
    )(input_K)

    InceptionV3 = K.applications.inception_v3.InceptionV3(include_top=False,
                                                          weights='imagenet',
                                                          input_shape=(224, 224, 3)
                                                          )
    activation = K.activations.relu

    X = InceptionV3(input_resize, training=False)
    X = K.layers.GlobalAvgPool2D()(X)
    X = K.layers.Dense(500, activation=activation)(X)
    X = K.layers.Dropout(0.2)(X)
    output_K = K.layers.Dense(10, activation='softmax')(X)
    model = K.models.Model(inputs=input_K, outputs=output_K)

    InceptionV3.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )
    history = model.fit(x=X_train, y=Y_train,
                        validation_data=(X_test, Y_test),
                        batch_size=64,
                        epochs=5, verbose=True)
    model.save('cifar10.h5')
