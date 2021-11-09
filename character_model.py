from tensorflow.keras import layers

from settings import *

import tensorflow as tf
import pickle


def build_model():
    pickle_in = open(EKUSH_DIR + "male/chars/x.pickle", "rb")
    x = pickle.load(pickle_in)

    pickle_in = open(EKUSH_DIR + "male/chars/y.pickle", "rb")
    y = pickle.load(pickle_in)

    model = tf.keras.Sequential()  # Initialise model

    # Input layer & convolutions
    model.add(layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

    model.add(layers.Dropout(0.3))

    # Second set of convolutions with batch normalisation
    model.add(layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

    model.add(layers.Dropout(0.3))

    # Third layer of convolutions
    model.add(layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

    model.add(layers.Dropout(0.3))

    # Final layer of convolutions
    model.add(layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation='relu'))

    model.add(layers.Dropout(0.3))

    # Output layer
    model.add(layers.Dense(50, activation='softmax'))  # 50 characters for ekush

    # Compile model
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Fit data to model and save
    model.fit(x, y, batch_size=32, epochs=20, validation_split=0.3)
    tf.keras.models.save_model(model, "models/char_model.h5")


if __name__ == "__main__":
    build_model()

