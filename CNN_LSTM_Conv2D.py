import global_variables as cs
from plotting_utils import r2_keras

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten, TimeDistributed, Activation, Conv2D, MaxPooling2D, BatchNormalization


def CNN_LSTM_Conv2D_model(learning_rate=1e-4, input_shape=None, loss_f=None):
    model = Sequential([
        Conv2D(filters=256, kernel_size=(9, 9), padding='same', activation="relu", input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(filters=256, kernel_size=(4, 3), padding='same', activation="relu"),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=2),
        TimeDistributed(Flatten()),
        Dense(256, activation="relu"),
        LSTM(512, activation="relu", return_sequences=False),
        LSTM(512, activation="relu", return_sequences=False),
        Dense(1024, activation="relu"),
        Dense(1024, activation="relu"),
        Dense(1, activation="linear"),
    ])

    model.compile(
      loss=loss_f,
      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
      metrics=[r2_keras]
    )

    return model
