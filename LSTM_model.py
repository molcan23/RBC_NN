from plotting_utils import r2_keras

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Flatten


def LSTM_model(learning_rate=1e-4, input_shape=None, loss_f=None):
    model = Sequential([
        LSTM(
            units=256,
            return_sequences=True,
            input_shape=input_shape,
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None),
            bias_initializer='zeros'
        ),
        Dropout(0.1),
        LSTM(units=64, return_sequences=True),
        Dropout(0.1),
        LSTM(units=32, return_sequences=True),
        Dropout(0.1),
        LSTM(units=10, return_sequences=True),
        Flatten(),
        Dropout(0.1),
        Dense(512),
        Dropout(0.1),
        Dense(256),
        Dense(1)
    ])

    model.compile(
      loss=loss_f,
      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
      metrics=[r2_keras]
    )
    return model
