import global_variables as cs
from plotting_utils import r2_keras

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten, TimeDistributed, Activation, Conv2D, MaxPooling2D,\
    BatchNormalization, Dropout


def CNN_LSTM_Conv2D_model_my(learning_rate=1e-4, input_shape=None, loss_f=None):
    model = Sequential([
        Conv2D(filters=256, kernel_size=(len(cs.SELECTED_COLUMNS), 3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),

        Conv2D(filters=256, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        MaxPooling2D(pool_size=2),
        TimeDistributed(Flatten()),
        LSTM(cs.LSTM_NODES, activation="relu", return_sequences=False),

        Dense(cs.N_NODES, activation="relu"),
        Dense(1, activation="linear"),
    ])

    model.compile(
      loss=loss_f,
      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
      metrics=[r2_keras]
    )

    return model
