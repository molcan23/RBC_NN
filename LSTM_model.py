from plotting_utils import r2_keras
import global_variables as cs

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten


def LSTM_model(learning_rate=1e-4, input_shape=None, loss_f=None):
    model = Sequential([
        LSTM(
            units=cs.N_NODES,
            return_sequences=True,
            input_shape=input_shape,
            # kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None),
            bias_initializer='zeros'
        ),
        Dropout(cs.DROPOUT_RATE),
        LSTM(units=64, return_sequences=True),
        Dropout(cs.DROPOUT_RATE),
        LSTM(units=32, return_sequences=True),
        Dropout(cs.DROPOUT_RATE),
        LSTM(units=10, return_sequences=True),
        Flatten(),
        Dropout(cs.DROPOUT_RATE),
        Dense(cs.N_NODES * 2),
        Dropout(cs.DROPOUT_RATE),
        Dense(cs.N_NODES),
        Dense(1)
    ])

    model.compile(
      loss=loss_f,
      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
      metrics=[r2_keras]
    )
    return model
