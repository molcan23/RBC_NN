import global_variables as cs
from plotting_utils import r2_keras

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten, TimeDistributed, Activation, Conv2D, MaxPooling2D, BatchNormalization


def CNN_LSTM_Conv2D_model(number_of_classes=0, learning_rate=1e-4, input_shape=None):
    model = Sequential([
        Conv2D(filters=64, kernel_size=(len(cs.SELECTED_COLUMNS), 3), padding='same', activation="relu", input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(filters=64, kernel_size=3, padding='same', activation="relu"),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=2),
        TimeDistributed(Flatten()),
        LSTM(cs.LSTM_NODES, activation="relu", return_sequences=False),
        Dense(cs.N_NODES, activation="relu"),
        Dense(units=number_of_classes, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=['accuracy']
    )

    return model
