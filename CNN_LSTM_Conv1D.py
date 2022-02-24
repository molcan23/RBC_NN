import global_variables as cs
from plotting_utils import r2_keras

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten, TimeDistributed, Activation,\
    Conv1D, MaxPooling1D, BatchNormalization


def CNN_LSTM_Conv1D_model(number_of_classes=0, learning_rate=1e-4, input_shape=None):
    model = Sequential([
        Conv1D(filters=64, kernel_size=len(cs.SELECTED_COLUMNS), padding='same', activation="relu", input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        Dropout(cs.DROPOUT_RATE),
        Conv1D(filters=64, kernel_size=len(cs.SELECTED_COLUMNS), padding='same', activation="relu"),   # TODO tu bolo Conv1D nechtiac, ale ide
        BatchNormalization(),
        Activation('relu'),
        Dropout(cs.DROPOUT_RATE),
        MaxPooling1D(pool_size=2),
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
