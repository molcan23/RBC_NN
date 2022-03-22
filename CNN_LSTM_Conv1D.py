import global_variables as cs
from plotting_utils import r2_keras

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten, TimeDistributed, Activation,\
    Conv1D, MaxPooling1D, BatchNormalization


def CNN_LSTM_Conv1D_model(number_of_classes=0, learning_rate=1e-4,
                          input_shape=None, loss_f=None):
    model = Sequential([
        Conv1D(filters=256, kernel_size=3, padding='same', activation="relu", input_shape=input_shape),
        Conv1D(filters=256, kernel_size=3, padding='same', activation="relu"),
        MaxPooling1D(pool_size=2),
        TimeDistributed(Flatten()),
        Dropout(cs.DROPOUT_RATE),
        LSTM(cs.LSTM_NODES, activation="relu", return_sequences=False),
        Dense(cs.N_NODES, activation="relu"),
        Dense(units=number_of_classes, activation='softmax')
    ])

    loss_f = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, label_smoothing=0.0, axis=-1,
        name='categorical_crossentropy'
    )

    model.compile(
        loss=loss_f,
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=['accuracy']
    )

    return model
