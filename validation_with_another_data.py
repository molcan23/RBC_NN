import global_variables as cs
import os
import re
import numpy as np
import pandas as pd
from tensorflow import keras
# import keras
import tensorflow as tf
from CNN_LSTM_Conv2D import CNN_LSTM_Conv2D_model


def standardize_columns(df, cols):
    for c in cols:
        df[c] = (df[c] - df[c].mean(axis=0)) / df[c].std(axis=0)

    return df


def create_training_examples(df, target, ts_length=30):
    """
    Split df into timeseries with length=ts_length and add target value to each mini-timeseries
    """
    training_data = []
    target_data = []

    try:
        for i in range(ts_length, df.shape[0], ts_length):
            sample = df.iloc[i - ts_length:i, :]
   
            for field in cs.SELECTED_COLUMNS_TO_NORMALIZE:
                mini = sample[field].min()
                sample[field] = sample[field] - mini

            # sample = tf.convert_to_tensor(sample[cs.SELECTED_COLUMNS])
            sample = sample[cs.SELECTED_COLUMNS]
            training_data.append(sample)
            target_data.append(target)
    except Exception as e:
         print("Exception", e)

    return training_data, target_data


def dataset_creation():
    print(cs.TS_LENGTH)
    df_all = pd.DataFrame()
    rbc_coefficients = []
    number_of_cells = 0

    for simulation, coef in zip(['three_types', 'another_three_types', 'gap_three'],
                                [[.3, .005, .03],
                                 [.15, .015, .009],
                                 [.225, .1, .05]]):

        full_path = f"data/{simulation}"
        only_cell_files = sorted([f for f in os.listdir(full_path) if re.match("rbc[0-9]+_.+.dat", f)])
        number_of_cells += len(only_cell_files)

        for i, file_ in enumerate(only_cell_files):
            print(file_)
            df = pd.read_table(f"{full_path}/{file_}", sep=" ", names=cs.head[1:]).drop(['NaN'], axis=1) \
                     .drop([0], axis=0)[2700:2900]
            df = df.astype('float32')
            df_all = pd.concat([df_all, df], ignore_index=True)
            if re.match(f"(rbc0_|rbc1_|rbc2_|rbc3_|rbc4_|rbc5_).*", file_):
                rbc_coefficients.append(coef[0])
            elif re.match(f"(rbc6_|rbc7_|rbc8_|rbc9_|rbc10_|rbc11_).*", file_):
                rbc_coefficients.append(coef[1])
            else:
                rbc_coefficients.append(coef[2])

    # calculation of RBC height and width
    for ax1 in ['x', 'y', 'z']:
        for ax2 in ['x', 'y', 'z']:
            df_all[f'{ax1}_{ax2}_size'] = df_all[f'rbc_cuboid_{ax1}_max_{ax2}'] - df_all[f'rbc_cuboid_{ax1}_min_{ax2}']

    if cs.STANDARDIZE:
        df_all = standardize_columns(df_all, cols=cs.SELECTED_COLUMNS_TO_STANDARDIZE)

    for i in range(0, number_of_cells):  # 52  # 48
        print(i, "/", 54)
        trd, tad = create_training_examples(
            df_all[i * 200:(i + 1) * 200],
            rbc_coefficients[i])
        
    training_data = np.array(trd)

    training_data_CNN = np.reshape(training_data,
                                   [training_data.shape[0],
                                    training_data.shape[1],
                                    training_data.shape[2],
                                    1])

    return training_data_CNN, np.array(tad)


if __name__ == '__main__':
    X_CNN, y_CNN = dataset_creation()
    loss_f = tf.keras.losses.MeanAbsolutePercentageError()
    model = CNN_LSTM_Conv2D_model(learning_rate=1e-4, input_shape=X_CNN[0].shape, loss_f=loss_f)
    model.load_weights('output/dataset/W_30_A_108_X_xz/weights_CNN-LSTM_Conv2D_30.h5')
    # model = tf.keras.models.load_model()
    score = model.evaluate(X_CNN, y_CNN, verbose=1)
    print(f'{score}')
