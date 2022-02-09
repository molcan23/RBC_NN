import global_variables as cs

import os
import re
import numpy as np
import pandas as pd


def standardize_columns(df, cols):
    for c in cols:
        df[c] = (df[c] - df[c].mean(axis=0)) / df[c].std(axis=0)

    return df


def augmentation(x, gaussian_noise_level=.001, offset_noise_level=.5):
    noise = gaussian_noise_level * np.random.normal(size=x.shape)
    offset_noise = 2. * np.random.uniform(size=x.shape) - 1.0
    x_result = np.array(x) + noise + offset_noise_level * offset_noise
    return x_result


def create_training_examples(df, target, ts_length=10, selected_columns=None, number_of_augmentation=10):
    """
    Split df into timeseries with length=ts_length and add target value to each mini-timeseries
    """
    training_data = []
    target_data = []

    for i in range(ts_length, df.shape[0], ts_length):
        sample = df.iloc[i - ts_length:i, :]

        for field in cs.SELECTED_COLUMNS_TO_NORMALIZE:
            mini = sample[field].min()
            sample[field] = sample[field] - mini

        # sample = tf.convert_to_tensor(sample[cs.SELECTED_COLUMNS])
        sample = sample[cs.SELECTED_COLUMNS]
        training_data.append(sample)
        target_data.append(target)

        for _ in range(number_of_augmentation):
            training_data.append(augmentation(sample))
            target_data.append(target)

    return training_data, target_data


def dataset_creation():
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
                     .drop([0], axis=0)[cs.START:cs.SAME_SIZE_OF_DF_FROM_SIMULATION]
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
        df_all

    dataset_path = f'data/dataset/W_{cs.TS_LENGTH}_A_{cs.NUMBER_OF_AUGMENTATION}_X_{cs.SELECTED_AXIS}'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    for i in range(0, number_of_cells):  # 52  # 48
        print(i, "/", 54)
        trd, tad = create_training_examples(
            df_all[i * (cs.SAME_SIZE_OF_DF_FROM_SIMULATION - cs.START):(i + 1) * (cs.SAME_SIZE_OF_DF_FROM_SIMULATION - cs.START)],
            rbc_coefficients[i],
            cs.TS_LENGTH,
            selected_columns=cs.SELECTED_COLUMNS,
            number_of_augmentation=cs.NUMBER_OF_AUGMENTATION)
        data_out = f'{dataset_path}/data_time_series_W_{cs.TS_LENGTH}_A_{cs.NUMBER_OF_AUGMENTATION}' \
                   f'_X_{cs.SELECTED_AXIS}_rbc_{str(i)}.npy'
        label_out = f'{dataset_path}/data_labels_W_{cs.TS_LENGTH}_A_{cs.NUMBER_OF_AUGMENTATION}' \
                    f'_X_{cs.SELECTED_AXIS}_rbc_{str(i)}.npy'

        np.save(data_out, np.array(trd))
        np.save(label_out, np.array(tad))

        # training_data1 = training_data1 + trd
        # target_data1 = target_data1 + tad

dataset_creation()
