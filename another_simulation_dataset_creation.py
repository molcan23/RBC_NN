import global_variables as cs
import os
import re
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def standardize_columns(df, cols):
    for c in cols:
        df[c] = (df[c] - df[c].mean(axis=0)) / df[c].std(axis=0)

    return df


def create_training_examples(df, target, ts_length=10):
    """
    Split df into timeseries with length=ts_length and add target value to each mini-timeseries
    """
    training_data = []
    target_data = []
    print(target)
    for i in range(ts_length, df.shape[0], ts_length):
        sample = df.iloc[i - ts_length:i, :]

        for field in cs.SELECTED_COLUMNS_TO_NORMALIZE:
            mini = sample[field].min()
            sample[field] = sample[field] - mini

        sample = sample[cs.SELECTED_COLUMNS]
        training_data.append(sample)
        target_data.append(target)

    return training_data, target_data


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def dataset_creation():
    print(cs.TS_LENGTH)
    df_all = pd.DataFrame()
    rbc_coefficients = []
    number_of_cells = 0
    cs.SAME_SIZE_OF_DF_FROM_SIMULATION = 1700

    for simulation, coef in zip(['sim_9xKSin1_Aa_hct10_seed02_1'],
                                [[.005, .009, .015, .03, .05, .1, .15, .225, .3]]):

        full_path = f"data/{simulation}"
        only_cell_files = sorted([f for f in os.listdir(full_path) if re.match("rbc[0-9]+_.+.dat", f)])
        number_of_cells += len(only_cell_files)

        for i, file_ in enumerate(only_cell_files):
            # print(file_)
            df = pd.read_table(f"{full_path}/{file_}", sep=" ", names=cs.head[1:]).drop(['NaN'], axis=1) \
                .drop([0], axis=0)[cs.START:cs.SAME_SIZE_OF_DF_FROM_SIMULATION]
            df = df.astype('float32')
            df_all = pd.concat([df_all, df], ignore_index=True)
            if re.match(f"rbc*", file_):
                a = file_.split("_")[0].split("c")[1]
                m = int(a) // 6
                rbc_coefficients.append(coef[m])
                # print(m)

    # calculation of RBC height and width
    for ax1 in ['x', 'y', 'z']:
        for ax2 in ['x', 'y', 'z']:
            df_all[f'{ax1}_{ax2}_size'] = df_all[f'rbc_cuboid_{ax1}_max_{ax2}'] - df_all[f'rbc_cuboid_{ax1}_min_{ax2}']

    if cs.STANDARDIZE:
        df_all = standardize_columns(df_all, cols=cs.SELECTED_COLUMNS_TO_STANDARDIZE)

    dataset_path = f'data/dataset/as_W_{cs.TS_LENGTH}_A_{cs.NUMBER_OF_AUGMENTATION}_X_{cs.SELECTED_AXIS}'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    for i in range(0, number_of_cells):  # 52  # 48
        # print(i, "/", 54)
        trd, tad = create_training_examples(
            df_all[i * (cs.SAME_SIZE_OF_DF_FROM_SIMULATION - cs.START):(i + 1) *
                                                                       (cs.SAME_SIZE_OF_DF_FROM_SIMULATION - cs.START)],
            rbc_coefficients[i],
            cs.TS_LENGTH)
        data_out = f'{dataset_path}/data_as_W_{cs.TS_LENGTH}_A_{cs.NUMBER_OF_AUGMENTATION}' \
                   f'_X_{cs.SELECTED_AXIS}_rbc_{str(i)}.npy'
        label_out = f'{dataset_path}/labels_as_W_{cs.TS_LENGTH}_A_{cs.NUMBER_OF_AUGMENTATION}' \
                    f'_X_{cs.SELECTED_AXIS}_rbc_{str(i)}.npy'

        trd, tad = unison_shuffled_copies(np.array(trd), np.array(tad))
        np.save(data_out, trd)
        np.save(label_out, tad)


if __name__ == '__main__':
    cs.TS_LENGTH = 5
    cs.NUMBER_OF_AUGMENTATION = 0
    dataset_creation()
