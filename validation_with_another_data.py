import global_variables as cs
import os
import re
import numpy as np
import pandas as pd
from tensorflow import keras
# import keras
import tensorflow as tf
from CNN_LSTM_Conv2D import CNN_LSTM_Conv2D_model
from CNN_LSTM_Conv1D import CNN_LSTM_Conv1D_model
from LSTM_model import LSTM_model
from boxplots_correct import *
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#import pandas as pd
pd.options.mode.chained_assignment = None


def standardize_columns(df, cols):
    for c in cols:
        df[c] = (df[c] - df[c].mean(axis=0)) / df[c].std(axis=0)

    return df


def create_training_examples(df, target, ts_length=3):
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


def dataset_creation(mod=None):
    #print(cs.TS_LENGTH)
    df_all = pd.DataFrame()
    rbc_coefficients = []
    number_of_cells = 0


    for simulation, coef in zip(["sim_5xKSin1_Aa_hct10_seed02_1/sim_5xKSin1_Aa_hct10_seed02_1"],
                                [[.005, .01, .03, .06, .3]]):

        full_path = f"data/{simulation}"
        only_cell_files = sorted([f for f in os.listdir(full_path) if re.match("rbc[0-9]+_.+.dat", f)])
        number_of_cells += len(only_cell_files)
        #print(number_of_cells)

        for i, file_ in enumerate(only_cell_files):
            if re.match(f"rbc*", file_):
                #print(file_)
                df = pd.read_table(f"{full_path}/{file_}", sep=" ", names=cs.head[1:]).drop(['NaN'], axis=1) \
                    .drop([0], axis=0)[200:700]
                df = df.astype('float32')
                df_all = pd.concat([df_all, df], ignore_index=True)
                # print(111)
                a = file_.split("_")[0].split("c")[1]
                m = int(a) // 21
                rbc_coefficients.append(coef[m])
                # print(m)
    """
    for simulation, coef in zip(["control_9_DIFF_SEED_2000"],
                                [[.005, .009, .015, .03, .05, .1, .15, .225, .3, ]]):

        full_path = f"data/{simulation}"
        only_cell_files = sorted([f for f in os.listdir(full_path) if re.match("rbc[0-9]+_.+.dat", f)])
        number_of_cells += len(only_cell_files)
        print(number_of_cells)

        for i, file_ in enumerate(only_cell_files):
            if re.match(f"rbc*", file_):
                print(file_)
                df = pd.read_table(f"{full_path}/{file_}", sep=" ", names=cs.head[1:]).drop(['NaN'], axis=1) \
                    .drop([0], axis=0)[200:700]
                df = df.astype('float32')
                df_all = pd.concat([df_all, df], ignore_index=True)
                # print(111)
                a = file_.split("_")[0].split("c")[1]
                m = int(a) // 6
                rbc_coefficients.append(coef[m])
                # print(m)
    """

    #print(pd.Series(rbc_coefficients).unique())
    #exit()
    # calculation of RBC height and width
    for ax1 in ['x', 'y', 'z']:
        for ax2 in ['x', 'y', 'z']:
            df_all[f'{ax1}_{ax2}_size'] = df_all[f'rbc_cuboid_{ax1}_max_{ax2}'] - df_all[f'rbc_cuboid_{ax1}_min_{ax2}']

    if cs.STANDARDIZE:
        df_all = standardize_columns(df_all, cols=cs.SELECTED_COLUMNS_TO_STANDARDIZE)

    #print(rbc_coefficients)
    #exit()
    training_data = []
    target_data = []
    for i in range(0, number_of_cells):  # 52  # 48
        #print(i, "/", number_of_cells)
        # print(i // 21)
        #print(rbc_coefficients[i])
        trd, tad = create_training_examples(
            df_all[i * 500:(i + 1) * 500],
            rbc_coefficients[i])

        training_data += trd
        target_data += tad

    training_data = np.array(training_data)
    target_data = np.array(target_data)

    training_data_CNN = np.reshape(training_data,
                                   [training_data.shape[0],
                                    training_data.shape[1],
                                    training_data.shape[2],
                                    1])

    tr = training_data if mod == "LSTM" else training_data_CNN

    return tr, np.array(target_data)


if __name__ == '__main__':
    results_table = []

    for variables in [[3, 31], [5, 52], [10, 105], [20, 212], [30, 318], [40, 425], [50, 531]]:
        s, aug = variables

        for model_type in [#f'weights_CNN-LSTM_Conv1D_{s}.h5',
                           #f'weights_CNN-LSTM_Conv2D_{s}.h5',
                           f'weights_LSTM_{s}.h5']:
            loss_f = tf.keras.losses.MeanAbsolutePercentageError()

            if model_type == f'weights_LSTM_{s}.h5':
                X, y = dataset_creation(mod="LSTM")
                model = LSTM_model(learning_rate=1e-4, input_shape=X[0].shape, loss_f=loss_f)
            elif model_type == f'weights_CNN-LSTM_Conv1D_{s}.h5':
                X, y = dataset_creation(mod="LSTM")
                model = CNN_LSTM_Conv1D_model(learning_rate=1e-4, input_shape=X[0].shape, loss_f=loss_f)
            else:
                X, y = dataset_creation()
                model = CNN_LSTM_Conv2D_model(learning_rate=1e-4, input_shape=X[0].shape, loss_f=loss_f)

            #try:
            model.load_weights(f"output/dataset/W_{s}_A_{aug}_X_xy_xz/{model_type}")
            print(model.summary)
            # model = tf.keras.models.load_model()
            pred = model.predict(X).reshape((-1,))
            a = pd.DataFrame([pred, y]).T
            #print(y_CNN)
            a.columns = ['pred', 'y_true']
            #print(a['y_true'].unique())

            a['diff'] = abs(a['pred'] - a['y_true'])
            a['percentage_error'] = a['diff'] / a['y_true'] * 100
            a.to_csv(f'a_w_{s}.csv')
            #print(a)
            #print(y_CNN.shape)
            score = model.evaluate(X, y, verbose=1)
            print(f'{score}')
            # bez 0.6 a priebeh chyby

            a = pd.read_csv(f'a_w_{s}.csv')
            #print(a['y_true'].unique())
            save = 'test'
            f = 'test'

            data, dir_labels, dir_prediction = data_for_plot(a['pred'], a['y_true'])
            if not os.path.exists(f'{save}/{f}_window_{s}'):
                os.makedirs(f'{save}/{f}_window_{s}')
            ks_boxplots(data, f'{save}/{f}_window_{s}', dir_labels, dir_prediction, False)
            results_table.append([s, model_type[9:-6], score[0]])

#            except Exception as e:
#                print(e)
#                print(f"output/dataset/W_{s}_A_{aug}_X_xy_xz/{model_type}")
#                print()

    results_table = pd.DataFrame(results_table)
    results_table.columns = ['window', 'Model type', 'MAPE']

    sns.lineplot(data=results_table, x='window', y='MAPE', hue='Model type', style='Model type',
                 markers=True, dashes=False, linestyle="dashed", palette="flare")  # , legend=False)
    plt.title("Validation of models for xy-xz data")

    plt.savefig(f'plots/model_validation_comparison.png')
    plt.close()

    print(results_table)
