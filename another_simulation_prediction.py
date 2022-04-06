import os
import global_variables as cs
from another_simulation_dataset_creation import dataset_creation
from another_simulation_dataset_load import dataset_load
from plotting_utils import plot_learning_acc_loss, ks_boxplots, data_for_plot
from LSTM_model import LSTM_model
from CNN_LSTM_Conv1D import CNN_LSTM_Conv1D_model
from CNN_LSTM_Conv2D import CNN_LSTM_Conv2D_model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import warnings

warnings.filterwarnings("ignore")

STANDARDIZE = True

if __name__ == '__main__':

    for ts_window in [3, 5]:
        cs.TS_LENGTH = ts_window
        cs.NUMBER_OF_AUGMENTATION = 0
        # NUMBER_OF_AUGMENTATION = round((10000 / ((cs.SAME_SIZE_OF_DF_FROM_SIMULATION - cs.START) / ts_window)) - 1)

        for selected_axis in ['xy', 'xz', 'xyz']:
            if selected_axis == 'xy':
                cs.SELECTED_AXIS = 'xy'
                cs.SELECTED_COLUMNS = cs.xy_reduced
                cs.SELECTED_COLUMNS_TO_STANDARDIZE = cs.xy_reduced_standardize
                cs.SELECTED_COLUMNS_TO_NORMALIZE = cs.xy_reduced_normalize
            if selected_axis == 'xz':
                cs.SELECTED_AXIS = 'xz'
                cs.SELECTED_COLUMNS = cs.xz_reduced
                cs.SELECTED_COLUMNS_TO_STANDARDIZE = cs.xz_reduced_standardize
                cs.SELECTED_COLUMNS_TO_NORMALIZE = cs.xz_reduced_normalize
            if selected_axis == 'xyz':
                cs.SELECTED_AXIS = 'xyz'
                cs.SELECTED_COLUMNS = cs.xyz_reduced
                cs.SELECTED_COLUMNS_TO_STANDARDIZE = cs.xyz_reduced_standardize
                cs.SELECTED_COLUMNS_TO_NORMALIZE = cs.xyz_reduced_normalize
            if selected_axis == 'xy_xz':
                cs.SELECTED_AXIS = 'xy_xz'
                cs.SELECTED_COLUMNS = cs.xy_xz
                cs.SELECTED_COLUMNS_TO_STANDARDIZE = cs.xy_xz_standardize
                cs.SELECTED_COLUMNS_TO_NORMALIZE = cs.xy_xz_normalize

            cs.SAVE_PATH = f'data/dataset/as_W_{cs.TS_LENGTH}_A_0_X_{cs.SELECTED_AXIS}'
            if not os.path.exists(cs.SAVE_PATH):
                os.makedirs(cs.SAVE_PATH)

            # print(cs.TS_LENGTH)
            # dataset_creation()

            cs.SAVE_OUT = f'output/dataset/as_W_{cs.TS_LENGTH}_A_0_X_{cs.SELECTED_AXIS}'
            if not os.path.exists(cs.SAVE_OUT):
                os.makedirs(cs.SAVE_OUT)

            X_, y_, X_CNN = dataset_load()

            for mode, label, data in zip([LSTM_model, CNN_LSTM_Conv1D_model, CNN_LSTM_Conv2D_model],
                                         ['LSTM', 'CNN-LSTM_Conv1D', 'CNN-LSTM_Conv2D'],
                                         [[X_, y_], [X_, y_], [X_CNN, y_]]):
                try:
                    loss_f = tf.keras.losses.MeanAbsolutePercentageError()
                    model = CNN_LSTM_Conv2D_model(learning_rate=1e-4, input_shape=X_CNN[0].shape, loss_f=loss_f)

                    import re
                    import os

                    path = "output/dataset"
                    dirs = os.listdir(path)
                    for file in dirs:
                        phoneNumRegex = re.compile(f'output/dataset/W_{cs.TS_LENGTH}_A_\d+_X_{cs.SELECTED_AXIS}'
                                                   f'/weights_{label}_{cs.TS_LENGTH}.h5')
                        if phoneNumRegex.search(file):
                            model.load_weights(file)

                    score = model.evaluate(data[0], data[1], verbose=1)
                    print(label, cs.TS_LENGTH, cs.SELECTED_AXIS)
                    print()
                    print(f'{score}')
                    print()
                    print()
                except Exception as e:
                    print(label)
                    print(e)
                    print()
