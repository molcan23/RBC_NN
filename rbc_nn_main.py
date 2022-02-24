import os
import numpy as np
import pandas as pd
import global_variables as cs
from dataset_creation import dataset_creation
from dataset_load import dataset_load
from plotting_utils import plot_learning_acc_loss, ks_boxplots, data_for_plot, plot_confusion_matrix
from LSTM_model import LSTM_model
from CNN_LSTM_Conv1D import CNN_LSTM_Conv1D_model
from CNN_LSTM_Conv2D import CNN_LSTM_Conv2D_model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay, confusion_matrix


STANDARDIZE = True

if __name__ == '__main__':

    for ts_window in [5, 10, 20, 30, 40, 50]:
        cs.TS_LENGTH = ts_window
        cs.NUMBER_OF_AUGMENTATION = round((8000 / (2200 / ts_window)) - 1)

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

            cs.SAVE_PATH = f'data/dataset/W_{cs.TS_LENGTH}_A_{cs.NUMBER_OF_AUGMENTATION}_X_{cs.SELECTED_AXIS}'
            if not os.path.exists(cs.SAVE_PATH):
                os.makedirs(cs.SAVE_PATH)

            print(cs.TS_LENGTH)
            dataset_creation()

            cs.SAVE_OUT = f'output/dataset/W_{cs.TS_LENGTH}_A_{cs.NUMBER_OF_AUGMENTATION}_X_{cs.SELECTED_AXIS}'
            if not os.path.exists(cs.SAVE_OUT):
                os.makedirs(cs.SAVE_OUT)

            X_train, X_test, y_train, y_test, X_train_CNN, X_test_CNN, y_train_CNN, y_test_CNN = dataset_load()

            for model, label, data in zip([LSTM_model, CNN_LSTM_Conv1D_model, CNN_LSTM_Conv2D_model],
                                          ['LSTM', 'CNN-LSTM_Conv1D', 'CNN-LSTM_Conv2D'],
                                          [[X_train, X_test, y_train, y_test],
                                           [X_train, X_test, y_train, y_test],
                                           [X_train_CNN, X_test_CNN, y_train_CNN, y_test_CNN]]):
                _X_train, _X_test, _y_train, _y_test = data
                loss_f = tf.keras.losses.MeanAbsolutePercentageError()

                model_1 = model(number_of_classes=9, learning_rate=1e-4, input_shape=_X_train[0].shape)

                es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
                rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
                mcp = ModelCheckpoint(filepath=f'{cs.SAVE_OUT}/weights_{label}_{cs.TS_LENGTH}.h5', monitor='val_loss',
                                      verbose=1, save_best_only=True, save_weights_only=True)

                history_1 = model_1.fit(_X_train, _y_train, shuffle=True, epochs=cs.EPOCHS, callbacks=[es, rlr, mcp],
                                        validation_split=0.2, verbose=1, batch_size=256)

                plot_learning_acc_loss(history_1, f"{label}_LF_{cs.LOSS_FN}_W_{cs.TS_LENGTH}"
                                                  f"_A_{cs.NUMBER_OF_AUGMENTATION}_SC_{cs.SELECTED_AXIS}")

                print('\n\n')
                score = model_1.evaluate(_X_test, _y_test, verbose=1)
                print(f'{score}')

                out_file = open(f'{cs.SAVE_OUT}/statistics.txt', "a")
                out_file.write(f"{label} \t {cs.TS_LENGTH} \t {score[0]} \t {score[1]}\n")
                out_file.close()

                out_file = open(f'data/{label}statistics.txt', "a")
                out_file.write(f"{label} \t {cs.TS_LENGTH} \t {score[0]} \t {score[1]}\n")
                out_file.close()

                # predictions
                predictions_test = model_1.predict(_X_test)

                np.save(f'{cs.SAVE_OUT}/y_test.txt', np.array(_y_test))
                np.save(f'{cs.SAVE_OUT}/y_test_predicted.txt', np.array(predictions_test))
                # data, dir_labels = data_for_plot(predictions_test, _y_test)
                # ks_boxplots(data,
                #             f"{label}_LF_{cs.LOSS_FN}_W_{cs.TS_LENGTH}_A_{cs.NUMBER_OF_AUGMENTATION}_SC_{cs.SELECTED_AXIS}_error",
                #             dir_labels,
                #             outliers=False)

                # training "predictions"
                predictions_train = model_1.predict(_X_train)

                np.save(f'{cs.SAVE_OUT}/y_train.txt', np.array(_y_train))
                np.save(f'{cs.SAVE_OUT}/y_train_predicted.txt', np.array(predictions_train))

                Y = y_test.idxmax(axis=1).astype(str)
                P = pd.DataFrame(predictions_train, columns=y_test.columns).idxmax(axis=1).astype(str)
                cm = confusion_matrix(Y, P)
                plot_confusion_matrix(cm,
                                      cs.k_s,
                                      title=f'Confusion matrix',
                                      cmap=None,
                                      normalize=True)

                # data, dir_labels, dir_predictions = data_for_plot(predictions_train, _y_test)
                # ks_boxplots(data,
                #             f"{label}_LF_{cs.LOSS_FN}_W_{cs.TS_LENGTH}_A_{cs.NUMBER_OF_AUGMENTATION}_SC_{cs.SELECTED_AXIS}_error",
                #             dir_labels,
                #             dir_predictions,
                #             outliers=False)
