from global_variables import *
from dataset_creation import dataset_creation
from dataset_load import dataset_load
from plotting_utils import plot_learning_acc_loss, ks_boxplots, data_for_plot
from LSTM_model import LSTM_model
from CNN_LSTM_Conv1D import CNN_LSTM_Conv1D_model
from CNN_LSTM_Conv2D import CNN_LSTM_Conv2D_model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np

STANDARDIZE = True


if __name__ == '__main__':

    for ts_window in [5, 10, 20, 30, 40, 50]:
        TS_LENGTH = ts_window
        NUMBER_OF_AUGMENTATION = round((8000 / (2200 / ts_window)) - 1)

        for selected_axis in ['xy', 'xz']:  # , 'xyz']:
            if selected_axis == 'xy':
                SELECTED_AXIS = 'xy'
                SELECTED_COLUMNS = xy_reduced
                SELECTED_COLUMNS_TO_STANDARDIZE = xy_reduced_standardize
                SELECTED_COLUMNS_TO_NORMALIZE = xy_reduced_normalize
            if selected_axis == 'xz':
                SELECTED_AXIS = 'xz'
                SELECTED_COLUMNS = xz_reduced
                SELECTED_COLUMNS_TO_STANDARDIZE = xz_reduced_standardize
                SELECTED_COLUMNS_TO_NORMALIZE = xz_reduced_normalize
            if selected_axis == 'xyz':
                SELECTED_AXIS = 'xyz'
                SELECTED_COLUMNS = xyz_reduced
                SELECTED_COLUMNS_TO_STANDARDIZE = xyz_reduced_standardize
                SELECTED_COLUMNS_TO_NORMALIZE = xyz_reduced_normalize

            SAVE_PATH = f'data/dataset/W_{TS_LENGTH}_A_{NUMBER_OF_AUGMENTATION}_X_{SELECTED_AXIS}'
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)

            dataset_creation()
                
            SAVE_OUT= f'output/dataset/W_{TS_LENGTH}_A_{NUMBER_OF_AUGMENTATION}_X_{SELECTED_AXIS}'
            if not os.path.exists(SAVE_OUT):
                os.makedirs(SAVE_OUT)

            X_train, X_test, y_train, y_test, X_train_CNN, X_test_CNN, y_train_CNN, y_test_CNN = dataset_load()

            for model, label, data in zip([CNN_LSTM_Conv2D_model],
                                    ['CNN-LSTM_Conv2D'],
                                    [[X_train_CNN, X_test_CNN, y_train_CNN, y_test_CNN]]):

                _X_train, _X_test, _y_train, _y_test = data
                loss_f = tf.keras.losses.MeanAbsolutePercentageError()

                model_1 = model(learning_rate=1e-4, input_shape=_X_train[0].shape, loss_f=loss_f)

                es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
                rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
                mcp = ModelCheckpoint(filepath=f'{SAVE_OUT}/weights_{label}_{TS_LENGTH}.h5', monitor='val_loss',
                                      verbose=1, save_best_only=True, save_weights_only=True)

                history_1 = model_1.fit(_X_train, _y_train, shuffle=True, epochs=EPOCHS, callbacks=[es, rlr, mcp],
                                        validation_split=0.2, verbose=1, batch_size=256)

                plot_learning_acc_loss(history_1,  f"{label}_LF_{LOSS_FN}_W_{TS_LENGTH}"
                                                   f"_A_{NUMBER_OF_AUGMENTATION}_SC_{SELECTED_AXIS}")

                print('\n\n')
                score = model_1.evaluate(_X_test, _y_test, verbose=1)
                print(f'{score}')

                out_file = open(f'{SAVE_OUT}/statistics.txt', "a")
                out_file.write(f"{label} \t {TS_LENGTH} \t {score[0]} \t {score[1]}\n")
                out_file.close()

                out_file = open(f'data/{label}statistics.txt', "a")
                out_file.write(f"{label} \t {TS_LENGTH} \t {score[0]} \t {score[1]}\n")
                out_file.close()

                # predictions
                predictions_test = model_1.predict(_X_test)
                   
                np.save(f'{SAVE_OUT}/y_test.txt', np.array(_y_test))
                np.save(f'{SAVE_OUT}/y_test_predicted.txt', np.array(predictions_test))
                data, dir_labels = data_for_plot(predictions_test, _y_test)
                ks_boxplots(data,
                            f"{label}_LF_{LOSS_FN}_W_{TS_LENGTH}_A_{NUMBER_OF_AUGMENTATION}_SC_{SELECTED_AXIS}_error",
                            dir_labels,
                            outliers=False)

                # training "predictions"
                predictions_train = model_1.predict(_X_train)
                    
                np.save(f'{SAVE_OUT}/y_train.txt', np.array(_y_train))
                np.save(f'{SAVE_OUT}/y_train_predicted.txt', np.array(predictions_train))

                data, dir_labels = data_for_plot(predictions_train, _y_test)
                ks_boxplots(data,
                            f"{label}_LF_{LOSS_FN}_W_{TS_LENGTH}_A_{NUMBER_OF_AUGMENTATION}_SC_{SELECTED_AXIS}_error",
                            dir_labels,
                            outliers=False)
