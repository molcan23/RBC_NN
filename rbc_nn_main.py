from global_variables import *
from dataset_load import dataset_load
from plotting_utils import plot_learning_acc_loss, ks_boxplots, data_for_plot
from LSTM_model import LSTM_model
from CNN_LSTM_Conv1D import CNN_LSTM_Conv1D_model
from CNN_LSTM_Conv2D import CNN_LSTM_Conv2D_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

X_train, X_test, y_train, y_test, X_train_CNN, X_test_CNN, y_train_CNN, y_test_CNN = dataset_load()

STANDARDIZE = True

if __name__ == '__main__':

    for ts_window in [10, 20, 30, 40, 50]:
        TS_LENGTH = ts_window
        NUMBER_OF_AUGMENTATION = round((8000 / (2200 / ts_window)) - 1)
        for selected_axis in ['xy', 'xz', 'xyz']:
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

                for model, label, data in zip([LSTM_model, CNN_LSTM_Conv1D_model, CNN_LSTM_Conv2D_model],
                                        ['LSTM', 'CNN-LSTM_Conv1D', 'CNN-LSTM_Conv2D'],
                                        [[X_train, X_test, y_train, y_test],
                                         [X_train, X_test, y_train, y_test],
                                         [X_train_CNN, X_test_CNN, y_train_CNN, y_test_CNN]]):

                    _X_train, _X_test, _y_train, _y_test = data
                    model_1 = model(learning_rate=1e-4, input_shape=_X_train[0].shape)

                    es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
                    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
                    mcp = ModelCheckpoint(filepath=f'{SAVE_PATH}/weights_{label}_{TS_LENGTH}.h5', monitor='val_loss',
                                          verbose=1, save_best_only=True, save_weights_only=True)

                    history_1 = model_1.fit(_X_train, _y_train, shuffle=True, epochs=EPOCHS, callbacks=[es, rlr, mcp],
                                            validation_split=0.2, verbose=1, batch_size=256)

                    plot_learning_acc_loss(history_1,  f"{label}_LF_{LOSS_FN}_W_{TS_LENGTH}"
                                                       f"_A_{NUMBER_OF_AUGMENTATION}_SC_{SELECTED_AXIS}")

                    print('\n\n')
                    score = model_1.evaluate(_X_test, _y_test, verbose=1)
                    print(f'{score}')

                    out_file = open(f'{SAVE_PATH}/statistics.txt', "a")
                    out_file.write(f"{label} \t {score}")
                    out_file.close()

                    predictions_future = model_1.predict(_X_test)

                    data, dir_labels = data_for_plot(predictions_future, _y_test)
                    ks_boxplots(data,
                                f"LSTM-v0_LF_{LOSS_FN}_W_{TS_LENGTH}_A_{NUMBER_OF_AUGMENTATION}_SC_{SELECTED_AXIS}_error",
                                dir_labels,
                                outliers=False)
