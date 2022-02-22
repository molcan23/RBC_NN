import global_variables as cs
import numpy as np
import sklearn.model_selection


def dataset_load(dataset_path=''):
    dataset_path=f'data/dataset/W_{cs.TS_LENGTH}_A_{cs.NUMBER_OF_AUGMENTATION}_X_{cs.SELECTED_AXIS}'
    training_data1 = np.empty([1, cs.TS_LENGTH, len(cs.SELECTED_COLUMNS)], dtype=float)
    target_data1 = []

    print(f'{cs.TS_LENGTH} {cs.NUMBER_OF_AUGMENTATION}')

    for i in range(cs.number_of_cells):  # 52  # 48
        print(i, "/", 54)
        data_out = f'{dataset_path}/data_time_series_W_{cs.TS_LENGTH}_A_{cs.NUMBER_OF_AUGMENTATION}' \
                   f'_X_{cs.SELECTED_AXIS}_rbc_{str(i)}.npy'
        label_out = f'{dataset_path}/data_labels_W_{cs.TS_LENGTH}_A_{cs.NUMBER_OF_AUGMENTATION}' \
                    f'_X_{cs.SELECTED_AXIS}_rbc_{str(i)}.npy'

        trd = np.load(data_out)
        tad = np.load(label_out)
        training_data1 = np.append(training_data1, trd, axis=0)
        target_data1 = np.append(target_data1, tad, axis=0)

    training_data1 = np.delete(training_data1, 0, axis=0)

    training_data = np.array(training_data1)

    target_data = np.array(target_data1)

    training_data_CNN = np.reshape(training_data,
                                   [training_data.shape[0],
                                    training_data.shape[1],
                                    training_data.shape[2],
                                    1])

    print('X_train shape == {}.'.format(training_data.shape))

    print('X_train shape == {}.'.format(training_data_CNN.shape))
    print('y_train shape == {}.'.format(target_data.shape))

    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(training_data, target_data, test_size=0.1, shuffle=True)

    X_train_CNN, X_test_CNN, y_train_CNN, y_test_CNN = \
        sklearn.model_selection.train_test_split(training_data_CNN, target_data, test_size=0.1, shuffle=True)

    return X_train, X_test, y_train, y_test, X_train_CNN, X_test_CNN, y_train_CNN, y_test_CNN
