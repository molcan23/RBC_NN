import global_variables as cs
import numpy as np
import sklearn.model_selection
import os


def augmentation(x, gaussian_noise_level=.1, offset_noise_level=.25):
    noise = gaussian_noise_level * np.random.normal(size=x.shape)
    offset_noise = 2. * np.random.uniform(size=x.shape) - 1.0
    x_result = x + noise + offset_noise_level * offset_noise
    return x_result


def dataset_load(dataset_path='', number_of_augmentations=10):
    dataset_path = f'data/dataset/W_{cs.TS_LENGTH}_A_{cs.NUMBER_OF_AUGMENTATION}_X_{cs.SELECTED_AXIS}'
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

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(training_data, target_data,
                                                                                test_size=.1, random_state=1)

    X_train, X_val2, y_train, y_val2 = sklearn.model_selection.train_test_split(X_train, y_train,
                                                                                 test_size=.1, random_state=1)

    X_train, X_val1, y_train, y_val1 = sklearn.model_selection.train_test_split(X_train, y_train,
                                                                                test_size=.1, random_state=1)

    trd = []
    tad = []
    for sample, target in zip(X_train, y_train):
        trd.append(sample)
        tad.append(target)
        for _ in range(number_of_augmentations):
            trd.append(augmentation(sample))
            tad.append(target)

    X_train, y_train = np.array(trd), np.array(tad)

    X_train_CNN = np.reshape(X_train, [X_train.shape[0], X_train.shape[1], X_train.shape[2], 1])
    X_val1_CNN = np.reshape(X_val1, [X_val1.shape[0], X_val1.shape[1], X_val1.shape[2], 1])
    X_val2_CNN = np.reshape(X_val2, [X_val2.shape[0], X_val2.shape[1], X_val2.shape[2], 1])
    X_test_CNN = np.reshape(X_test, [X_test.shape[0], X_test.shape[1], X_test.shape[2], 1])

    # print('X_train shape == {}.'.format(training_data.shape))

    print('X_train shape == {}.'.format(X_train.shape))
    print('X_val1 shape == {}.'.format(X_val1.shape))
    print('X_val2 shape == {}.'.format(X_val2.shape))
    print('X_test shape == {}.'.format(X_test.shape))
    print('y_train shape == {}.'.format(target_data.shape))

    if not os.path.exists(f'data/{cs.TS_LENGTH}'):
        os.makedirs(f'data/{cs.TS_LENGTH}')

    np.save(f'data/{cs.TS_LENGTH}/X_train', np.array(X_train))
    np.save(f'data/{cs.TS_LENGTH}/y_train', np.array(y_train))
    np.save(f'data/{cs.TS_LENGTH}/X_val1', np.array(X_val1))
    np.save(f'data/{cs.TS_LENGTH}/y_val1', np.array(y_val1))
    np.save(f'data/{cs.TS_LENGTH}/X_val2', np.array(X_val2))
    np.save(f'data/{cs.TS_LENGTH}/y_val2', np.array(y_val2))
    np.save(f'data/{cs.TS_LENGTH}/X_test', np.array(X_test))
    np.save(f'data/{cs.TS_LENGTH}/y_test', np.array(y_test))

    return X_train, X_val1, X_val2, X_test, y_train, y_val1, y_val2, y_test,\
           X_train_CNN, X_val1_CNN, X_val2_CNN, X_test_CNN
