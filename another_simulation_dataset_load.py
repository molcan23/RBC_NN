import global_variables as cs
import numpy as np
import os


def dataset_load():
    dataset_path = f'data/dataset/as_W_{cs.TS_LENGTH}_A_{cs.NUMBER_OF_AUGMENTATION}_X_{cs.SELECTED_AXIS}'
    training_data1 = np.empty([1, cs.TS_LENGTH, len(cs.SELECTED_COLUMNS)], dtype=float)
    target_data1 = []

    # print(f'{cs.TS_LENGTH} {cs.NUMBER_OF_AUGMENTATION}')

    for i in range(cs.number_of_cells):  # 52  # 48
        # print(i, "/", 54)
        data_out = f'{dataset_path}/data_as_W_{cs.TS_LENGTH}_A_{cs.NUMBER_OF_AUGMENTATION}' \
                   f'_X_{cs.SELECTED_AXIS}_rbc_{str(i)}.npy'
        label_out = f'{dataset_path}/labels_as_W_{cs.TS_LENGTH}_A_{cs.NUMBER_OF_AUGMENTATION}' \
                    f'_X_{cs.SELECTED_AXIS}_rbc_{str(i)}.npy'

        trd = np.load(data_out)
        tad = np.load(label_out)
        training_data1 = np.append(training_data1, trd, axis=0)
        target_data1 = np.append(target_data1, tad, axis=0)

    training_data1 = np.delete(training_data1, 0, axis=0)

    X_ = np.array(training_data1)

    y_ = np.array(target_data1)

    X_CNN = np.reshape(X_, [X_.shape[0], X_.shape[1], X_.shape[2], 1])

    # print('X_train shape == {}.'.format(X_.shape))
    
    if not os.path.exists(f'data/as_{cs.TS_LENGTH}'):
        os.makedirs(f'data/as_{cs.TS_LENGTH}')

    return X_, y_, X_CNN
