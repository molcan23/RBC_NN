import os
from boxplots_correct import *

path = 'output/dataset'


def percentage_difference(y_hat, y):
    return np.abs(((y_hat - y) / y) * 100)


only_cell_files = sorted([f for f in os.listdir(path)])

save = 'boxplots'
if not os.path.exists(save):
    os.makedirs(save)

for f in only_cell_files[2:]:
    #try:
    y_train = np.load(f'{path}/{f}/y_train.npy')
    y_hat_train = np.load(f'{path}/{f}/y_train_predicted.npy')

    y_test = np.load(f'{path}/{f}/y_val.npy')
    y_hat_test = np.load(f'{path}/{f}/y_val_predicted.npy')

    data, dir_labels, dir_prediction = data_for_plot(y_hat_test, y_test)

    if not os.path.exists(f'{save}/{f}'):
        os.makedirs(f'{save}/{f}')
    ks_boxplots(data, f'{save}/{f}', dir_labels, dir_prediction, False)
    #except Exception as e:
    #    print(e, f)
