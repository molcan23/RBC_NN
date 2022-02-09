from global_variables import *

import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K


def percentage_difference(y_hat, y):
    return np.abs(((y_hat - y) / y) * 100)


def absolute_error(y_hat, y):
    return np.abs(y_hat - y)


def data_for_plot(pred, real):
    dir_labels = {}
    diff_fun = percentage_difference

    for i, j in zip(pred, real):
        if str(j) in dir_labels.keys():
            dir_labels[str(j)].append(diff_fun(i[0], j))
        else:
            dir_labels[str(j)] = [diff_fun(i[0], j)]

    return [dir_labels[x] for x in sorted(dir_labels.keys())], dir_labels


def ks_boxplots(data, name, dir_labels, outliers):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    bp = ax.boxplot(data, showfliers=outliers, showmeans=True)

    plt.title('Absolute Percentage Error by True Value of ks')
    ax.set_xlabel('Value of ks')
    ax.set_ylabel('Absolute Percentage Error')
    ax.set_xticklabels(labels=sorted(dir_labels.keys()))
    ax.grid()

    for i in range(len(data)):
        y = data[i]
        x = np.random.normal(i, 0.02, len(y))
        plt.plot(x+1, y, 'r.', alpha=0.2)

    plt.savefig(f'{SAVE_PATH}/{name}.png')
    plt.show()

    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    bp = ax.boxplot(data, showfliers=outliers, showmeans=True)

    plt.title('Absolute Percentage Error by True Value of ks')
    ax.set_xlabel('Value of ks')
    ax.set_ylabel('Absolute Percentage Error')
    ax.set_xticklabels(labels=sorted(dir_labels.keys()))
    ax.grid()

    plt.savefig(f'{SAVE_PATH}/{name}_NO.png')
    plt.show()


#########################################################
############ Plotting function for training #############

# from sklearn.metrics import r2_score
def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return 1 - (SS_res / (SS_tot + K.epsilon()))


def plot_learning_acc_loss(history, name=""):
    # summarize history for r^2
    plt.plot(history.history['r2_keras'])
    plt.plot(history.history['val_r2_keras'])
    plt.title('r2_keras')
    plt.ylabel('mean_absolute_percentage_error')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'{SAVE_PATH}/{name}_model_r2_keras.png')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'{SAVE_PATH}/{name}_model_loss.png')
    plt.show()
