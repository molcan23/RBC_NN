import numpy as np
import os
from boxplots_correct import *
import matplotlib.pyplot as plt


def percentage_difference(y_hat, y):
    return np.abs(((y_hat - y) / y) * 100)


y_train = np.load('investigation/y_train_5.npy')
y_hat_train = np.load('investigation/y_train_predicted_5.npy')

y_test = np.load('investigation/y_test_5.npy')
y_hat_test = np.load('investigation/y_test_predicted_5.npy')

data, dir_labels, dir_prediction = data_for_plot(y_hat_test, y_test)
print(1)
ks_boxplots(data, '', dir_labels, dir_prediction, False)
