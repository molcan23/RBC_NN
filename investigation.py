import numpy as np
import os
from plotting_utils import *
import matplotlib.pyplot as plt


def percentage_difference(y_hat, y):
    return np.abs(((y_hat - y) / y) * 100)


y_train = np.load('investigation/y_train.txt.npy')
y_hat_train = np.load('investigation/y_train_predicted.txt.npy')

y_test = np.load('investigation/y_test.txt.npy')
y_hat_test = np.load('investigation/y_test_predicted.txt.npy')

# print(y_train)
# for i, j in zip(y_test, y_hat_test):
#     print(percentage_difference(i, j))

dir_labels = {}
diff_fun = percentage_difference

for i, j in zip(y_hat_train, y_train):
    # print(diff_fun(i, j)[0])
    # exit()
    if str(j) in dir_labels.keys():
        dir_labels[str(j)].append(diff_fun(i, j)[0])
    else:
        dir_labels[str(j)] = [diff_fun(i, j)[0]]

dir_labels_o = dir_labels
dir_labels = [dir_labels[x] for x in sorted(dir_labels.keys())]

fig = plt.figure(figsize=(10, 7))
ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot(dir_labels, showfliers=False, showmeans=True)
print(1)
plt.title('Absolute Percentage Error by True Value of ks')
ax.set_xlabel('Value of ks')
ax.set_ylabel('Absolute Percentage Error')
ax.set_xticklabels(labels=sorted(dir_labels_o.keys()))
ax.grid()

for i in range(len(dir_labels)):
    y = dir_labels[i]
    x = np.random.normal(i, 0.02, len(y))
    plt.plot(x + 1, y, 'r.', alpha=0.2)

plt.savefig(f'box.png', bbox_inches='tight')
# plt.show()
plt.close()


fig = plt.figure(figsize=(10, 7))
ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot(dir_labels, showfliers=False, showmeans=True)
print(1)
plt.title('Absolute Percentage Error by True Value of ks')
ax.set_xlabel('Value of ks')
ax.set_ylabel('Absolute Percentage Error')
ax.set_xticklabels(labels=sorted(dir_labels_o.keys()))
ax.grid()

# for i in range(len(dir_labels)):
#     y = dir_labels[i]
#     x = np.random.normal(i, 0.02, len(y))
#     plt.plot(x + 1, y, 'r.', alpha=0.2)

plt.savefig(f'box_NO.png', bbox_inches='tight')
# plt.show()
plt.close()

# data, dir_labels = data_for_plot(y_test, y_hat_test)
#
# ks_boxplots(data, '', dir_labels, False)
