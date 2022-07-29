import global_variables as cs
import numpy as np
import matplotlib.pyplot as plt
# from tensorflow.keras import backend as K


def percentage_difference(y_hat, y):
    y_hat, y = np.array(y_hat), np.array(y)
    #print(np.abs(((y_hat - y) / y) * 100))
    return np.abs(((y_hat - y) / y) * 100)


def absolute_error(y_hat, y):
    return np.abs(y_hat - y)


def data_for_plot(pred, real):
    dir_labels = {}
    dir_prediction = {}
    diff_fun = percentage_difference

    for i, j in zip(pred, real):  # zip(predictions_future, y_test):
        if str(j) in dir_labels.keys():
            dir_prediction[str(j)].append(i)
            dir_labels[str(j)].append(diff_fun(i, j))
        else:
            dir_labels[str(j)] = [diff_fun(i, j)]
            dir_prediction[str(j)] = [i]

    return [dir_labels[x] for x in sorted(dir_labels.keys())], dir_labels, dir_prediction


def ks_boxplots(data, name, dir_labels, dir_prediction, outliers):

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    #print(data)
    ax.boxplot(np.array(data), showfliers=outliers, showmeans=True)

    plt.title('Absolute Percentage Error by True Value of ks')
    ax.set_xlabel('Value of ks')
    ax.set_ylabel('Absolute Percentage Error')
    ax.set_xticklabels(labels=sorted(dir_labels.keys()))
    ax.grid()

    for i in range(len(data)):
        y = data[i]
        x = np.random.normal(i, 0.02, len(y))
        plt.plot(x + 1, y, 'r.', alpha=0.2)

    plt.savefig(f'{name}/a.png', bbox_inches='tight')
    #plt.show()
    plt.close()

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.boxplot(np.array(data), showfliers=outliers, showmeans=True)

    plt.title('Absolute Percentage Error by True Value of ks')
    ax.set_xlabel('Value of ks')
    ax.set_ylabel('Absolute Percentage Error')
    ax.set_xticklabels(labels=sorted(dir_labels.keys()))
    ax.grid()

    plt.savefig(f'{name}/a_NO.png', bbox_inches='tight')
    # plt.savefig(f'{cs.SAVE_OUT}/{name}_NO.png', bbox_inches='tight')
    # plt.show()
    plt.close()

    for idx, label in enumerate(dir_labels.keys()):
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_axes([0, 0, 1, 1])

        x = dir_prediction[label]
        y = np.random.normal(idx, 0.02, len(x))
        plt.plot(x, y, 'r.', alpha=0.2)

        plt.title(f'Absolute Percentage Error of ks={label}')
        ax.set_xlabel('Value of predicted ks')
        ax.set_ylabel('Absolute Percentage Error')
        ax.grid()

        plt.savefig(f'{name}/{str(label)}.png', bbox_inches='tight')
        # plt.show()
        plt.close()


# def ks_boxplots(data, name, dir_labels, outliers):
#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_axes([0, 0, 1, 1])
#     bp = ax.boxplot(data, showfliers=outliers, showmeans=True)
#
#     plt.title('Absolute Percentage Error by True Value of ks')
#     ax.set_xlabel('Value of ks')
#     ax.set_ylabel('Absolute Percentage Error')
#     ax.set_xticklabels(labels=sorted(dir_labels.keys()))
#     ax.grid()
#
#     for i in range(len(data)):
#         y = data[i]
#         x = np.random.normal(i, 0.02, len(y))
#         plt.plot(x + 1, y, 'r.', alpha=0.2)
#
#     plt.savefig(f'{cs.SAVE_OUT}/{name}.png', bbox_inches='tight')
#     # plt.show()
#     plt.close()
#
#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_axes([0, 0, 1, 1])
#     bp = ax.boxplot(data, showfliers=outliers, showmeans=True)
#
#     plt.title('Absolute Percentage Error by True Value of ks')
#     ax.set_xlabel('Value of ks')
#     ax.set_ylabel('Absolute Percentage Error')
#     ax.set_xticklabels(labels=sorted(dir_labels.keys()))
#     ax.grid()
#
#     plt.savefig(f'{cs.SAVE_OUT}/{name}_NO.png', bbox_inches='tight')
#     # plt.show()
#     plt.close()


#########################################################
############ Plotting function for training #############

# from sklearn.metrics import r2_score
def r2_keras(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - (SS_res / (SS_tot + K.epsilon()))


def plot_learning_acc_loss(history, name=""):
    # summarize history for r^2
    plt.plot(history.history['r2_keras'])
    plt.plot(history.history['val_r2_keras'])
    plt.title('r2_keras')
    plt.ylabel('mean_absolute_percentage_error')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'{cs.SAVE_OUT}/{name}_model_r2_keras.png', bbox_inches='tight')
    # plt.show()
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'{cs.SAVE_OUT}/{name}_model_loss.png', bbox_inches='tight')
    # plt.show()
    plt.close()
