import global_variables as cs
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K


def percentage_difference(y_hat, y):
    return np.abs(((y_hat - y) / y) * 100)


def absolute_error(y_hat, y):
    return np.abs(y_hat - y)


def data_for_plot(pred, real):
    dir_labels = {}
    dir_prediction = {}
    diff_fun = percentage_difference

    for i, j in zip(pred, real):
        if str(j) in dir_labels.keys():
            dir_prediction[str(j)].append(i[0])
            dir_labels[str(j)].append(diff_fun(i[0], j))
        else:
            dir_labels[str(j)] = [diff_fun(i[0], j)]
            dir_prediction[str(j)] = [i[0]]

    return [dir_labels[x] for x in sorted(dir_labels.keys())], dir_labels, dir_prediction


def ks_boxplots(data, name, dir_labels, dir_prediction, outliers):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.boxplot(data, showfliers=outliers, showmeans=True)

    plt.title('Absolute Percentage Error by True Value of ks')
    ax.set_xlabel('Value of ks')
    ax.set_ylabel('Absolute Percentage Error')
    ax.set_xticklabels(labels=sorted(dir_labels.keys()))
    ax.grid()

    for i in range(len(data)):
        y = data[i]
        x = np.random.normal(i, 0.02, len(y))
        plt.plot(x + 1, y, 'r.', alpha=0.2)

    plt.savefig(f'{cs.SAVE_OUT}/{name}.png', bbox_inches='tight')
    # plt.show()
    plt.close()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.boxplot(data, showfliers=outliers, showmeans=True)

    plt.title('Absolute Percentage Error by True Value of ks')
    ax.set_xlabel('Value of ks')
    ax.set_ylabel('Absolute Percentage Error')
    ax.set_xticklabels(labels=sorted(dir_labels.keys()))
    ax.grid()

    plt.savefig(f'{cs.SAVE_OUT}/{name}_NO.png', bbox_inches='tight')

    # plt.show()
    plt.close()

    for idx, label in enumerate(dir_labels.keys()):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_axes([0, 0, 1, 1])

        x = dir_prediction[label]
        y = np.random.normal(idx, 0.02, len(x))
        plt.plot(x, y, 'r.', alpha=0.2)

        plt.title(f'Absolute Percentage Error of ks={label}')
        ax.set_xlabel('Value of predicted ks')
        ax.set_ylabel('Absolute Percentage Error')
        ax.grid()

        plt.savefig(f'{cs.SAVE_OUT}/{str(label)}.png', bbox_inches='tight')
        # plt.show()
        plt.close()


#########################################################
############ Plotting function for training #############

# from sklearn.metrics import r2_score
def r2_keras(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - (SS_res / (SS_tot + K.epsilon()))


def plot_learning_acc_loss(history, name=""):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'{cs.SAVE_OUT}/{name}_model_accuracy.png', bbox_inches='tight')
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

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    print(accuracy)
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('confusion_matrix.png')  # show()