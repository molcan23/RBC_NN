import numpy as np
from numpy import array
import os
import sklearn.preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay, confusion_matrix
import global_variables as cs

if not os.path.exists(cs.COMPARISON_PLOTS):
    os.makedirs(cs.COMPARISON_PLOTS)

path = 'output/dataset'
only_cell_files = sorted([f for f in os.listdir(path)])
k_s = ['0.005', '0.009', '0.015', '0.03', '0.05', '0.1', '0.15', '0.225', '0.3']
k_s_ = [[0.005, 1], [0.009, 2], [0.015, 3], [0.03, 4], [0.05, 5],
        [0.1, 6], [0.15, 7], [0.225, 8], [0.3, 9]]
borders = [float('-inf'), 0.007, 0.012, 0.0225, 0.04, 0.075, 0.125, 0.1875, 0.2625, float('inf')]


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          name='',
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
    print(name, '\t', accuracy)
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
    plt.savefig(f'{cs.COMPARISON_PLOTS}/{name}_confusion_matrix.png')  # show()
    plt.close()

def percentage_difference(y_hat, y):
    return np.abs(((y_hat - y) / y) * 100)


def real_to_class(real):
    for i in range(len(borders) - 1):
        if borders[i] < real <= borders[i + 1]:
            return k_s[i]


def data_for_plot(pred, real, folder):
    """
    TODO: finish this function with OneHoteEncoding
    """
    oh_pred = []
    oh_real = []
    for i, j in zip(pred, real):
        oh_pred.append(real_to_class(i))
        oh_real.append(real_to_class(j))

    oh_real = np.array(oh_real)
    oh_pred = np.array(oh_pred)

    cm = confusion_matrix(oh_real, oh_pred)

    plot_confusion_matrix(cm,
                          k_s,
                          title=f'Confusion matrix {folder}',
                          name=folder,
                          cmap=None,
                          normalize=True)


if __name__ == '__main__':

    print(only_cell_files)
    for f in only_cell_files[1:]:
        try:
            y_train = np.load(f'{path}/{f}/y_train.npy')
            y_hat_train = np.load(f'{path}/{f}/y_train_predicted.npy')

            y_test = np.load(f'{path}/{f}/y_val.npy')
            y_hat_test = np.load(f'{path}/{f}/y_val_predicted.npy')

            data_for_plot(y_hat_test, y_test, f)

        except Exception as e:
            print(e)

