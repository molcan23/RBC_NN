import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import global_variables as cs

if __name__ == '__main__':

    path = 'output/dataset'
    name = 'cnn_training'

    if not os.path.exists(cs.COMPARISON_PLOTS):
        os.makedirs(cs.COMPARISON_PLOTS)

    df_all = pd.DataFrame()

    for i in os.listdir(path):
        try:
            print(i)
            a = pd.read_csv(f'{path}/{i}/statistics.txt', sep='\t', header=None)
            a.columns = ['model', 'window', 'loss', 'accuracy']
            # print(a)
            a['ax'] = i.split('X')[-1][1:]
            df_all = pd.concat([df_all, a], ignore_index=True)
        except Exception as e:
            print(e, i)

    df_all['window'] = df_all['window'].astype('int')
    df_all['loss'] = df_all['loss'].astype('float')
    df_all['accuracy'] = df_all['accuracy'].astype('float')
    df_all['model'] = df_all['model'].apply(lambda x: x[:-1])
    df_all['model_ax'] = df_all['model'] + '_' + df_all['ax']

    # print(df_all)

    plt.figure(figsize=(15, 10))
    sns.color_palette("light:#5A9", as_cmap=True)
    g = sns.lineplot(data=df_all, x='window', y='loss', hue='model_ax', style='model_ax',
                     markers=True, dashes=False, linestyle="dashed")  # , legend=False)

    handles, labels = plt.gca().get_legend_handles_labels()

    # specify order
    # order = [2, 3, 4, 8, 9, 10, 5, 6, 7, 0, 1, 11]
    # plt.legend([handles[i] for i in order], [labels[i] for i in order])
    plt.title("Model-input data comparison")
    plt.savefig(f'plots/model_ax_loss_comparison1.png')
    # plt.show()
    plt.close()

    sns.lineplot(data=df_all, x='window', y='accuracy', hue='model_ax', style='model_ax',
                 markers=True, dashes=False, linestyle="dashed", palette="flare")  # , legend=False)

    handles, labels = plt.gca().get_legend_handles_labels()

    # specify order
    # order = [2, 3, 4, 8, 9, 10, 5, 6, 7, 0, 1, 11]
    # plt.legend([handles[i] for i in order], [labels[i] for i in order])

    plt.savefig(f'plots/model_ax_accuracy_comparison1.png')
    plt.close()
