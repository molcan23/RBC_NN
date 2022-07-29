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
            a.columns = ['model', 'WINDOW', 'MAPE', 'r2']
            # print(a)
            a['ax'] = i.split('X')[-1][1:]
            df_all = pd.concat([df_all, a], ignore_index=True)
        except Exception as e:
            print(e, i)

    df_all['WINDOW'] = df_all['WINDOW'].astype('int')
    df_all['MAPE'] = df_all['MAPE'].astype('float')
    df_all['r2'] = df_all['r2'].astype('float')
    df_all['model'] = df_all['model'].apply(lambda x: x[:-1])
    df_all['model_ax'] = df_all['model'] + '_' + df_all['ax']

    #df_all = df_all[(df_all['model'] == 'CNN-LSTM_Conv2D')]
    #df_all = df_all[(df_all['ax'] == 'xyz') | (df_all['ax'] == 'xzjh')]
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #    print(df_all)

    plt.figure(figsize=(9.5, 6))
    sns.color_palette("light:#5A9", as_cmap=True)
    g = sns.lineplot(data=df_all, x='WINDOW', y='MAPE', hue='model_ax', style='model_ax',
                     markers=True, dashes=False, linestyle="dashed")  # , legend=False)

    #handles, labels = plt.gca().get_legend_handles_labels()

    # specify order
    # order = [3, 4, 5, 9, 10, 11, 6, 7, 8, 0, 1, 2]
    # plt.legend([handles[i] for i in order], [labels[i] for i in order])
    plt.title("Model-input data comparison")
    plt.savefig(f'plots/model_ax_comparison.png')
    # plt.show()
    plt.close()

    sns.lineplot(data=df_all, x='WINDOW', y='r2', hue='model_ax', style='model_ax',
                 markers=True, dashes=False, linestyle="dashed", palette="flare")  # , legend=False)

    #handles, labels = plt.gca().get_legend_handles_labels()
    #
    # # specify order
    #order = [3, 4, 5, 9, 10, 11, 6, 7, 8, 0, 1, 2]
    #plt.legend([handles[i] for i in order], [labels[i] for i in order])

    plt.savefig(f'plots/model_ax_r2_comparison.png')
    plt.close()
