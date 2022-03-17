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
            # print(i)
            a = pd.read_csv(f'{path}/{i}/statistics.txt', sep='\t', header=None)
            a.columns = ['model', 'window', 'mape', 'r2']
            # print(a)
            a['ax'] = i.split('_')[-1]
            df_all = pd.concat([df_all, a], ignore_index=True)
        except Exception as e:
            print(e, i)

    df_all['window'] = df_all['window'].astype('int')
    df_all['mape'] = df_all['mape'].astype('float')
    df_all['r2'] = df_all['r2'].astype('float')
    df_all['model'] = df_all['model'].apply(lambda x: x[:-1])
    df_all['model_ax'] = df_all['model'] + '_' + df_all['ax']

    print(df_all[df_all['window'] == 5])

    plt.figure(figsize=(15, 10))
    sns.color_palette("light:#5A9", as_cmap=True)
    sns.lineplot(data=df_all, x='window', y='mape', hue='model_ax', style='model_ax',
                 markers=True, dashes=False, linestyle="dashed")  # , legend=False)
    # sns.scatterplot(data=dff, x='window', y='mape', hue='model')
    plt.savefig(f'plots/model_ax_comparison.png')
    # plt.show()
    plt.close()

    sns.lineplot(data=df_all, x='window', y='r2', hue='model_ax', style='model_ax',
                 markers=True, dashes=False, linestyle="dashed", palette="flare")  # , legend=False)
    # sns.scatterplot(data=dff, x='window', y='mape', hue='model')
    plt.savefig(f'plots/model_ax_r2_comparison.png')
    # plt.show()
    plt.close()
