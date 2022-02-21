import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path = 'to_download_CNN/cnn_var_RBC_NN/output/dataset'
name = 'cnn_training'

df_all = pd.DataFrame()

for i in os.listdir(path):
    # print(i)
    a = pd.read_csv(f'{path}/{i}/statistics.txt', sep='\t', header=None)
    a.columns = ['model', 'window', 'mape', 'r2']
    # print(a)
    a['ax'] = i.split('_')[-1]
    df_all = pd.concat([df_all, a], ignore_index=True)

df_all['window'] = df_all['window'].astype('int')
df_all['mape'] = df_all['mape'].astype('float')
df_all['r2'] = df_all['r2'].astype('float')

print(df_all)

for ax in ['xy', 'xz', 'xyz']:
    dff = df_all[(df_all['ax'] == ax)]
    sns.lineplot(data=dff, x='window', y='mape', hue='model')
    plt.savefig(f'plots/{name}_{ax}.png')
    plt.close()
    # break
