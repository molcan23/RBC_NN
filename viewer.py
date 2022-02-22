import numpy as np
import pandas as pd


a = np.load('data_time_series_W_10_A_30_X_xy_rbc_4.npy')

print(pd.DataFrame(a[0]))
print()

a = np.load('data_time_series_W_10_A_30_X_xy_rbc_5.npy')

print(pd.DataFrame(a[0]))