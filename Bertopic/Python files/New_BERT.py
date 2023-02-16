import numpy as np
import pandas as pd
import pickle
#%%

path = "'D:/CSV_file_NYT_1940-1979/{}_{}.csv"

df = pd.Dataframe()

for i in range(1940, 1941):
    for j in range(1, 13):
        tmp = pd.read_csv(path.format(i, j))
        df = pd.concat(data_list, axis=0, ignore_index=True)