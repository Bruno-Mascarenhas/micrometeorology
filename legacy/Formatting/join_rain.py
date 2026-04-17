from functools import reduce

import numpy as np
import pandas as pd

names = ['LBM_rain_2016.dat','LBM_rain_2017.dat','LBM_rain_2018_2019.dat','LBM_rain_2019.dat','LBM_rain_2020.dat']

dfs = []

for name in names:
    tmp = pd.read_csv(name,sep=',',low_memory=False)
    tmp.drop(columns=['rtime(9)', 'rtime(1)', 'rtime(4)', 'rtime(5)', 'RECORD'], inplace=True)
    for col in tmp.columns:
        if 'TIMESTAMP' not in col:
            tmp[col] = tmp[col].astype(np.float64)
    tmp.index = pd.to_datetime(tmp['TIMESTAMP'])
    tmp.index.name = None
    dfs.append(tmp)

dfT = reduce(lambda left, right: pd.merge_ordered(left,right),dfs)

dfT.to_csv('lbm_rain_all_corr.dat',index=False,sep=',',na_rep='nan')
