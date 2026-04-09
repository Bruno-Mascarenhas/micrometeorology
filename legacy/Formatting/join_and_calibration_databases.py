import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from datetime import datetime

names = ['LBM_lenta_2016.dat','LBM_lenta_2017.dat','LBM_lenta_2018_1.dat','LBM_lenta_2018_2.dat',
        'LBM_lenta_2019.dat','LBM_lenta_2019_2.dat','LBM_lenta_2019_3.dat','LBM_lenta_2019_4.dat','LBM_lenta_2020.dat']

# Alterações feitas nos arquivos:
## Nomes com cor e Cr representando o mesmo dado

dfs = []

for name in names:
    tmp = pd.read_csv(name,sep=',',low_memory=False)
    for col in tmp.columns:
        if 'TIMESTAMP' not in col:
            tmp[col] = tmp[col].astype(np.float64)
    tmp.index = pd.to_datetime(tmp['TIMESTAMP'])
    tmp.index.name = None
    tmp.drop(columns='RECORD rtime batt_volt'.split(), inplace=True)
    dfs.append(tmp)

dfT = reduce(lambda left, right: pd.merge_ordered(left,right),dfs)

#dfT = pd.read_csv('labmim_bruto_completo.dat',sep=',')
dfT.index = pd.to_datetime(dfT['TIMESTAMP'])

### Calibrações
#cmp21
dfT.loc[dfT.index <= datetime(2019,10,12), 'CMP21_Wm2_Avg'] = np.nan
dfT.loc[:,'CMP21_Wm2_Avg'] *= (9.38/9.52)

#cm3up
dfT.loc[dfT.index.year < 2019, 'CM3Up_Wm2_Avg'] *= (10.46/10.79)
dfT.loc[dfT.index.year >= 2019, 'CM3Up_Wm2_Avg'] *= (10.46/10.4)

#psp
dfT.loc[dfT.index.year == 2018, 'PSP1_Wm2_Avg'] *= (8.37/8.92)
dfT.loc[dfT.index.year > 2018, 'PSP1_Wm2_Avg'] *= (8.37/7.66)

dfT.to_csv('lbm_lenta_all_corr.dat',index=False,sep=',',na_rep='nan')