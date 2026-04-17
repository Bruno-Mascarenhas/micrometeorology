from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import rad2deg


def count_data(data):
    ans = 0
    for x in data:
        if np.isnan(x):
            continue
        ans+=1
    return ans

def factor0(df):
    new_swdw = []
    for Eg, precip in zip(df['sw_dw'],df['precip']):
        if precip>0:
            new_swdw.append(np.nan)
        else:
            new_swdw.append(Eg)
    df['sw_dw'] = new_swdw
    return df

#Alpha (Younes)
def factor1(df):
    new_swdw = []
    for Eg, alpha in zip(df['sw_dw'],df['elev_solar']):
        if rad2deg(alpha) > 10:
            new_swdw.append(Eg)
        else:
            new_swdw.append(np.nan)

    df['sw_dw'] = new_swdw
    return df

#Eg/Et < 1 (Younes)
def factor2(df):
    new_swdw = []
    for Eg, Et in  zip(df['sw_dw'],df['oc_topo']):
        if Et!=0 and Eg/Et < 1.1:
            new_swdw.append(Eg)
        else:
            new_swdw.append(np.nan)
    df['sw_dw'] = new_swdw
    return df

#Journéé PT5 S1
def factor5(df):
    new_swdw = [df['sw_dw'][0]]
    G, E, timestamp = df['sw_dw'], df['oc_topo'], df.index
    for i in range(1,len(timestamp)):
        if timestamp[i] == timestamp[i-1]-timedelta(hours=1):
            if not (np.isnan(G[i]) or np.isnan(G[i-1])):
                if abs(G[i]/E[i] - G[i-1]/E[i-1]) < 0.75:
                    new_swdw.append(G[i])
                    continue
                else:
                    new_swdw.append(np.nan)
                    continue
        new_swdw.append(G[i])
    df['sw_dw'] = new_swdw
    return df


inmet = pd.read_csv('SSA_inmet_completo.dat',sep=' ')
inmet.index = pd.to_datetime(inmet[['year','month','day','hour']])

quantity = {'total':0,'factor0':0,'factor1':0,'factor2':0,'factor5':0}

quantity['total'] = count_data(inmet['sw_dw'])

inmet = factor0(inmet)

quantity['factor0'] = count_data(inmet['sw_dw'])

inmet = factor1(inmet)

quantity['factor1'] = count_data(inmet['sw_dw'])

inmet = factor2(inmet)

quantity['factor2'] = count_data(inmet['sw_dw'])

inmet = factor5(inmet)

quantity['factor5'] = count_data(inmet['sw_dw'])

print(quantity)
print(1-quantity['factor0']/quantity['total'])
print(1-quantity['factor1']/quantity['total'])
print(1-quantity['factor2']/quantity['total'])
print(1-quantity['factor5']/quantity['total'])

inmet.to_csv('inmet_completo_controle.dat',sep=';',na_rep='nan',index=False)

inmet['oc_topo'].plot()
inmet['sw_dw'].plot()
plt.show()
