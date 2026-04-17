
from datetime import datetime, timedelta

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

#Alpha (Younes)
def factor1(df):
    new_swdw = []
    new_swdif = []
    for Eg, alpha, Edf in zip(df['Sw_dw'],df['elev_solar'],df['Sw_dif']):
        if rad2deg(alpha) > 10:
            new_swdw.append(Eg)
            new_swdif.append(Edf)
        else:
            new_swdw.append(np.nan)
            new_swdif.append(np.nan)
    df['Sw_dw'] = new_swdw
    df['Sw_dif'] = new_swdif
    return df

#Eg/Et < 1 (Younes)
def factor2(df):
    new_swdw = []
    new_swdif = []
    for Eg, Et, Edf in  zip(df['Sw_dw'],df['oc_topo'],df['Sw_dif']):
        if Et!=0 and Eg/Et < 1:
            new_swdw.append(Eg)
            #new_swdif.append(Edf)
        else:
            new_swdw.append(np.nan)
            #new_swdif.append(np.nan)
    df['Sw_dw'] = new_swdw
    #df['Sw_dif'] = new_swdif
    return df

#Edf/Et < 0.8 (Younes)
def factor3(df):
    new_swdw = []
    new_swdif = []
    for Edf, Et, Eg in  zip(df['Sw_dif'],df['oc_topo'],df['Sw_dw']):
        if not np.isnan(Edf) and Et!=0 and Edf/Et > 0.8:
            #new_swdw.append(np.nan)
            new_swdif.append(np.nan)
        else:
            #new_swdw.append(Eg)
            new_swdif.append(Edf)
    #df['Sw_dw'] = new_swdw
    df['Sw_dif'] = new_swdif
    return df

#Edf/Eg < 1.1 (Younes)
def factor4(df):
    new_swdw = []
    new_swdif = []
    for Edf, Eg in  zip(df['Sw_dif'],df['Sw_dw']):
        if not np.isnan(Edf) and Eg!=0 and Edf/Eg > 1.1:
            #new_swdw.append(np.nan)
            new_swdif.append(np.nan)
        else:
            #new_swdw.append(Eg)
            new_swdif.append(Edf)
    #df['Sw_dw'] = new_swdw
    df['Sw_dif'] = new_swdif
    return df

#Journéé PT5 S1
def factor5(df):
    new_swdw = [df['Sw_dw'][0]]
    new_swdif = [df['Sw_dif'][0]]
    G, D, E, timestamp = df['Sw_dw'], df['Sw_dif'], df['oc_topo'], df.index
    for i in range(1,len(timestamp)):
        if timestamp[i] == timestamp[i-1]-timedelta(hours=1):
            if not (np.isnan(G[i]) or np.isnan(G[i-1]) or np.isnan(D[i-1]) or np.isnan(D[i-1])):
                if abs(G[i]/E[i] - G[i-1]/E[i-1]) < 0.75 and abs(D[i]/E[i] - D[i-1]/E[i-1]) < 0.35:
                    new_swdw.append(G[i])
                    new_swdif.append(D[i])
                    continue
                else:
                    new_swdw.append(np.nan)
                    new_swdif.append(np.nan)
                    continue
        new_swdw.append(G[i])
        new_swdif.append(D[i])
    df['Sw_dw'] = new_swdw
    df['Sw_dif'] = new_swdif
    return df

#Ajuste do anel (controle da difusa pela data)
def factor6(df):
    new_swdif = []
    observ = datetime(2018,11,7)

    for day, dif in zip(df.index,df['Sw_dif']):
        if day < observ:
            new_swdif.append(np.nan)
        else:
            new_swdif.append(dif)

    df['Sw_dif'] = new_swdif
    return df

labmim = pd.read_csv('labmim_horario_final.dat',sep=',')
labmim.index = pd.to_datetime(labmim[['year','month','day','hour']])

quantity = {'total':0,'factor1':0,'factor2':0,'factor3':0,'factor4':0,'factor5':0,'factor6':0}

quantity['total'] = count_data(labmim['Sw_dw'])

labmim = factor1(labmim)

quantity['factor1'] = count_data(labmim['Sw_dw'])

labmim = factor2(labmim)

quantity['factor2'] = count_data(labmim['Sw_dw'])

labmim = factor3(labmim)

quantity['factor3'] = count_data(labmim['Sw_dw'])

labmim = factor4(labmim)

quantity['factor4'] = count_data(labmim['Sw_dw'])

labmim = factor5(labmim)

quantity['factor5'] = count_data(labmim['Sw_dw'])

labmim = factor6(labmim)

quantity['factor6'] = count_data(labmim['Sw_dw'])

print(quantity)
print(1-quantity['factor1']/quantity['total'])
print(1-quantity['factor2']/quantity['total'])
print(1-quantity['factor3']/quantity['total'])
print(1-quantity['factor4']/quantity['total'])
print(1-quantity['factor5']/quantity['total'])
print(1-quantity['factor6']/quantity['total'])

labmim.to_csv('labmim_completo_controle.dat',sep=';',na_rep='nan',index=False)

