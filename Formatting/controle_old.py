import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from datetime import datetime, timedelta
import warnings 
warnings.filterwarnings('ignore')

df = pd.read_csv('LBM_bruto_premed.dat',sep=',')
df.index = pd.to_datetime(df['TIMESTAMP'])
df.drop(columns=['TIMESTAMP'],inplace=True)
df.index.name = None

limits = {'PSP1_Wm2':[0,1370], 'PIR1_Wm2':[340,460], 'NRLite_Wm2Cr':[-100,1000], 'PAR_Wm2':[0,600], 'Temp1':[15,35], 'RH1':[20,100],
       'WS_WVT':[0,10], 'WindDir_WVT':[0,360], 'CM3Up_Wm2':[0,1400], 'CM3Dn_Wm2':[0,500], 'CNR1TC':[15,40], 'Temp_WXT':[15,37],
       'CG3Up_Wm2Cr':[350,500], 'CG3Dn_Wm2Cr':[350,580], 'Net_Wm2':[-100,1100], 'WS_WXT':[0,10], 'WD_WXT':[0,360], 'precip':[0,15],
       'RH_WXT':[20,100], 'Pmb_WXT':[1000,1020], 'CMP21_Wm2':[0,700]}

skip = ['precip','WD_WXT','WindDir_WVT']

means = dict()
means['TIMESTAMP'] = []
means['u_WVT'] = []
means['v_WVT'] = []
means['u_WXT'] = []
means['v_WXT'] = []

for col,(dw,up) in limits.items():
    df.loc[(df[col]<dw) | (df[col]>up),col] = np.nan
    means[col] = []

df['u_WXT'] = -df['WS_WXT']*np.sin(np.radians(df['WD_WXT']))
df['v_WXT'] = -df['WS_WXT']*np.cos(np.radians(df['WD_WXT']))
df['u_WVT'] = -df['WS_WVT']*np.sin(np.radians(df['WindDir_WVT']))
df['v_WVT'] = -df['WS_WVT']*np.cos(np.radians(df['WindDir_WVT']))

start, end = df.index[0], df.index[-1]
start, end = start.replace(minute=30), end.replace(minute=30)
end = end + timedelta(hours=1)

total = len(df.index)
i = 0

while start <= end:
    next_date = start + timedelta(hours=1)
    means['TIMESTAMP'].append(next_date.replace(minute=0))

    mask = (df.index >= start) & (df.index < next_date)

    for col in df.columns:
        if col not in skip:
            tmp = df.loc[mask,col].dropna()
            if len(tmp) >= 6:
                means[col].append(np.mean(tmp))
            else:
                means[col].append(np.nan)

    remaining = ['precip','u_WXT','v_WXT','u_WVT','v_WVT']
    tmpSkip = []
    for x in remaining:
        tmpSkip.append(df.loc[mask,x].dropna())

    if len(tmpSkip[0])>=6:
        means['precip'].append(np.sum(tmpSkip[0]))
    else:
        means['precip'].append(np.nan)

    if len(tmpSkip[1])>=6 and len(tmpSkip[2])>=6:
        alfa = np.arctan2(np.mean(tmpSkip[2]),np.mean(tmpSkip[1]))
        means['WD_WXT'].append(np.fmod(3*(np.pi/2) - alfa, 2*np.pi)*(180/np.pi))
    else:
        means['WD_WXT'].append(np.nan)
    
    if len(tmpSkip[3])>=6 and len(tmpSkip[4])>=6:
        alfa = np.arctan2(np.mean(tmpSkip[4]),np.mean(tmpSkip[3]))
        means['WindDir_WVT'].append(np.fmod(3*(np.pi/2) - alfa, 2*np.pi)*(180/np.pi))
    else:
        means['WindDir_WVT'].append(np.nan)

    if i%1000==0:
        print('Current progress: {}'.format((i/(total/12))*100))
    start = next_date
    i+=1

labmim = pd.DataFrame(means)
labmim.to_csv('lbm_horario_completo.dat',index=False,sep=',',na_rep='nan')