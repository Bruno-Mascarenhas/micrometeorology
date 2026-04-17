from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['figure.figsize'] = 8, 4
path_fig = 'E:/LABMIM_UFBA/formatado/figuras/'

path_csv = 'E:/LABMIM_UFBA/'

labmim = pd.read_csv(path_csv + 'LBM_lenta_2021.dat', sep=',', skiprows=[0,2,3], parse_dates=True, infer_datetime_format=True)
labmim.index = pd.to_datetime(labmim['TIMESTAMP'])
labmim.drop(columns=['TIMESTAMP', 'RECORD', 'rtime', 'batt_volt', 'panel_temp', 'CM3Up_mv_Avg', 'CG3Up_mv_Avg', 'CM3Dn_mv_Avg', 'CG3Dn_mv_Avg', 'NRLite_Wm2_Avg', 'CMP21_Avg', 'PAR_Den_Avg'], inplace=True)
labmim.index.name = None

labmim_rain = pd.read_csv(path_csv + 'LBM_rain_2021.dat', sep=',', skiprows=[0,2,3], parse_dates=True, infer_datetime_format=True)
labmim_rain.index = pd.to_datetime(labmim_rain['TIMESTAMP'])
labmim_rain.drop(columns=['TIMESTAMP', 'RECORD', 'rtime(9)', 'rtime(1)', 'rtime(4)', 'rtime(5)'], inplace=True)
rain_col = 'PL01_mm_Tot'

labmim['u'] = -labmim['WS_WXT_Avg']*np.sin(np.radians(labmim['WD_WXT_Avg']))
labmim['v'] = -labmim['WS_WXT_Avg']*np.cos(np.radians(labmim['WD_WXT_Avg']))

means = {}
for col in labmim.columns:
    labmim.loc[(labmim[col] < -900),col] = np.nan
    means[col] = []
means['TIMESTAMP'] = []

labmim_rain.loc[(labmim_rain[rain_col] < -900),rain_col] = np.nan
labmim_rain.loc[(labmim_rain[rain_col] > 400),rain_col] = np.nan
means[rain_col] = []

date_end = datetime.now()
current_date = date_end - timedelta(days=7)
current_date = current_date.replace(minute=30)

#current_date = datetime(2020,7,15)
#date_end = datetime(2020,7,22)

while current_date <= date_end:
    means['TIMESTAMP'].append(current_date.replace(minute=0) + timedelta(hours=1))

    next_date = current_date + timedelta(hours=1)
    idxs = np.array(list(map(lambda x: current_date <= x < next_date, labmim.index)))
    idxs_r = np.array(list(map(lambda x: current_date <= x < next_date, labmim_rain.index)))

    for col in labmim.columns:
        df = labmim.loc[idxs,col]
        if len(df) >= 6:
            if col == 'WD_WXT_Avg':
                alfa = np.arctan2(np.nanmean(labmim.loc[idxs,'v']), np.nanmean(labmim.loc[idxs,'u']))
                means[col].append(np.fmod(3*(np.pi/2) - alfa, 2*np.pi)*(180/np.pi))
            else:
                means[col].append(np.nanmean(df))
        else:
            means[col].append(np.nan)

    df = labmim_rain.loc[idxs_r,rain_col]
    if len(df) >= 6:
        means[rain_col].append(np.sum(df))
    else:
        means[rain_col]. append(np.nan)

    current_date = next_date

new = pd.DataFrame(means)
new.index = pd.to_datetime(new['TIMESTAMP'])
new.index.name = None

##salvando temporariamente
#new['year'] = new.index.year
#new['month'] = new.index.month
#new['day'] = new.index.day
#new['hour'] = new.index.hour
#new.to_csv('LAB.dat', sep=',', index=False, na_rep='nan')

#wrfwrf = pd.read_csv(path_csv + 'formatado/series_operacional.dat',sep=',')
#wrfwrf.index = pd.to_datetime(wrf.iloc[:,:4])

date_end = datetime.now()
start_date = date_end - timedelta(days=7)
#start_date = datetime(2020,7,15)
#date_end = datetime(2020,7,22)

idxs = np.array(list(map(lambda x: start_date <= x < date_end, labmim.index)))

#wrfidxWrf = np.array(list(map(lambda x: start_date <= x < date_end, wrf.index)))

#Radiação Solar (sw_dw)
fig = plt.figure(1)
ax = fig.add_subplot(111)

ax.plot(labmim.loc[idxs,:].index, labmim.loc[idxs,'CM3Up_Wm2_Avg'], 'o', color='yellow', markersize=6, label='Média 5 min')
ax.plot(labmim.loc[idxs,:].index, labmim.loc[idxs,'CMP21_Wm2_Avg'], 'o', color='silver', markersize=6, label='Média 5 min')

ax.plot(new.index, new['CM3Up_Wm2_Avg'], '-vr', label='SW_dw 1h')
ax.plot(new.index, new['CMP21_Wm2_Avg'], '-db', label='SW_df 1h')

#wrfax.plot(wrf.loc[idxWrf,:].index, wrf.loc[idxWrf,'Sw_dw'], '--', color='black', label='SW_dw-wrf 1h')

ax.set_ylim(0, 1360)
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d - %b"))
ax.xaxis.set_minor_locator(mdates.HourLocator(np.arange(0,25,6)))

plt.ylabel('Radiação Solar (W/m²)', fontsize=12, horizontalalignment='center', verticalalignment='center')
ax.text(1., .95, date_end.strftime("%Y-%m-%d %H:%M"), fontsize=10, color='black', horizontalalignment='right', transform =ax.transAxes)
ax.text(0.5, 0.55, 'LabMiM & LaPO (IF)', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform =ax.transAxes)
ax.text(0.5, 0.45, 'UFBA', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform =ax.transAxes)
ax.legend(bbox_to_anchor=(0., 1., 1., .1), loc=3, ncol=4, mode="expand", borderaxespad=0.)
ax.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.5)


plt.savefig(path_fig+'radiacao_difusa.png')
#plt.show()

#Balanço(Lw)
fig = plt.figure(2)
ax = fig.add_subplot(111)
ax.plot(labmim.loc[idxs,:].index, labmim.loc[idxs,'CG3Up_Wm2Cr_Avg'], 'o', color='silver', markersize=6)
ax.plot(labmim.loc[idxs,:].index, labmim.loc[idxs,'CM3Up_Wm2_Avg'], 'o', color='yellow', markersize=6)
ax.plot(labmim.loc[idxs,:].index, -labmim.loc[idxs,'CG3Dn_Wm2Cr_Avg'], 'o', color='silver', markersize=6)
ax.plot(labmim.loc[idxs,:].index, -labmim.loc[idxs,'CM3Dn_Wm2_Avg'], 'o', color='silver', markersize=6)


ax.plot(new.index, new['CG3Up_Wm2Cr_Avg'], 'p-', color='black', label='LW_dw')
ax.plot(new.index, -new['CG3Dn_Wm2Cr_Avg'], 'p-', color='orange', label='LW_up')
ax.plot(new.index, new['CM3Up_Wm2_Avg'], 'p-', color='red', label='SW_dw')
ax.plot(new.index, -new['CM3Dn_Wm2_Avg'], 'p-', color='blue', label='SW_up')


ax.set_ylim(-750, 1200)
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d - %b"))
ax.xaxis.set_minor_locator(mdates.HourLocator(np.arange(0,25,6)))

plt.ylabel('Balanço de Radiação (W/m²)', fontsize=12, horizontalalignment='center', verticalalignment='center')
ax.text(1., .95, date_end.strftime("%Y-%m-%d %H:%M"), fontsize=10, color='black', horizontalalignment='right', transform =ax.transAxes)
ax.text(0.5, 0.55, 'LabMiM & LaPO (IF)', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform =ax.transAxes)
ax.text(0.5, 0.45, 'UFBA', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform =ax.transAxes)
ax.legend(bbox_to_anchor=(0., 1., 1., .1), loc=3, ncol=4, mode="expand", borderaxespad=0.)
ax.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.5)

plt.savefig(path_fig+'balanco.png')
#plt.show()

#Radiação Short Wave Par
fig = plt.figure(3)
ax = fig.add_subplot(111)
ax.plot(labmim.loc[idxs,:].index, labmim.loc[idxs,'PAR_Wm2_Avg'], 'o', color='silver', markersize=6, label='Média 5 min')
ax.plot(new.index, new['PAR_Wm2_Avg'], '-*g', label='Média 1 h')

ax.set_ylim(0, 500)
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d - %b"))
ax.xaxis.set_minor_locator(mdates.HourLocator(np.arange(0,25,6)))

plt.ylabel('Radiação PAR (W/m²)', fontsize=12, horizontalalignment='center', verticalalignment='center')
ax.text(1., .95, date_end.strftime("%Y-%m-%d %H:%M"), fontsize=10, color='black', horizontalalignment='right', transform =ax.transAxes)
ax.text(0.5, 0.55, 'LabMiM & LaPO (IF)', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform = ax.transAxes)
ax.text(0.5, 0.45, 'UFBA', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform =ax.transAxes)
ax.legend(bbox_to_anchor=(0., 1., 1., .1), loc=3, ncol=2, mode="expand", borderaxespad=0.)
ax.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.5)

plt.savefig(path_fig+'radiacao_par.png')
#plt.show()

#Temperatura do Ar
fig = plt.figure(4)
ax = fig.add_subplot(111)
ax.plot(labmim.loc[idxs,:].index, labmim.loc[idxs,'Temp_WXT_Avg'], 'o', color='silver', markersize=6, label='Média 5 min')
ax.plot(labmim.loc[idxs,:].index, labmim.loc[idxs,'Temp1_Avg'], 'o', color='silver', markersize=6)

ax.plot(new.index, new['Temp_WXT_Avg'], '^-g', label='WXT 1h')
ax.plot(new.index, new['Temp1_Avg'], '^-r', label='CS215 1h')

#wrfax.plot(wrf.loc[idxWrf,:].index, wrf.loc[idxWrf,'T'], '--', color='black', label='wrf 1h')

ax.set_ylim(18, 32)
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d - %b"))
ax.xaxis.set_minor_locator(mdates.HourLocator(np.arange(0,25,6)))

plt.ylabel('Temperatura do Ar (°C)', fontsize=12, horizontalalignment='center', verticalalignment='center')
ax.text(1., .95, date_end.strftime("%Y-%m-%d %H:%M"), fontsize=10, color='black', horizontalalignment='right', transform =ax.transAxes)
ax.text(0.5, 0.55, 'LabMiM & LaPO (IF)', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform = ax.transAxes)
ax.text(0.5, 0.45, 'UFBA', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform =ax.transAxes)
ax.legend(bbox_to_anchor=(0., 1., 1., .1), loc=3, ncol=3, mode="expand", borderaxespad=0.)
ax.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.5)

plt.savefig(path_fig+'temperatura.png')
#plt.show()

#umidade Relativa
fig = plt.figure(5)
ax = fig.add_subplot(111)
ax.plot(labmim.loc[idxs,:].index, labmim.loc[idxs,'RH_WXT_Avg']+10.339, 'o', color='silver', markersize=6, label='Média 5 min')
ax.plot(labmim.loc[idxs,:].index, labmim.loc[idxs,'RH1_Avg'], 'o', color='silver', markersize=6)

ax.plot(new.index, new['RH_WXT_Avg']+10.339, 's-b', label='WXT 1h')
ax.plot(new.index, new['RH1_Avg'], 's-r', label='CS215 1h')

#wrfax.plot(wrf.loc[idxWrf,:].index, wrf.loc[idxWrf,'ur'], '--',  color='black', label='wrf 1h')

ax.set_ylim(50, 100)
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d - %b"))
ax.xaxis.set_minor_locator(mdates.HourLocator(np.arange(0,25,6)))

plt.ylabel('Umidade Relativa do Ar (%)', fontsize=12, horizontalalignment='center', verticalalignment='center')
ax.text(.88, .05, date_end.strftime("%Y-%m-%d %H:%M"), fontsize=10, color='black', horizontalalignment='center', transform =ax.transAxes)
ax.text(0.5, 0.55, 'LabMiM & LaPO (IF)', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform = ax.transAxes)
ax.text(0.5, 0.45, 'UFBA', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform =ax.transAxes)
ax.legend(bbox_to_anchor=(0., 1., 1., .1), loc=3, ncol=3, mode="expand", borderaxespad=0.)
ax.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.5)

plt.savefig(path_fig+'umidade.png')
#plt.show()

#Pressao
fig = plt.figure(6)
ax = fig.add_subplot(111)
ax.plot(labmim.loc[idxs,:].index, labmim.loc[idxs,'Pmb_WXT'], 'o', color='silver', markersize=6, label='Média 5 min')
ax.plot(new.index, new['Pmb_WXT'], 's-b', label='Média 1h')

#wrfax.plot(wrf.loc[idxWrf,:].index, wrf.loc[idxWrf,'pressure'], '--', color='black', label='wrf 1h')

ax.set_ylim(1000, 1030)
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d - %b"))
ax.xaxis.set_minor_locator(mdates.HourLocator(np.arange(0,25,6)))

plt.ylabel('Pressão Atmosférica (hPa)', fontsize=12, horizontalalignment='center', verticalalignment='center')
ax.text(1., .95, date_end.strftime("%Y-%m-%d %H:%M"), fontsize=10, color='black', horizontalalignment='right', transform =ax.transAxes)
ax.text(0.5, 0.55, 'LabMiM & LaPO (IF)', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform = ax.transAxes)
ax.text(0.5, 0.45, 'UFBA', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform =ax.transAxes)
ax.legend(bbox_to_anchor=(0., 1., 1., .1), loc=3, ncol=3, mode="expand", borderaxespad=0.)
ax.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.5)

plt.savefig(path_fig+'pressao.png')
#plt.show()

#Velocidade do Ar
fig = plt.figure(7)
ax = fig.add_subplot(111)

ax.plot(labmim.loc[idxs,:].index, labmim.loc[idxs,'WS_WXT_Avg'], 'o', color='silver', markersize=6, label='Média 5 min')
ax.plot(new.index, new['WS_WXT_Avg'], '-*k', label='WXT 1h')

#wrfax.plot(wrf.loc[idxWrf,:].index, wrf.loc[idxWrf,'WS'], '--', color='black', label='wrf 1h')

ax.set_ylim(0, 10)
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d - %b"))
ax.xaxis.set_minor_locator(mdates.HourLocator(np.arange(0,25,6)))

plt.ylabel('Velocidade do Vento (m/s)', fontsize=12, horizontalalignment='center', verticalalignment='center')
ax.text(1., .95, date_end.strftime("%Y-%m-%d %H:%M"), fontsize=10, color='black', horizontalalignment='right', transform =ax.transAxes)
ax.text(0.5, 0.55, 'LabMiM & LaPO (IF)', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform = ax.transAxes)
ax.text(0.5, 0.45, 'UFBA', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform =ax.transAxes)
ax.legend(bbox_to_anchor=(0., 1., 1., .1), loc=3, ncol=3, mode="expand", borderaxespad=0.)
ax.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.5)

plt.savefig(path_fig+'velocidade.png')
#plt.show()

#Direção do Ar
fig = plt.figure(8)
ax = fig.add_subplot(111)

ax.plot(labmim.loc[idxs,:].index, labmim.loc[idxs,'WD_WXT_Avg'], 'o', color='silver', markersize=6, label='Média 5 min')
ax.plot(new.index, new['WD_WXT_Avg'], '*k', label='WXT 1h')

#wrfax.plot(wrf.loc[idxWrf,:].index, wrf.loc[idxWrf,'WD'], '--', color='black', label='wrf 1h')

ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d - %b"))
ax.xaxis.set_minor_locator(mdates.HourLocator(np.arange(0,25,6)))

plt.ylabel('Direção do Vento (°)', fontsize=12, horizontalalignment='center', verticalalignment='center')
ax.text(1., .95, date_end.strftime("%Y-%m-%d %H:%M"), fontsize=10, color='black', horizontalalignment='right', transform =ax.transAxes)
ax.text(0.5, 0.55, 'LabMiM & LaPO (IF)', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform =ax.transAxes)
ax.text(0.5, 0.45, 'UFBA', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform =ax.transAxes)
ax.legend(bbox_to_anchor=(0., 1., 1., .1), loc=3, ncol=3, mode="expand", borderaxespad=0.)
ax.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.5)

plt.savefig(path_fig+'direcao.png')
#plt.show()

#Precipitação
fig = plt.figure(9)
ax = fig.add_subplot(111)
ax.plot(labmim_rain.loc[idxs_r,:].index, labmim_rain.loc[idxs_r, rain_col], '-', label='Acumulada 5 min', color='grey', lw=2) # sem xticks
ax.plot(new.index, new[rain_col], 'o', color='blue', markersize=3, label='Acumulada 1h') # sem xticks
ax.set_ylim(0, 30)
#ax.plot(precSum, '.-b', label='Acumulada 5 min') # com xticks
#plt.xticks(indexes, datesTicks[indexes]) # com xticks
plt.ylabel('Precipitação (mm)',fontsize=12, horizontalalignment='center',verticalalignment='center')
ax.text(1., .95, date_end.strftime("%Y-%m-%d %H:%M"), fontsize=10, color='black', horizontalalignment='right', transform =ax.transAxes)
ax.xaxis.set_major_locator(mdates.DayLocator()) # sem xticks
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d - %b")) # sem xticks
ax.xaxis.set_minor_locator(mdates.HourLocator(np.arange(0,25,6))) # sem xticks

ax.text(0.5, 0.55, 'LabMiM & LaPO (IF)', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform = ax.transAxes)
ax.text(0.5, 0.45, 'UFBA', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform =ax.transAxes)
ax.legend(bbox_to_anchor=(0., 1., 1., .1), loc=1, ncol=3, mode="expand", borderaxespad=0.)
ax.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.5)
plt.savefig(path_fig+'precipitacao.png')
#plt.show()
