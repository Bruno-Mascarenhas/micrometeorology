import numpy as np
import math
from datetime import datetime
from datetime import timedelta
from matplotlib.dates import date2num
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import os

#Definindo Variáveis
#__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
#csv = np.genfromtxt(os.path.join(__location__, 'LBMUFBA_lenta.dat'))
csv = np.genfromtxt('E:/LABMIM_UFBA/formatado/LBMUFBA_lenta_2019_0531_total.dat')
csv2 = np.genfromtxt('E:/LABMIM_UFBA/formatado/LBMUFBA_lenta_2019.dat')

dates = np.array([datetime(int(x[0]), int(x[1]), int(x[2]), int(x[3]), int(x[4]), int(x[5])) for x in csv.tolist()])
dates2 = np.array([datetime(int(x[0]), int(x[1]), int(x[2]), int(x[3]), int(x[4]), int(x[5])) for x in csv2.tolist()])

dates = np.append( dates, dates2)

sw_dw1  = np.array([x[14] for x in csv.tolist()]); sw_dw2  = np.array([x[14] for x in csv2.tolist()])
sw_up1 = np.array([x[16] for x in csv.tolist()]); sw_up2 = np.array([x[16] for x in csv2.tolist()])
sw_df1  = np.array([x[23] for x in csv.tolist()]);sw_df2  = np.array([x[23] for x in csv2.tolist()])
lw_dw1  = np.array([x[19] for x in csv.tolist()]);lw_dw2  = np.array([x[19] for x in csv2.tolist()])
lw_up1 = np.array([x[20] for x in csv.tolist()]); lw_up2 = np.array([x[20] for x in csv2.tolist()])
rn11     = np.array([x[26] for x in csv.tolist()]); rn12     = np.array([x[26] for x in csv2.tolist()])
rn21 = np.array([x[21] for x in csv.tolist()]); rn22 = np.array([x[21] for x in csv2.tolist()])
sw_par1 = np.array([x[28] for x in csv.tolist()]); sw_par2 = np.array([x[28] for x in csv2.tolist()])
temp11 = np.array([x[29] for x in csv.tolist()]); temp12 = np.array([x[29] for x in csv2.tolist()])
ur11 = np.array([x[30] for x in csv.tolist()]); ur12 = np.array([x[30] for x in csv2.tolist()])

vel21 = np.array([x[34] for x in csv.tolist()]); vel22 = np.array([x[31] for x in csv2.tolist()])
dir21 = np.array([x[35] for x in csv.tolist()]); dir22 = np.array([x[32] for x in csv2.tolist()])
temp21 = np.array([x[36] for x in csv.tolist()]); temp22 = np.array([x[33] for x in csv2.tolist()])
ur21 = np.array([x[37]+10.578 for x in csv.tolist()]); ur22 = np.array([x[34]+10.578 for x in csv2.tolist()])
compHor21 = np.array([-x[34]*math.sin(math.radians(x[35])) for x in csv.tolist()]); compHor22 = np.array([-x[31]*math.sin(math.radians(x[32])) for x in csv2.tolist()])
compMer21 =np.array([-x[34]*math.cos(math.radians(x[35])) for x in csv.tolist()]); compMer22 =np.array([-x[31]*math.cos(math.radians(x[32])) for x in csv2.tolist()])
pressao1 = np.array([x[38] for x in csv.tolist()]); pressao2 = np.array([x[35] for x in csv2.tolist()])

sw_dw  = np.append(sw_dw1 , sw_dw2)
sw_up = np.append(sw_up1 , sw_up2)
sw_df = np.append(sw_df1 , sw_df2)
lw_dw = np.append(lw_dw1 , lw_dw2)
lw_up = np.append(lw_up1 , lw_up2)
rn1 = np.append(rn11 , rn12)
rn2 = np.append(rn21 , rn22)
sw_par = np.append(sw_par1 , sw_par2)
temp1 = np.append(temp11 , temp12)
ur1 = np.append(ur11 , ur12)
vel2 = np.append(vel21 , vel22)
dir2 = np.append(dir21 , dir22)
temp2 = np.append(temp21 , temp22)
ur2 = np.append(ur21 , ur22)
compHor2 = np.append(compHor21 , compHor22)
compMer2 = np.append(compMer21 , compMer22)
pressao = np.append(pressao1 , pressao2)

#Retirando o -999.999
sw_dw[sw_dw == -999.999] = np.nan
lw_dw[lw_dw == -999.999] = np.nan
rn1[rn1 == -999.999] = np.nan
rn2[rn2 == -999.999] = np.nan
sw_par[sw_par == -999.999] = np.nan
temp1[temp1 == -999.999] = np.nan
ur1[ur1 == -999.999] = np.nan
pressao[pressao == -999.999] = np.nan
vel2[vel2 == -999.999] = np.nan
dir2[dir2 == -999.999] = np.nan
temp2[temp2 == -999.999] = np.nan
ur2[ur2 == -999.999] = np.nan
compHor2[compHor2 == -999.999] = np.nan
compMer2[compMer2 == -999.999] = np.nan


#Data de inicio / final do que vai plotar
""" Data arbitrária
date_start = datetime(2017, 1, 20)
date_end = datetime(2017, 1, 27)
"""

#""" Data Automática
date_end = datetime.now()
date_start = date_end - timedelta(days=7)
#"""

#Varre o array pegando só o necessário de acordo a data selecionada
indexes = np.array(list(map(lambda x: date_start <= x <= date_end, dates)))

delta_date = dates[indexes]
delta_sw_dw = sw_dw[indexes]
delta_sw_up = sw_up[indexes]
delta_sw_df = sw_df[indexes]
delta_lw_dw = lw_dw[indexes]
delta_lw_up = lw_up[indexes]
delta_rn1 = rn1[indexes]
delta_rn2 = rn2[indexes]
delta_sw_par = sw_par[indexes]
delta_temp1 = temp1[indexes]
delta_ur1 = ur1[indexes]

delta_pressao = pressao[indexes]
delta_vel2 = vel2[indexes]
delta_dir2 = dir2[indexes]
delta_temp2 = temp2[indexes]
delta_ur2 = ur2[indexes]
delta_compHor2 = compHor2[indexes]
delta_compMer2 = compMer2[indexes]

sw_dwMean = []
sw_upMean = []
sw_dfMean = []
lw_dwMean = []
lw_upMean = []
rn1Mean = []
rn2Mean = []
sw_parMean = []
temp1Mean = []
ur1Mean = []
vel1Mean = []
dir1Mean = []
compHor1Mean = []
compMer1Mean = []
datesTicks = []
datesPlot = []

pressaoMean = []
vel2Mean = []
dir2Mean = []
temp2Mean = []
ur2Mean = []
compHor2Mean = []
compMer2Mean = []

current_date = date_start
current_date = current_date.replace(minute=30)
#Calculando médias horárias
while current_date <= date_end:

    datep = current_date.replace(minute=59)
    datesTicks.append(datep.strftime("%d - %b"))
    next_date = current_date + timedelta(hours=1)

    current_hour = current_date.hour
    iterator = current_date

    count_hour = 0


    #Confere a quantidade de valores no horário
    while current_hour == iterator.hour:
        count_hour += 1
        iterator = iterator + timedelta(minutes=5)

    datesPlot.append(datep)

    #Calcula médias horárias de houver determinada quantidade mínima nas horas
    if count_hour >= 6:
        indexes = np.array(list(map(lambda x: current_date <= x < next_date, delta_date.tolist())))
        sw_dwMean.append(np.nanmean(delta_sw_dw[indexes]))
        sw_upMean.append(np.nanmean(delta_sw_up[indexes]))
        sw_dfMean.append(np.nanmean(delta_sw_df[indexes]))
        lw_dwMean.append(np.nanmean(delta_lw_dw[indexes]))
        lw_upMean.append(np.nanmean(delta_lw_up[indexes]))
        rn1Mean.append(np.nanmean(delta_rn1[indexes]))
        rn2Mean.append(np.nanmean(delta_rn2[indexes]))
        sw_parMean.append(np.nanmean(delta_sw_par[indexes]))
        temp1Mean.append(np.nanmean(delta_temp1[indexes]))
        ur1Mean.append(np.nanmean(delta_ur1[indexes]))

        pressaoMean.append(np.nanmean(delta_pressao[indexes]))
        vel2Mean.append(np.nanmean(delta_vel2[indexes]))
        compHor2Mean.append(np.nanmean(delta_compHor2[indexes]))
        compMer2Mean.append(np.nanmean(delta_compMer2[indexes]))
        alfa = np.arctan2(np.nanmean(delta_compMer2[indexes]), np.mean(delta_compHor2[indexes]))
        dir2Mean.append(np.fmod(3*(np.pi/2) - alfa, 2*np.pi)*(180/np.pi))
        temp2Mean.append(np.nanmean(delta_temp2[indexes]))
        ur2Mean.append(np.nanmean(delta_ur2[indexes]))

    else:
        sw_dwMean.append(np.nan)
        sw_upMean.append(np.nan)
        sw_dfMean.append(np.nan)
        lw_dwMean.append(np.nan)
        lw_upMean.append(np.nan)
        rn1Mean.append(np.nan)
        rn2Mean.append(np.nan)
        sw_parMean.append(np.nan)
        temp1Mean.append(np.nan)
        ur1Mean.append(np.nan)

        pressaoMean.append(np.nan)
        vel2Mean.append(np.nan)
        compHor2Mean.append(np.nan)
        compMer2Mean.append(np.nan)
        dir2Mean.append(np.nan)
        temp2Mean.append(np.nan)
        ur2Mean.append(np.nan)

    current_date = next_date


#Array de médias horárias
datesTicks = np.array(datesTicks)
sw_dwMean = np.array(sw_dwMean)
sw_upMean = np.array(sw_upMean)
sw_dfMean = np.array(sw_dfMean)
lw_dwMean = np.array(lw_dwMean)
lw_upMean = np.array(lw_upMean)
rn1Mean = np.array(rn1Mean)
rn2Mean = np.array(rn2Mean)
sw_parMean = np.array(sw_parMean)
temp1Mean = np.array(temp1Mean)
ur1Mean = np.array(ur1Mean)
compHor2Mean = np.array(compHor2Mean)
compMer2Mean = np.array(compMer2Mean)

pressaoMean = np.array(pressaoMean)
vel2Mean = np.array(vel2Mean)
compHor2Mean = np.array(compHor2Mean)
compMer2Mean = np.array(compMer2Mean)
dir2Mean = np.array(dir2Mean)
temp2Mean = np.array(temp2Mean)
ur2Mean = np.array(ur2Mean)

datesPlot = date2num(datesPlot)

"""
#Controle de qualidade das médias
sw_dwMean[sw_dwMean < -10] = np.nan
lw_dwMean[lw_dwMean < 0] = np.nan
rnMean[rnMean < -200] = np.nan
sw_parMean[sw_parMean < -10] = np.nan
temp1Mean[temp1Mean < 0] = np.nan
ur1Mean[ur1Mean < 0] = np.nan
vel1Mean[vel1Mean < 0] = np.nan
dir1Mean[dir1Mean < 0] = np.nan
"""
#######################################################################################3
# gerando figuras
#######################################################################################

#indexes = range(0,len(sw_dwMean),720)
indexes = range(0, len(sw_dwMean), 24)
ind = np.array(list(map(lambda x: date_start <= x <= date_end, dates)))

#Tamanho padrão das imagens
plt.rcParams['figure.figsize'] = 8, 3

#Radiacao liquida
fig = plt.figure(15)
ax = fig.add_subplot(111)

ax.plot(dates[ind], delta_rn1, 'o', color='silver', markersize=6, label='Média 5 min')
ax.plot(dates[ind], delta_rn2, 'o', color='silver', markersize=6)

ax.plot(datesPlot, rn1Mean, '-vr', label='RN1 1h')
ax.plot(datesPlot, rn2Mean, '-db', label='RN2 1h')

ax.set_ylim(-200, 800)
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d - %b"))
ax.xaxis.set_minor_locator(mdates.HourLocator(np.arange(0,25,6)))

plt.ylabel('Radiação Líquida (W/m²)', fontsize=12, horizontalalignment='center', verticalalignment='center')
ax.text(0.5, 0.55, 'LabMiM & LaPO (IF) / LMAC (IM)', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform =ax.transAxes)
ax.text(0.5, 0.45, 'UFBA', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform =ax.transAxes)
ax.legend(bbox_to_anchor=(0., 1., 1., .1), loc=3, ncol=4, mode="expand", borderaxespad=0.)
ax.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.5)


plt.savefig('E:/LABMIM_UFBA/formatado/figuras/radiacao_liq.png')
#fig.show()

#Radiação Solar (sw_dw)
fig = plt.figure(1)
ax = fig.add_subplot(111)

ax.plot(dates[ind], delta_sw_dw, 'o', color='yellow', markersize=6, label='Média 5 min')
ax.plot(dates[ind], delta_sw_df, 'o', color='silver', markersize=6, label='Média 5 min')

ax.plot(datesPlot, sw_dwMean, '-vr', label='SW_dw 1h')
ax.plot(datesPlot, sw_dfMean, '-db', label='SW_df 1h')

ax.set_ylim(0, 1360)
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d - %b"))
ax.xaxis.set_minor_locator(mdates.HourLocator(np.arange(0,25,6)))

plt.ylabel('Radiação Solar (W/m²)', fontsize=12, horizontalalignment='center', verticalalignment='center')
ax.text(0.5, 0.55, 'LabMiM & LaPO (IF) / LMAC (IM)', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform =ax.transAxes)
ax.text(0.5, 0.45, 'UFBA', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform =ax.transAxes)
ax.legend(bbox_to_anchor=(0., 1., 1., .1), loc=3, ncol=4, mode="expand", borderaxespad=0.)
ax.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.5)


plt.savefig('E:/LABMIM_UFBA/formatado/figuras/radiacao_difusa.png')
#fig.show()


#Balanço(Lw)
fig = plt.figure(2)
ax = fig.add_subplot(111)
ax.plot(dates[ind], delta_lw_dw, 'o', color='silver', markersize=6)
ax.plot(dates[ind], delta_sw_dw, 'o', color='yellow', markersize=6)
ax.plot(dates[ind], -delta_lw_up, 'o', color='silver', markersize=6)
ax.plot(dates[ind], -delta_sw_up, 'o', color='silver', markersize=6)

ax.plot(datesPlot, lw_dwMean, 'p-', color='black', label='LW_dw')
ax.plot(datesPlot, -lw_upMean, 'p-', color='orange', label='LW_up')
ax.plot(datesPlot, sw_dwMean, 'p-', color='red', label='SW_dw')
ax.plot(datesPlot, -sw_upMean, 'p-', color='blue', label='SW_up')

ax.set_ylim(-750, 1200)
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d - %b"))
ax.xaxis.set_minor_locator(mdates.HourLocator(np.arange(0,25,6)))

plt.ylabel('Balanço de Radiação (W/m²)', fontsize=12, horizontalalignment='center', verticalalignment='center')
ax.text(0.5, 0.55, 'LabMiM & LaPO (IF) / LMAC (IM)', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform =ax.transAxes)
ax.text(0.5, 0.45, 'UFBA', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform =ax.transAxes)
ax.legend(bbox_to_anchor=(0., 1., 1., .1), loc=3, ncol=4, mode="expand", borderaxespad=0.)
ax.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.5)

plt.savefig('E:/LABMIM_UFBA/formatado/figuras/balanco.png')
#fig.show()


#Saldo Radiação (lw_dw)
#fig = plt.figure(3)
#ax = fig.add_subplot(111)
#ax.plot(dates[ind], delta_rn, 'o', color='silver', markersize=6, label='Média 5 min')
#ax.plot(datesPlot, rnMean, '-db', label='Média 1 h')

#ax.set_ylim(-200, 1000)
#ax.xaxis.set_major_locator(mdates.DayLocator())
#ax.xaxis.set_major_formatter(mdates.DateFormatter("%d - %b"))
#ax.xaxis.set_minor_locator(mdates.HourLocator(np.arange(0,25,6)))

#plt.ylabel('Saldo de Radiação (W/m²)', fontsize=12, horizontalalignment='center', verticalalignment='center')
#ax.text(0.5, 0.55, 'LabMiM & LaPO (IF) / LMAC (IM)', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform =ax.transAxes)
#ax.text(0.5, 0.45, 'UFBA', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform =ax.transAxes)
#ax.legend(bbox_to_anchor=(0., 1., 1., .1), loc=3, ncol=2, mode="expand", borderaxespad=0.)
#ax.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.5)

#plt.savefig('E:/LABMIM_UFBA/formatado/figuras/radiacao_saldo.png')
#fig.show()


#Radiação Short Wave Par
fig = plt.figure(4)
ax = fig.add_subplot(111)
ax.plot(dates[ind], delta_sw_par, 'o', color='silver', markersize=6, label='Média 5 min')
ax.plot(datesPlot, sw_parMean, '-*g', label='Média 1 h')

ax.set_ylim(0, 500)
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d - %b"))
ax.xaxis.set_minor_locator(mdates.HourLocator(np.arange(0,25,6)))

plt.ylabel('Radiação PAR (W/m²)', fontsize=12, horizontalalignment='center', verticalalignment='center')
ax.text(0.5, 0.55, 'LabMiM & LaPO (IF) / LMAC (IM)', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform = ax.transAxes)
ax.text(0.5, 0.45, 'UFBA', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform =ax.transAxes)
ax.legend(bbox_to_anchor=(0., 1., 1., .1), loc=3, ncol=2, mode="expand", borderaxespad=0.)
ax.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.5)

plt.savefig('E:/LABMIM_UFBA/formatado/figuras/radiacao_par.png')
#plt.show()


#Temperatura do Ar
fig = plt.figure(5)
ax = fig.add_subplot(111)
ax.plot(dates[ind], delta_temp1, 'o', color='silver', markersize=6, label='Média 5 min')
ax.plot(dates[ind], delta_temp2, 'o', color='silver', markersize=6)

ax.plot(datesPlot, temp2Mean, '^-g', label='WXT 1h')
ax.plot(datesPlot, temp1Mean, '^-r', label='CS215 1h')

ax.set_ylim(18, 32)
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d - %b"))
ax.xaxis.set_minor_locator(mdates.HourLocator(np.arange(0,25,6)))

plt.ylabel('Temperatura do Ar (°C)', fontsize=12, horizontalalignment='center', verticalalignment='center')
ax.text(1., .95, date_end.strftime("%Y-%m-%d %H:%M"), fontsize=10, color='black', horizontalalignment='right', transform =ax.transAxes)
ax.text(0.5, 0.55, 'LabMiM & LaPO (IF) / LMAC (IM)', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform = ax.transAxes)
ax.text(0.5, 0.45, 'UFBA', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform =ax.transAxes)
ax.legend(bbox_to_anchor=(0., 1., 1., .1), loc=3, ncol=3, mode="expand", borderaxespad=0.)
ax.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.5)

plt.savefig('E:/LABMIM_UFBA/formatado/figuras/temperatura.png')
#plt.show()


#umidade Relativa
fig = plt.figure(6)
ax = fig.add_subplot(111)
ax.plot(dates[ind], delta_ur1, 'o', color='silver', markersize=6, label='Média 5 min')
ax.plot(dates[ind], delta_ur2, 'o', color='silver', markersize=6)

ax.plot(datesPlot, ur2Mean, 's-b', label='WXT 1h')
ax.plot(datesPlot, ur1Mean, 's-r', label='CS215 1h')


ax.set_ylim(50, 100)
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d - %b"))
ax.xaxis.set_minor_locator(mdates.HourLocator(np.arange(0,25,6)))

plt.ylabel('Umidade Relativa do Ar (%)', fontsize=12, horizontalalignment='center', verticalalignment='center')
ax.text(0.5, 0.55, 'LabMiM & LaPO (IF) / LMAC (IM)', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform = ax.transAxes)
ax.text(0.5, 0.45, 'UFBA', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform =ax.transAxes)
ax.legend(bbox_to_anchor=(0., 1., 1., .1), loc=3, ncol=3, mode="expand", borderaxespad=0.)
ax.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.5)

plt.savefig('E:/LABMIM_UFBA/formatado/figuras/umidade.png')
#plt.show()

#Pressao
fig = plt.figure(10)
ax = fig.add_subplot(111)
ax.plot(dates[ind], delta_pressao, 'o', color='silver', markersize=6, label='Média 5 min')

ax.plot(datesPlot, pressaoMean, 's-b', label='Média 1h')


ax.set_ylim(1000, 1030)
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d - %b"))
ax.xaxis.set_minor_locator(mdates.HourLocator(np.arange(0,25,6)))

plt.ylabel('Pressão Atmosférica (hPa)', fontsize=12, horizontalalignment='center', verticalalignment='center')
ax.text(0.5, 0.55, 'LabMiM & LaPO (IF) / LMAC (IM)', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform = ax.transAxes)
ax.text(0.5, 0.45, 'UFBA', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform =ax.transAxes)
ax.legend(bbox_to_anchor=(0., 1., 1., .1), loc=3, ncol=3, mode="expand", borderaxespad=0.)
ax.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.5)

plt.savefig('E:/LABMIM_UFBA/formatado/figuras/pressao.png')
#plt.show()


#Velocidade do Ar
fig = plt.figure(7)
ax = fig.add_subplot(111)
ax.plot(dates[ind], delta_vel2, 'o', color='silver', markersize=6, label='Média 5 min')

ax.plot(datesPlot, vel2Mean, '-*k', label='WXT 1h')

ax.set_ylim(0, 10)
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d - %b"))
ax.xaxis.set_minor_locator(mdates.HourLocator(np.arange(0,25,6)))

plt.ylabel('Velocidade do Vento (m/s)', fontsize=12, horizontalalignment='center', verticalalignment='center')
ax.text(0.5, 0.55, 'LabMiM & LaPO (IF) / LMAC (IM)', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform = ax.transAxes)
ax.text(0.5, 0.45, 'UFBA', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform =ax.transAxes)
ax.legend(bbox_to_anchor=(0., 1., 1., .1), loc=3, ncol=3, mode="expand", borderaxespad=0.)
ax.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.5)

plt.savefig('E:/LABMIM_UFBA/formatado/figuras/velocidade.png')
#plt.show()


#Direção do Ar
fig = plt.figure(8)
ax = fig.add_subplot(111)
ax.plot(dates[ind], delta_dir2, 'o', color='silver', markersize=6, label='Média 5 min')

ax.plot(datesPlot, dir2Mean, '*k', label='WXT 1h')

ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d - %b"))
ax.xaxis.set_minor_locator(mdates.HourLocator(np.arange(0,25,6)))

plt.ylabel('Direção do Vento (°)', fontsize=12, horizontalalignment='center', verticalalignment='center')
ax.text(0.5, 0.55, 'LabMiM & LaPO (IF) / LMAC (IM)', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform =ax.transAxes)
ax.text(0.5, 0.45, 'UFBA', fontsize=16, color='#CFCFCF', horizontalalignment='center', transform =ax.transAxes)
ax.legend(bbox_to_anchor=(0., 1., 1., .1), loc=3, ncol=3, mode="expand", borderaxespad=0.)
ax.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.5)

plt.savefig('E:/LABMIM_UFBA/formatado/figuras/direcao.png')
#plt.show()