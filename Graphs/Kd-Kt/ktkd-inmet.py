import seaborn as sns
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from datetime import datetime, timedelta
from scipy import stats
plt.style.use('seaborn-whitegrid')
from sys import exit 
import scipy
import warnings

#######################################################################################
# pareando inmet com os dados teóricos 
#######################################################################################
inmet = pd.read_csv('/home/labmim/meteorologia/estacoes/RMS/SSA_inmet_completo.dat', sep=' ')
inmet.index = pd.to_datetime(inmet[['year','month','day','hour']])
inmet['kt'] = inmet['sw_dw'] / inmet['oc_topo']

########################################################################################
dados = {}
for x in inmet.columns:
    dados[x] = []

for i, kt in enumerate(inmet['kt']):
    if not np.isnan(kt) and kt > 0:
        for x in inmet.columns:
            dados[x].append(inmet[x][i])

dados = pd.DataFrame(dados)
dados.index = pd.to_datetime(dados['year month day hour'.split()])

#########################################################################
# Calculo do kt diario
##########################################################################
ktday = []
i = 0
datas = dados.index
curr = datas[0]
tmpd, tmpt, days = 1,1,1

while i < len(datas)-1:
    if curr.day == datas[i+1].day:
        tmpd += dados['sw_dw'][i]
        tmpt += dados['oc_topo'][i]
        days+=1
    else:
        for x in range(days):
            ktday.append(tmpd/tmpt)
        tmpd, tmpt, days = dados['sw_dw'][i],dados['oc_topo'][i],1
        curr = datas[i+1]
    i+=1

for x in range(days):
    ktday.append(tmpd/tmpt)

dados['ktday'] = ktday

###########################################################################
# Calculo do psi - Lemos el al. (2017)
##########################################################################
kt, time = dados['kt'],dados.index
psi = []
psi.append(kt[0])

for i in range(1,len(kt)-1):
    now = time[i]
    nextt = time[i] + timedelta(hours=1)
    prev =  time[i] - timedelta(hours=1)
    
    if now.hour == 6:
        if time[i+1] == nextt:
            psi.append(kt[i+1])
        else:
            psi.append(kt[i])
    elif now.hour == 17:
        if time[i-1] == prev:
            psi.append(kt[i-1])
        else:
            psi.append(kt[i])
    else:
        if time[i-1] == prev and time[i+1] == nextt:
            psi.append((kt[i+1]+kt[i-1])/2)
        else:
            psi.append(kt[i])

psi.append(kt[-1])

dados['psi'] = psi

#Lemos et al. (2017)
model1 = []

#Marques Filhos et al. (2016)
model2 = []

#Ridley et al. (2010)
model3 = []

for i in range(len(dados.index)):
    model1.append(1 / (1+ np.exp(-4.41 + 7.87*dados['kt'][i] + -0.088*dados['ast_h'][i] + -0.0049*dados['elev_solar'][i] + 1.47*dados['ktday'][i] + 1.1*dados['psi'][i])))
    model2.append(0.13 + 0.86 * (1/(1+np.exp(-6.29+12.26*dados['kt'][i]))))
    model3.append(1 / (1+ np.exp(-5.38 + 6.63*dados['kt'][i] + -0.006*dados['ast_h'][i] + -0.007*dados['elev_solar'][i] + 1.75*dados['ktday'][i] + 1.31*dados['psi'][i])))

dados['model1'] = model1
dados['model2'] = model2
dados['model3'] = model3

dados['sw_dif1'] = dados['model1']*dados['sw_dw']
dados['sw_dif2'] = dados['model2']*dados['sw_dw']
dados['sw_dif3'] = dados['model3']*dados['sw_dw']

#dados['sw_top'].plot()
#dados['sw_dw'].plot()
#dados['sw_dif3'].plot()
#plt.show()

dados = dados['year;month;day;hour;compHor;compMer;e;fc;kt;precip;pressao;pto_orvalho;q;sw_dw;temp;theta;ur;vento_dir;vento_vel;ang_hor;ang_hor_por;ang_zen;ast_h;decl_rad;elev_solar;nascer_h;oc_topo;por_h;ktday;psi;model1;model2;model3;sw_dif1;sw_dif2;sw_dif3'.split(';')]
print(dados.head())
dados.to_csv('ktkd_SSA_inmet.dat',na_rep='nan',sep=';',index=False)