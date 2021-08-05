import seaborn as sns
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from datetime import datetime, timedelta
from scipy import stats
plt.style.use('seaborn-whitegrid')
from sys import exit 
import scipy

#labmim = pd.read_csv(filepath_or_buffer='/home/labmim/meteorologia/estacoes/rms/labmim_completo_controle.dat',sep=';')
labmim = pd.read_csv('/home/labmim/meteorologia/estacoes/rms/labmim_completo_controle2020.dat',sep=';')
labmim.index = pd.to_datetime(labmim[['year','month','day','hour']])

labmim['Sw_dif'] = labmim['Sw_dif'] * labmim['fc']

n_dif = []
n_sw = []
for dif, sw, precip in zip(labmim['Sw_dif'],labmim['Sw_dw'],labmim['precip']):
    if precip > 0:
        n_dif.append(np.nan)
        n_sw.append(np.nan)
    else:
        n_dif.append(dif)
        n_sw.append(sw)

labmim['Sw_dif'] = n_dif
labmim['Sw_dw'] = n_sw


###############################################################################
# Alterações calibração #######################################################
"""
#antiga dif = 1.1074
#labmim['Sw_dif'] = labmim['Sw_dif'] * 1.1009        #fator constante de calibração
labmim['Sw_dif'] = labmim['Sw_dif'] * 1.0933        #fator constante de calibração  8.37/7.66

#antiga sw_dw = 0.9983
#labmim['Sw_dw'] = labmim['Sw_dw'] * 0.9983
tmp = labmim.loc['2018':'2018-09','Sw_dw']
tmp *= 0.9383 #constante corrigida de calibração    8.37/8.92
labmim.loc['2018':'2018-09','Sw_dw'] = tmp

#cnr
tmp = labmim.loc['2018-09':'2019-01','Sw_dw']
tmp *= 0.9694 #constante corrigida de calibração    10.46/10.79
labmim.loc['2018-09':'2019-01','Sw_dw'] = tmp

#cnr 2019
tmp = labmim.loc['2019':'2020','Sw_dw']
tmp *= 1.006 #constante corrigida de calibração    10.46/10.40
labmim.loc['2019':'2020','Sw_dw'] = tmp
"""
################################################################################
################################################################################

labmim['kt'] = labmim['Sw_dw']/labmim['oc_topo']

labmim['kd'] = labmim['Sw_dif'] / labmim['Sw_dw']

ktf = []
kdf = []
porf = []
nascerf = []
timef = []
ast_hf = []
elev_solarf = []
ktdayf = []
dw = []
top = []
diff = []

for kt, kd, ast_h, elev_solar, time, sw_dw, sw_top, dif in zip(labmim['kt'],labmim['kd'],labmim['ast_h'],labmim['elev_solar'],labmim.index,labmim['Sw_dw'],labmim['oc_topo'], labmim['Sw_dif']):
    if not np.isnan(kt) and not np.isnan(kd) and kd <= 1.2:
        kdf.append(kd)
        ktf.append(kt)
        ast_hf.append(ast_h)
        elev_solarf.append(elev_solar)
        timef.append(time)
        dw.append(sw_dw)
        top.append(sw_top)
        diff.append(dif)

dados = pd.DataFrame({'kt':ktf,'kd':kdf,'elev':elev_solarf,'ast':ast_hf,'sw_dw':dw,'sw_top':top,'sw_dif':diff})
dados.index = timef

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
        tmpt += dados['sw_top'][i]
        days+=1
    else:
        for x in range(days):
            ktday.append(tmpd/tmpt)
        tmpd, tmpt, days = dados['sw_dw'][i],dados['sw_top'][i],1
        curr = datas[i+1]
    i+=1

for x in range(days):
    ktday.append(tmpd/tmpt)

dados['ktday'] = ktday

###########################################################################
# Calculo do psi - Lemos el al. (2017)
##########################################################################
kt, kd, time = dados['kt'],dados['kd'],dados.index
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
#########################################################################

#Lemos et al. (2017)
model1 = []

#Marques Filhos et al. (2016)
model2 = []

#Ridley et al. (2010)
model3 = []

#########################################################################
# Calculo dos modelos
##########################################################################
for i in range(len(dados.index)):
    model1.append(1 / (1+ np.exp(-4.41 + 7.87*dados['kt'][i] + -0.088*dados['ast'][i] + -0.0049*dados['elev'][i] + 1.47*dados['ktday'][i] + 1.1*dados['psi'][i])))
    model2.append(0.13 + 0.86 * (1/(1+np.exp(-6.29+12.26*dados['kt'][i]))))
    model3.append(1 / (1+ np.exp(-5.38 + 6.63*dados['kt'][i] + -0.006*dados['ast'][i] + -0.007*dados['elev'][i] + 1.75*dados['ktday'][i] + 1.31*dados['psi'][i])))

dados['model_lemos'] = model1
dados['model_marquesfh'] = model2
dados['model_ridley'] = model3

dados['year'] = dados.index.year
dados['month'] = dados.index.month 
dados['day'] = dados.index.day 
dados['hour'] = dados.index.hour

##############################################################
# Plot das medidas e modelos de KT-KD
##############################################################
plt.scatter(x=dados['kt'],y=dados['kd'],label='medidas')
plt.scatter(y=dados['model_lemos'],x=dados['kt'],label='model_lemos')
plt.scatter(y=dados['model_marquesfh'],x=dados['kt'],label='model_marquesfh')
plt.scatter(y=dados['model_ridley'],x=dados['kt'],label='model_ridley')
plt.ylabel('Kd')
plt.xlabel('Kt')
plt.legend()
#plt.savefig('ktkdmodel.png',dpi=300,bbox_inches='tight')
plt.show()

################################################################



#Organização e exportação da tabela dos dados usados
dados = dados['year month day hour sw_dw sw_dif sw_top kt ktday kd ast elev psi model_lemos model_marquesfh model_ridley'.split()]
dados.to_csv('ktkd_labmim.dat',na_rep='nan',sep=';',index=False)