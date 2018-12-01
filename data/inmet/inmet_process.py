import numpy as np
import os
from operator import itemgetter
from datetime import datetime
from datetime import timezone

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
file = np.genfromtxt(os.path.join(__location__, 'inmet.dat'),dtype=None,delimiter='')

def getNum(number):
    if len(str(number)) == 1:
        return '0' + str(number)
    else:
        return str(number)

strdate = [x[0].decode("utf-8") for x in file.tolist()]
dates = [datetime.strptime(x,'%d/%m/%Y') for x in strdate]
hours = np.array([x[1] for x in file.tolist()])
temp = np.array([(x[3]+x[4])/2 for x in file.tolist()])
umid = np.array([(x[6]+x[7])/2 for x in file.tolist()])
pto_orvalho = np.array([(x[9]+x[10])/2 for x in file.tolist()])
pressao = np.array([(x[12]+x[13])/2 for x in file.tolist()])
vento_direcao = np.array([x[15] for x in file.tolist()])
vento_vel = np.array([x[14] for x in file.tolist()])
vento_rajada = np.array([x[16] for x in file.tolist()])
radiacao = np.array([x[17]*0.277777 for x in file.tolist()])
precipitacao = np.array([x[18] for x in file.tolist()])

#########################################################
# Conversão de unidades
#########################################################
tempK = np.array([x+273.15 for x in temp])
urFrac = np.array([x/100 for x in umid])
pressaoPa = np.array([x*100 for x in pressao])

for i in range(len(dates)):
    dates[i] = dates[i].replace(hour=hours[i])

dates = [x.replace(tzinfo=timezone.utc).astimezone(tz=None) for x in dates]

ano = np.array([x.year for x in dates])
mes = np.array([x.month for x in dates])
dia = np.array([x.day for x in dates])
hora = np.array([x.hour for x in dates])

lists = []
#es - (pressao saturação) = temp pto de orvalho  |||  e - (pressao do vapor) = temp do ar  ||| q = umidade específica ||
e = []; es = []; q = []; thetaV = []; theta = []
for i in range(len(tempK)):
    es.append(611.2*np.exp((17.67*(tempK[i]-273.15))/((tempK[i]-273.15)+243.5)))
    e.append(urFrac[i] * es[i])
    q.append((0.622*e[i])/((pressaoPa[i])+(0.622-1)*e[i]))
    theta.append(tempK[i] * (101300 / pressaoPa[i])**(287.04/1005.7))
    thetaV.append(theta[i] * (1+0.61*q[i]))

#########################################################
# Conversão de unidades
#########################################################
for i in range(len(es)):
    es[i] = es[i]/100 #hPa
    e[i] = e[i]/100 #hPa
    q[i] = q[i]*1000 #g/kg
    theta[i] = theta[i]-273.15
    thetaV[i] = thetaV[i]-273.15


for i in range(len(dates)):
    toOrd = getNum(ano[i])+getNum(mes[i])+getNum(dia[i])+getNum(hora[i])
    lists.append([int(toOrd),ano[i],mes[i],dia[i],hora[i],temp[i],umid[i],pto_orvalho[i],pressao[i],vento_direcao[i],vento_vel[i],vento_rajada[i],radiacao[i],precipitacao[i],e[i],es[i],q[i],thetaV[i],theta[i]])

lists.sort(key=itemgetter(0))

lists = np.array(lists)

np.savetxt('inmet_output.dat', lists, fmt='%.10g', delimiter=" ")

#temp_inst -> 5
#umid_inst -> 6
#pto_orvalho_inst -> 7
#pressao -> 8
#vento_direcao -> 9
#vento_vel -> 10
#vento_rajada -> 11
#radiacao -> 12
#precipitacao -> 13