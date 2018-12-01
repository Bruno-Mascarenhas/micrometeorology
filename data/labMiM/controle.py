import numpy as np
import math
from datetime import datetime
from datetime import timedelta
from matplotlib.dates import date2num
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.mlab as mlab
from scipy.stats import norm
import os
import plott

#File 1 = lenta || File 2 = precipitacao || File 3 = radiacao topo (teorica)
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
file1 = np.genfromtxt(os.path.join(__location__, 'LBMUFBA_lenta_all.dat'))
file3 = np.genfromtxt(os.path.join(__location__, 'LMBUFBA_sw_teorica.dat'))
file2 = np.genfromtxt(os.path.join(__location__, 'LBMUFBA_rain_all.dat'))


##########################
#Array das Datas brutas  #
##########################
dateslenta = np.array([datetime(int(x[0]), int(x[1]), int(x[2]), int(x[3]), int(x[4]), int(x[5])) for x in file1.tolist()])
datesrain = np.array([datetime(int(x[0]), int(x[1]), int(x[2]), int(x[3]), int(x[4]), int(x[5])) for x in file2.tolist()])
datesteorico = np.array([datetime(int(x[4]), int(x[5]), int(x[6]), int(x[7]), int(x[8])) for x in file3.tolist()])


######################################################
radTeo = np.array([x[10] for x in file3.tolist()])
######################################################
precip = np.array([x[7] for x in file2.tolist()])
######################################################
radSw = np.array([x[10] for x in file1.tolist()])
atmLw = np.array([x[12] for x in file1.tolist()])
balRad = np.array([x[17] for x in file1.tolist()])
parSw = np.array([x[19] for x in file1.tolist()])
tempAr = np.array([x[20] for x in file1.tolist()])
humRel = np.array([x[21] for x in file1.tolist()])
velAr = np.array([x[22] for x in file1.tolist()])
dirAr = np.array([x[23] for x in file1.tolist()])
compHor = np.array([-x[22]*math.sin(math.radians(x[23])) for x in file1.tolist()])
compMer = np.array([-x[22]*math.cos(math.radians(x[23])) for x in file1.tolist()])

#lat_deg = -12.999095
#So = 1366

########################################
#Primeiro controle - valores absolutos #
########################################
precip[(precip < -800)] = np.nan
radSw[(radSw < 0)] = np.nan
atmLw[(atmLw < 300)] = np.nan
balRad[(balRad < -250)] = np.nan
parSw[(parSw < 0)] = np.nan
tempAr[(tempAr < 18)] = np.nan
humRel[(humRel < 20)] = np.nan
velAr[(velAr < 0)] = np.nan
dirAr[(dirAr < 0)] = np.nan
radSw[(radSw > 1366)] = np.nan
atmLw[(atmLw > 500)] = np.nan
balRad[(balRad > 750)] = np.nan
parSw[(parSw > 800)] = np.nan
tempAr[(tempAr > 40)] = np.nan
humRel[(humRel > 100)] = np.nan
velAr[(velAr > 20)] = np.nan
dirAr[(dirAr > 360)] = np.nan

#################################
#Escolha das datas              #
#################################

#""" Data arbitrária
date_start = datetime(2017, 12, 1)
date_end = datetime(2018, 1, 30)
#"""

#date_start = datetime(2017, 12, 1)
#date_end = datetime(2018, 3, 30)


indexeslenta = np.array(list(map(lambda x: date_start <= x <= date_end, dateslenta)))
indexesteo = np.array(list(map(lambda x: date_start <= x <= date_end, datesteorico)))

##################################################
#Separando valores de acordo a data escolhida    #
##################################################
delta_dateteo = datesteorico[indexesteo]
delta_datelenta = dateslenta[indexeslenta]
delta_radteo = radTeo[indexesteo]
delta_radSw = radSw[indexeslenta]
delta_atmLw = atmLw[indexeslenta]
delta_balRad = balRad[indexeslenta]
delta_parSw = parSw[indexeslenta]
delta_tempAr = tempAr[indexeslenta]
delta_humRel = humRel[indexeslenta]
delta_velAr = velAr[indexeslenta]
delta_dirAr = dirAr[indexeslenta]
delta_compHor = compHor[indexeslenta]
delta_compMer = compMer[indexeslenta]

for i in range(len(delta_dateteo)):
    delta_dateteo[i] = delta_dateteo[i] - timedelta(minutes=30)
for i in range(len(delta_datelenta)):
    delta_datelenta[i] = delta_datelenta[i] + timedelta(minutes=30)

diaCont = [];radSwMean = [];atmLwMean = [];balRadMean = [];parSwMean = [];tempArMean = [];
humRelMean = [];velArMean = [];dirArMean = [];compHorMean = [];compMerMean = [];datesPlot = []


#########################################
# Calculo das medias horarias           #
#########################################
current_date = date_start

while current_date <= date_end:

    next_date = current_date + timedelta(hours=1)
    current_hour = current_date.hour
    iterator = current_date
    count_hour = 0

    #Confere a quantidade de valores no horário
    while current_hour == iterator.hour:
        count_hour += 1
        iterator = iterator + timedelta(minutes=5)

    datesPlot.append(current_date)
    diaCont.append(plott.cont_year(current_date))

    if count_hour >= 6:
        indexeslento = np.array(list(map(lambda x: current_date <= x < next_date, delta_datelenta.tolist())))
        radSwMean.append(np.nanmean(delta_radSw[indexeslento]))
        atmLwMean.append(np.nanmean(delta_atmLw[indexeslento]))
        balRadMean.append(np.nanmean(delta_balRad[indexeslento]))
        parSwMean.append(np.nanmean(delta_parSw[indexeslento]))
        tempArMean.append(np.nanmean(delta_tempAr[indexeslento]))
        humRelMean.append(np.nanmean(delta_humRel[indexeslento]))
        velArMean.append(np.nanmean(delta_velAr[indexeslento]))
        compHorMean.append(np.nanmean(delta_compHor[indexeslento]))
        compMerMean.append(np.nanmean(delta_compMer[indexeslento]))
        alfa = math.atan2(np.nanmean(delta_compMer[indexeslento]), np.nanmean(delta_compHor[indexeslento]))
        dirArMean.append(math.fmod(3*(math.pi/2) - alfa, 2*math.pi)*(180/math.pi))
    else:
        radSwMean.append(np.nan)
        atmLwMean.append(np.nan)
        balRadMean.append(np.nan)
        parSwMean.append(np.nan)
        tempArMean.append(np.nan)
        humRelMean.append(np.nan)
        velArMean.append(np.nan)
        compHorMean.append(np.nan)
        compMerMean.append(np.nan)
        dirArMean.append(np.nan)

    current_date = next_date

######################################
#Array de médias horárias            #
######################################
radSwMean = np.array(radSwMean)
atmLwMean = np.array(atmLwMean)
balRadMean = np.array(balRadMean)
parSwMean = np.array(parSwMean)
tempArMean = np.array(tempArMean)
humRelMean = np.array(humRelMean)
velArMean = np.array(velArMean)
dirArMean = np.array(dirArMean)
datesPlotb = datesPlot
datesPlot = date2num(datesPlot)
datesTeo = date2num(datesteorico)
diaCont = np.array(diaCont)
######################################################
#Valores absolutos após o cálculo com as componentes #
######################################################
dirArMean[(dirArMean < 0)] = np.nan
dirArMean[(dirArMean > 360)] = np.nan


################################
#NAMELIST PARA OS GRAFICOS     #
#tar - temperatura do ar       #
#var - velocidade do vento     #
#dar - direcao do vento        #
#hum - humidade relativa       #
#par - radiacao par            #
#saldo - saldo radiacao        #
#longa - onda longa atmosfera  #
#sol - radiacao solar          #
################################


"""

#########################################
# Boxplots                              #
#########################################

for i in range(12):
    plott.drawbox(plott.databoxplt(tempArMean,datesPlotb, i+1), "tar" + str(i+1),i,"tar")
    plott.drawbox(plott.databoxplt(velArMean,datesPlotb, i+1), "var"+ str(i+1),i,"var")
    plott.drawbox(plott.databoxplt(dirArMean,datesPlotb, i+1), "dar"+ str(i+1),i,"dar")
    plott.drawbox(plott.databoxplt(humRelMean,datesPlotb, i+1), "hum"+ str(i+1),i,"hum")
    plott.drawbox(plott.databoxplt(parSwMean,datesPlotb, i+1), "par"+ str(i+1),i,"par")
    plott.drawbox(plott.databoxplt(balRadMean,datesPlotb, i+1), "saldo"+ str(i+1),i,"saldo")
    plott.drawbox(plott.databoxplt(atmLwMean,datesPlotb, i+1), "longa"+ str(i+1),i,"longa")
    plott.drawbox(plott.databoxplt(radSwMean,datesPlotb, i+1), "sol"+ str(i+1),i,"sol")


plott.drawboxseason(plott.dataseason(tempArMean,datesPlotb,12,1,2),12,1,2,'temp','tar')
plott.drawboxseason(plott.dataseason(tempArMean,datesPlotb,4,5,6),4,5,6,'temp','tar')

plott.drawboxseason(plott.dataseason(humRelMean,datesPlotb,12,1,2),12,1,2,'hum','hum')
plott.drawboxseason(plott.dataseason(humRelMean,datesPlotb,4,5,6),4,5,6,'hum','hum')
"""


#################################################################################################################
#################################################################################################################
"""
tverao = plott.dataseason(tempArMean,datesPlotb,12,1,2)
tinverno = plott.dataseason(tempArMean,datesPlotb,4,5,6)
hverao = plott.dataseason(humRelMean,datesPlotb,12,1,2)
hinverno = plott.dataseason(humRelMean,datesPlotb,4,5,6)
fig = plt.figure(1)
fig.suptitle("Verão (DJF)")
ax = fig.add_subplot(111)
ax2 = ax.twinx()
hours = ['00h', '01h', '02h', '03h', '04h', '05h', '06h', '07h', '08h', '09h', '10h', '11h', '12h', '13h',
         '14h', '15h', '16h', '17h', '18h', '19h', '20h', '21h', '22h', '23h']
ax.set_title('Temperatura do Ar (°C)', fontsize=12, loc='left')
bp1 = ax.boxplot(tverao, sym='', labels=hours, patch_artist=True)
for box in bp1['boxes']:
    box.set( color='#7570b3', linewidth=1)
    box.set( facecolor = '#b72619')
ax.set_ylim(20, 35)

bp2 = ax2.boxplot(hverao, sym='', labels=hours, patch_artist=True)
ax2.set_title('Umidade Relativa do Ar (%)', fontsize=12, loc='right')
for box in bp2['boxes']:
    box.set( color='#1b979e', linewidth=1)
    box.set( facecolor='#163b82')
ax2.set_ylim(40, 100)

plt.plot([], [], label='Temperatura', color='#b72619')
plt.plot([], [], label='Humidade Relativa', color='#163b82')
leg = plt.legend(loc='lower right')

plt.savefig("veraoboxplot_unico.png", bbox_inches='tight')
plt.close()
"""

tverao = plott.dataseason(tempArMean,datesPlotb,12,1,2)
tinverno = plott.dataseason(tempArMean,datesPlotb,4,5,6)
hverao = plott.dataseason(humRelMean,datesPlotb,12,1,2)
hinverno = plott.dataseason(humRelMean,datesPlotb,4,5,6)
fig = plt.figure(1)
fig.suptitle("Verão (DJF)")
ax = fig.add_subplot(111)
ax2 = ax.twinx()
hours = ['00h', '01h', '02h', '03h', '04h', '05h', '06h', '07h', '08h', '09h', '10h', '11h', '12h', '13h',
         '14h', '15h', '16h', '17h', '18h', '19h', '20h', '21h', '22h', '23h']
ax.set_title('Temperatura do Ar (°C)', fontsize=12, loc='left')
bp1 = ax.boxplot(tverao, sym='', labels=hours, patch_artist=True)
for box in bp1['boxes']:
    box.set( color='#7570b3', linewidth=1)
    box.set( facecolor = '#b72619')
ax.set_ylim(20, 35)

bp2 = ax2.boxplot(hverao, sym='', labels=hours, patch_artist=True)
ax2.set_title('Umidade Relativa do Ar (%)', fontsize=12, loc='right')
for box in bp2['boxes']:
    box.set( color='#1b979e', linewidth=1)
    box.set( facecolor='#163b82')
ax2.set_ylim(40, 100)

plt.plot([], [], label='Temperatura', color='#b72619')
plt.plot([], [], label='Humidade Relativa', color='#163b82')
leg = plt.legend(loc='lower right')

plt.savefig("veraoboxplot.png", bbox_inches='tight')
plt.close()

#################################################################################################################
#################################################################################################################

#################################
# Comando para salvar variaveis #
#################################

#np.savetxt('test.dat', np.c_[DatesF,radSwF,atmLwF,balRadF,parSwF,tempArF,
#           humRelF,velArF,dirArF], fmt='%.10g', delimiter=' ')


###################################################################
# Plotagem de gráficos bruto x media (ano contínuo) || usar draw2 #
###################################################################
#ind = np.array(list(map(lambda x: date_start <= x <= date_end, dateslenta)))
#datescont = np.array([plott.cont_year(x) for x in dateslenta[ind]])

#plott.draw2(datescont, delta_radSw, diaCont, radSwMean, "sol")
#plott.draw2(datescont, delta_atmLw, diaCont, atmLwMean, "longa")
#plott.draw2(datescont, delta_balRad, diaCont, balRadMean, "saldo")
#plott.draw2(datescont, delta_parSw, diaCont, parSwMean, "par")
#plott.draw2(datescont, delta_tempAr, diaCont, tempArMean, "tar")
#plott.draw2(datescont, delta_humRel, diaCont, humRelMean, "hum")
#plott.draw2(datescont, delta_velAr, diaCont, velArMean, "var")
#plott.draw2(datescont, delta_dirAr, diaCont, dirArMean, "dar")

######################################################################################
# Plotagem de grafico de 1 variavel ou variavel em cima do valor bruto || usar draw  #
######################################################################################

#plott.draw(diaCont, tempArMean, "dar")

#####################################
# Precipitacao                      #
#####################################

"""
indexesrain = np.array(list(map(lambda x: date_start <= x <= date_end, datesrain)))
current_date = date_start
delta_datesRain = datesrain[indexesrain]
datesPLotRain = [];diaContRain = []; precipSum = []

while current_date <= date_end:

    next_date = current_date + timedelta(hours=1)
    current_hour = current_date.hour
    iterator = current_date
    count_hour = 0

    #Confere a quantidade de valores no horário
    while current_hour == iterator.hour:
        count_hour += 1
        iterator = iterator + timedelta(minutes=5)

    datesPLotRain.append(current_date + timedelta(minutes=30))
    diaContRain.append(plott.cont_year(current_date + timedelta(minutes=30)))

    if count_hour >= 6:
        indexesrain = np.array(list(map(lambda x: current_date <= x < next_date, delta_datesRain.tolist())))
        precipSum.append(np.sum(precip[indexesrain]))
    else:
        precipSum.append(np.nan)

    current_date = next_date

precipSum = np.array(precipSum)
"""