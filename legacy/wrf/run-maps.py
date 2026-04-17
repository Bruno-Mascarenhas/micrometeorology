
from mapsGenerator import *

#configs for ssa runs
"""
path = '/media/labmim/dados/coral-sol/simula/'
grades = ['d01','d02','d03']
cases = ['costa/','costa_sst/','costa_sst_mur/','padrao/']
base = 'wrfout_xxx_2014-'
months = '12'.split()
"""

path = '/home/models/WRF/wrf-case/d-output/calpuff/'

grades = ['d01','d02','d03','d04']

cases = ['teste/']

#all year complete
base = 'wrfout_xxx_2013-'

#months = '01 02 03 04 05 06 07 08 09 10 11 12'.split()
months = ['01']

arqs = [base+x+'-'+months[0] for x in months]
#month = [['01-01','02-01','03-01'],['06-01','07-01','08-01']]
month = [['01-01']]
#season = ['verao-','inverno-']
season = ['verao-']

for case in cases:
    for grade in grades:
        files = [path+case+x[:7] + grade + x[10:] for x in arqs]

        for (m,s) in zip(month,season):
            nfile = []

            for file in files:
                if any(True if x in file else False for x in m):
                    nfile.append(file)

            foutput = '/home/models/WRF/wrf-case/d-output/maps/'
            variables = ['T','q','wind']
            drawMapSeason(variables,nfile,foutput,case[:-1]+'-'+grade+'-'+s,13)

"""
foutput = '/home/models/WRF/wrf-case/d-output/maps/brisa/'

file = '/home/models/WRF/wrf-case/d-output/petrobras/sfcpbl112_sst/wrfout_d04_2013-06-01'
drawBreezeWind(file,foutput,'d04-wind-inverno',168)
drawBreezeHumidity(file,foutput,'d04-q-inverno',168)


file = '/home/models/WRF/wrf-case/d-output/petrobras/sfcpbl112_sst/wrfout_d04_2013-01-01'
drawBreezeWind(file,foutput,'d04-wind-verao',168)
drawBreezeHumidity(file,foutput,'d04-q-verao',168)
"""
