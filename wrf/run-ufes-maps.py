from mapsGenerator import *
from sys import exit

path = '/home/models/WRF/wrf-case/d-output/petrobras/'

grades = ['d01','d02','d03','d04']

cases = ['sfcpbl112_sst/','sfcpbl17_sst/']

#all year complete
base = 'wrfout_xxx_2013-'

months = '01 02 03 04 05 06 07 08 09 10 11 12'.split()

arqs = [base+x+'-'+months[0] for x in months]
month = [['01-01','02-01','03-01'],['06-01','07-01','08-01']]
season = ['verao-','inverno-']

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

foutput = '/home/models/WRF/wrf-case/d-output/maps/brisa/'

file = '/home/models/WRF/wrf-case/d-output/petrobras/sfcpbl112_sst/wrfout_d04_2013-06-01'
drawBreeze(file,foutput,'d04-inverno',168)

file = '/home/models/WRF/wrf-case/d-output/petrobras/sfcpbl112_sst/wrfout_d04_2013-01-01'
drawBreeze(file,foutput,'d04-verao',168)
