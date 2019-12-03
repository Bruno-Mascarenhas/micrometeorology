from mapsGenerator import *
from sys import exit

path = '/media/labmim/Dados2/calpuff/'

grades = ['d01','d02','d03','d04']

cases = ['sfcpbl11/']

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

            foutput = '/media/labmim/Dados2/calpuff/mapas/'
            variables = ['wind']
            drawMapSeason(variables,nfile,foutput,case[:-1]+'-'+grade+'-'+s,13)

file = '/media/labmim/Dados2/calpuff/sfcpbl11/wrfout_d04_2013-06-01'
foutput = '/media/labmim/Dados2/calpuff/mapas/brisa/'
drawBreeze(file,foutput,'d04-inverno',168)

file = '/media/labmim/Dados2/calpuff/sfcpbl11/wrfout_d04_2013-01-01'
drawBreeze(file,foutput,'d04-verao',168)
