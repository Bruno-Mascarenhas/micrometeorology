from netcdfUtils import *


#path 22
path2 = '/media/labmim/Dados2/calpuff/'

grades = ['d02','d04']

cases = ['sfcpbl11/','sfcpbl22_sst/']

#all year complete
base = 'wrfout_xxx_2013-'

months = '01 02 03 04 05 06 07 08 09 10 11 12'.split()

arqs = [base+x+'-'+months[0] for x in months]

#d04
stations1 =     {'wrf_COP':[-22.988286,-43.190436],'wrf_SER':[-22.757868,-43.684843],'wrf_DEO':[-22.861322,-43.411410],'wrf_SBG':[-22.808910,-43.249932],
                'wrf_SBR':[-22.913496,-43.166118],'wrf_SBJ':[-22.987733,-43.370019],'wrf_SBS':[-22.936906,-43.712884],'wrf_CEN':[-22.908339,-43.178156],
                'wrf_LAB':[-22.857222,-43.233716]}

#d02
stations2 = {'wrf_SBL':[-22.109461,-39.916933],'wrf_SBC':[-21.698617,-41.306108],'wrf_SAN':[-25.439500,-45.036100]}

#d03
stations3 = {'PIRATA':[-18.867,-32.5336]}

for case in cases:

    if '11' in case: 
        path = path1
    else:
        path = path2

    for grade in ['d04','d02','d01']:
        files = [path+case+x[:7] + grade + x[10:] for x in arqs]

        if grade == 'd04':
            foutput = '/home/labmim/Documentos/bruno/sheets/model/'+case
            generateSeries(files,stations1,foutput)
        elif grade == 'd02':
            foutput = '/home/labmim/Documentos/bruno/sheets/model/'+case
            generateSeries(files,stations2,foutput)
        else:
            foutput = '/home/labmim/Documentos/bruno/sheets/model/'+case
            generateSeries(files,stations3,foutput)