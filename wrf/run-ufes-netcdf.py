from netcdfUtils import *

path = '/home/models/WRF/wrf-case/d-output/petrobras/'

grades = ['d02','d04']

#cases = ['sfcpbl11/','sfcpbl11_sst/','sfcpbl17_sst/','sfcpbl22_sst/']
cases = ['sfcpbl112_sst/','sfcpbl17_sst/']


#all year complete
base = 'wrfout_xxx_2013-'

months = '01 02 03 04 05 06 07 08 09 10 11 12'.split()

arqs = [base+x+'-'+months[0] for x in months]

#d04
stations1 =     {'wrf_COP':[-22.988286,-43.190436],'wrf_SER':[-22.757868,-43.684843],'wrf_DEO':[-22.861322,-43.411410],'wrf_SBG':[-22.808910,-43.249932],
                'wrf_SBR':[-22.913496,-43.166118],'wrf_SBJ':[-22.987733,-43.370019],'wrf_SBS':[-22.936906,-43.712884],'wrf_CEN':[-22.908339,-43.178156],
                'wrf_LAB':[-22.857222,-43.233716]}

#d02
stations2 = {'wrf_SBL':[-22.109461,-39.916933],'wrf_SAN':[-25.439500,-45.036100],'wrf_SBC':[-21.698617,-41.306108]}

for case in cases:
    for grade in ['d04','d02']:
        files = [path+case+x[:7] + grade + x[10:] for x in arqs]

        if grade == 'd04':
            foutput = '/home/models/WRF/wrf-case/d-output/petrobras/'+case+'series/'
            generateSeries(files,stations1,foutput)
        else:
            foutput = '/home/models/WRF/wrf-case/d-output/petrobras/'+case+'series/'
            generateSeries(files,stations2,foutput)
