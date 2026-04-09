from netcdfUtils import *

#configs for ssa runs
"""
path = '/media/labmim/dados/coral-sol/simula/'
cases = ['costa/','costa_sst/','costa_sst_mur/','padrao/']
base = 'wrfout_xxx_2014-'
months = '12'.split()

#d03
stations1 =     {'SSA_wrf':[-12.906608,-38.321383],'mt1_wrf':[-13.014864,-38.480533],'mt2_wrf':[-12.873015,-38.676755],'mt3_wrf':[-12.816463,-38.642604],
                'IGT_wrf':[-12.978068,-38.468746],'DIQ_wrf':[-12.983728,-38.506913],'ITG_wrf':[-12.993365,-38.461875],'CAB_wrf':[-12.953696,-38.428270],
                'PIR_wrf':[-12.898911,-38.457865],'RVE_wrf':[-13.005552,-38.487150],'MTR_wrf':[-12.906608,-38.321383],'TSM':[-12.818583,-38.657767]}

#d02
stations2 = {'CDA_wrf':[-12.675422,-39.089580],'FDS_wrf':[-12.196200,-38.967384]}
"""

path = '/home/models/WRF/wrf-case/d-output/petrobras/'

cases = ['sfcpblz1730/','sfcpblz2230/','sfcpblz1130/']
#cases = ['sfcpblz1130/']

#all year complete
base = 'wrfout_xxx_2013-'

#months = '01 02 03 04 05 06 07 08 09 10 11 12'.split()
months = '07'.split()

arqs = [base+x+'-01' for x in months]

#grades RJ
#d04
stations1 =     {'wrf_COP':[-22.988286,-43.190436],'wrf_SER':[-22.757868,-43.684843],'wrf_DEO':[-22.861322,-43.411410],'wrf_SBG':[-22.808910,-43.249932],
                'wrf_SBR':[-22.913496,-43.166118],'wrf_SBJ':[-22.987733,-43.370019],'wrf_SBS':[-22.936906,-43.712884],'wrf_CEN':[-22.908339,-43.178156],
                'wrf_LAB':[-22.857222,-43.233716],'wrf_LAB1':[-22.83429,-43.218628],'wrf_LAB2':[-22.861252,-43.208862]}

#d02
stations2 = {'wrf_SBL':[-22.109461,-39.916933],'wrf_SAN':[-25.439500,-45.036100],'wrf_SBC':[-21.698617,-41.306108]}

for case in cases:
    foutput = '/home/models/WRF/wrf-case/d-output/petrobras/'+case+'timeseries/'
    #for grade in ['d04','d02']:
    for grade in ['d04']:
        files = [path+case+x[:7] + grade + x[10:] for x in arqs]
        if grade == 'd04':
            generateSeries(files,stations1,foutput)
        else:
            generateSeries(files,stations2,foutput)