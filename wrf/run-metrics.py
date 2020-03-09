from comparissons import PairDf, generate_metrics, generate_distributions, generate_mean, area_graph
import pandas as pd 
import numpy as np 
from warnings import filterwarnings
from datetime import datetime
from sys import exit
filterwarnings('ignore')

#path and name os files to analyse, real = colected | model = output from model
obs_path = '/home/labmim/meteorologia/rmrj/'

model_path = '/home/models/WRF/wrf-case/d-output/petrobras/'

"""real = ['COP_inmet','SER_inmet','DEO_inmet', \
    'SBG_metar','SBR_metar','SBJ_metar','SBS_metar', \
    'CEN_smac','LAB_igeo','LB1_igeo','LB2_igeo', \
    'SBC_metar','SBL_metar','SAN_pnboia']

model = ['wrf_COP','wrf_SER','wrf_DEO', \
    'wrf_SBG','wrf_SBR','wrf_SBJ','wrf_SBS', \
    'wrf_CEN','wrf_LAB', 'wrf_LAB1','wrf_LAB2', \
    'wrf_SBC','wrf_SBL','wrf_SAN']
"""

real = ['COP_inmet','SER_inmet','DEO_inmet', \
    'SBG_metar','SBR_metar','SBJ_metar','SBS_metar', \
    'CEN_smac','LAB_igeo','LB1_igeo','LB2_igeo']

model = ['wrf_COP','wrf_SER','wrf_DEO', \
    'wrf_SBG','wrf_SBR','wrf_SBJ','wrf_SBS', \
    'wrf_CEN','wrf_LAB', 'wrf_LAB1','wrf_LAB2']

#names of stations
stations = [x[:3] for x in real]

#path to save sheets
out_path_sheets = '/home/models/WRF/wrf-case/d-output/sheets/'

#path to save distributions
out_path_distribution = '/home/models/WRF/wrf-case/d-output/distributions/'

#path to save area graphs
out_path_plots = '/home/models/WRF/wrf-case/d-output/area_graphs/'


#params = ['sfcpbl11','sfcpbl11_sst','sfcpbl17_sst','sfcpbl22_sst']
params = ['sfcpblz2230','sfcpblz1130','sfcpblz1730']

"""
for case in params:
    #absolute paths
    files = [(obs_path+x+'.dat',model_path+case+'/timeseries/'+y+'.dat',z) for x,y,z in zip(real,model,stations)]
    
    #names and months   
    #months = [[1,2,3,4,5,6,7,8,9,10,11,12],[1,2,3],[6,7,8]]
    months = [[7]]
    #names = [case+'_completo_',case+'_verao_',case+'_inverno_']
    names = [case+'_inverno_']

    #varibales
    variables = ['T','ur','q','WS','Sw_dw','pressure','ustar','LE','H', 'WD']

    for name, month in zip(names,months):
        #function to generate sheets
        generate_metrics(files,out_path_sheets,name,variables,month)
        
        #function to generate distributions
        generate_distributions(files,out_path_distribution,name,variables,month)


    variables = ['T','ur','pressure','WS','q','TSM']
    generate_mean(files,out_path_sheets,case,variables,months[0])
"""

params = ['sfcpblz2230/','sfcpblz1130/','sfcpblz1730/']
variables = 'T ur q WS Sw_dw H LE ustar'.split()

obs = {}
for x in real:
    obs[x] = obs_path+x+'.dat'

pred = {}
for i,x in enumerate(real):
    pred[x] = [(p,model_path+p+'timeseries/'+model[i]+'.dat') for p in params]

period = [datetime(2013,7,9),datetime(2013,7,20)]
area_graph(obs,pred,out_path_plots,variables,period)
