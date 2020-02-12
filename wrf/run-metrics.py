from comparissons import PairDf, generate_metrics, generate_distributions, generate_mean
import pandas as pd 
import numpy as np 
from warnings import filterwarnings
from sys import exit
filterwarnings('ignore')

#path and name os files to analyse, real = colected | model = output from model
obs_path = '/home/labmim/meteorologia/rmrj/'

model_path = '/home/models/WRF/wrf-case/d-output/petrobras/'

real = ['COP_inmet','SER_inmet','DEO_inmet', \
    'SBG_metar','SBR_metar','SBJ_metar','SBS_metar', \
    'CEN_smac','LAB_igeo', \
    'SBC_metar','SBL_metar','SAN_pnboia']

model = ['wrf_COP','wrf_SER','wrf_DEO', \
    'wrf_SBG','wrf_SBR','wrf_SBJ','wrf_SBS', \
    'wrf_CEN','wrf_LAB', \
    'wrf_SBC','wrf_SBL','wrf_SAN']

#names of stations
stations = [x[:3] for x in real]

#path to save sheets
out_path_sheets = '/home/models/WRF/wrf-case/d-output/sheets/'

#path to save distributions
out_path_distribution = '/home/models/WRF/wrf-case/d-output/distributions/'

#params = ['sfcpbl11','sfcpbl11_sst','sfcpbl17_sst','sfcpbl22_sst']
params = ['sfcpblz1730_sst']

for case in params:
    #absolute paths
    files = [(obs_path+x+'.dat',model_path+case+'/timeseries/'+y+'.dat',z) for x,y,z in zip(real,model,stations)]
    
    #names and months   
    months = [[1,2,3,4,5,6,7,8,9,10,11,12],[1,2,3],[6,7,8]]
    names = [case+'_completo_',case+'_verao_',case+'_inverno_']

    #varibales
    variables = ['T','ur','q','WS','Sw_dw','pressure']

    for name, month in zip(names,months):
        #function to generate sheets
        generate_metrics(files,out_path_sheets,name,variables,month)
        
        #function to generate distributions
        #generate_distributions(files,out_path_distribution,name,variables,month)

    variables = ['T','ur','pressure','WS','q','TSM']
    generate_mean(files,out_path_sheets,case,variables,months[0])
