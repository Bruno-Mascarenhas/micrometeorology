from comparissons import PairDf, generate_metrics, generate_distributions, generate_mean
import pandas as pd 
import numpy as np 
from warnings import filterwarnings
from sys import exit
filterwarnings('ignore')

#path and name os files to analyse, real = colected | model = output from model
obs_path = '/home/labmim/Documentos/bruno/sheets/obs/'

model_path = '/home/labmim/Documentos/bruno/sheets/model/'

params = ['sfcpbl11']

real = ['COP_inmet','SER_inmet','DEO_inmet', \
    'SBG_metar','SBR_metar','SBJ_metar','SBS_metar', \
    'CEN_smac','LAB_igeo', \
    'SBC_metar','SBL_metar','SAN_pnboia']

model = ['wrf_COP','wrf_SER','wrf_DEO', \
    'wrf_SBG','wrf_SBR','wrf_SBJ','wrf_SBS', \
    'wrf_CEN','wrf_LAB', \
    'wrf_SBC','wrf_SBL','wrf_SAN']

#path to save sheets
out_path_sheets = '/home/labmim/Documentos/bruno/sheets/out_sheets/'

#path to save distributions
out_path_distribution = '/home/labmim/Documentos/bruno/sheets/out_dist/'

#names of stations
stations = [x[:3] for x in real]

for case in params:
    #absolute paths
    files = [(obs_path+x+'.dat',model_path+case+'/'+y+'.dat',z) for x,y,z in zip(real,model,stations)]
    #name to save file
    names = [case+'_completo_',case+'_verao_',case+'_inverno_']
    #months to analyse
    months = [[1,2,3,4,5,6,7,8,9,10,11,12],[1,2,3],[6,7,8]]

    #varibales for monthly/hourly statistics    
    variables = ['T','ur','q','WS','Sw_dw','ustar','H','pressure','TSM']
    
    for name, month in zip(names,months):
        #generate_metrics(files,out_path_sheets,name,variables,month)
        generate_distributions(files,out_path_distribution,name,variables,month)
    
    #variables for yearly means
    #variables = ['T','ur','pressure','WS','u','v','q','TSM']
    #generate_mean(files,out_path_sheets,case,variables,months[0])