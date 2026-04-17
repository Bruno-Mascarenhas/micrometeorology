from datetime import datetime
from warnings import filterwarnings

from comparissons import area_graph

filterwarnings('ignore')

#configs for ssa runs
"""
obs_path = '/home/labmim/meteorologia/estacoes/rms/'
model_path = '/media/labmim/dados/coral-sol/simula/'

real = ['mt1','mt2','mt3','MTR_SSA', \
    'CDA_inmet','FDS_inmet','SSA_inmet','IGT_pms', \
    'CAB_pms','PIR_pms']

model = ['mt1_wrf','mt2_wrf','mt3_wrf','MTR_wrf', \
    'CDA_wrf','FDS_wrf','SSA_wrf','IGT_wrf', \
    'CAB_wrf','PIR_wrf']

out_path_sheets = '/media/labmim/dados/coral-sol/metrics/'
out_path_distribution = '/media/labmim/dados/coral-sol/distributions/'
params = ['costa','costa_sst','costa_sst_mur','padrao']


"""
#path and name os files to analyse, real = colected | model = output from model
obs_path = '/home/labmim/meteorologia/rmrj/'

model_path = '/home/models/WRF/wrf-case/d-output/petrobras/'

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

#definitions for area graphs
params = ['sfcpblz2230/','sfcpblz1130/','sfcpblz1730/']
variables = ['T', 'ur', 'q', 'WS', 'Sw_dw', 'H', 'LE', 'ustar']

obs = {}
for x in real:
    obs[x] = obs_path+x+'.dat'

pred = {}
for i,x in enumerate(real):
    pred[x] = [(p,model_path+p+'timeseries/'+model[i]+'.dat') for p in params]

period = [datetime(2013,7,9),datetime(2013,7,20)]
area_graph(obs,pred,out_path_plots,variables,period)
