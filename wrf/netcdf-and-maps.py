from netcdfUtils import *
from sys import exit
import glob

path = '/disco4/coralsol/taoan/wrf-longo-periodo'

stations1 =     {'wrf_ITA':[-12.965947, -38.608518],'wrf_LAB':[-12.906608,-38.321383]}

for year in range(2016,2022):
    rel_path = path + '/' + str(year) + '/'
    files = glob.glob(rel_path + '*d03*')
    print(files)
    #generateSeries(files,stations1,rel_path)
    break