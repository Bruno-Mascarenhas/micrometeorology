import os
import sys
import glob
import netCDF4
import numpy as np
from math import sqrt
from datetime import datetime, timedelta
import pandas as pd

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def getFileNames(dataDir):
    month = str(datetime.now().month); day = str(datetime.now().day)
    if len(month) == 1:
        month = '0'+str(month)
    if len(day) == 1:
        day = '0'+str(day)
    today = str(datetime.now().year)+'-'+month+'-'+day
    #today = '2018-08-30'
    #grades = ['d01','d02','d03','d04']
    grades = ['d01','d02']
    os.chdir(dataDir)
    fileList = os.listdir() #glob.glob('*.nc')
    file = open(os.path.join(__location__,'files.dat'), 'w')
    files = []
    for item in fileList:
        if today in item and any(grade in item for grade in grades):
            item = dataDir + "/" + item
            file.write("%s\n" % item)
            files.append(item)
        else:
            continue
    file.close()
    return files

#getFileNames("/home/labmim/Build_WRF/d-output")
	
def ordDirectory(dataDir)	:
    for name in os.listdir(dataDir):
        current_name = os.path.join(dataDir, name)
        new_name = os.path.join(dataDir, name[0:21] + '.nc')
        os.rename(current_name, new_name)

#ordDirectory('C:\\Users\\BrunoM\\Documents\\working now\\wrf-2018')

def getLatLon(dataset,lat,lon,name):
    try:
        xlat = dataset.variables['XLAT'][:]
        xlong = dataset.variables['XLONG'][:]

        xlat = xlat[0:1, :, :].squeeze()
        xlong = xlong[0:1, :, :].squeeze()

        idi = xlat.shape[0]
        idj = xlat.shape[1]
        coord = [[lat, lon]]

        matrix = np.dstack((xlat,xlong))
        #matrix[0,0][0] = lat  || matrix[0,0][1] = long || coord[0][0] = lat || coord[0][1] = lon
        min = 99999.99999; ncoord = [[]]

        for i in range(idi):
            for j in range(idj):
                aux = sqrt((matrix[i,j][0] - coord[0][0])**2+(matrix[i,j][1] - coord[0][1])**2)
                if aux<=min:
                    min = aux
                    ncoord = [[matrix[i,j][0],matrix[i,j][1]]]

        a = np.argwhere(matrix == ncoord)
        for i in range(len(a)):
            if a[i-1][0] == a[i][0] and a[i-1][1] == a[i][1]:
                latCoord = a[i][0]; longCoord = a[i][1]

        print("LambMiM = lat = {} long = {}".format(latCoord,longCoord))
        print("lat = {} long = {}".format(ncoord[0][0], ncoord[0][1]))

        return latCoord, longCoord

    except IndexError:
        print('File {i} is out of shape! shape is {j}'.format(i=name, j=id))

        xlat = dataset.variables['XLAT'][:]
        xlong = dataset.variables['XLONG'][:]
        xlat = xlat[0:1, :, :].squeeze()
        xlong = xlong[0:1, :, :].squeeze()
        id = xlat.shape
        print('Dim = {}\n'.format(id))
        return 0, 0

#dataset = netCDF4.Dataset("C:\\Users\\BrunoM\\Documents\\working now\\allGraphs\\wrfout_d03_2018-04-27_00_00_00.nc")
#getLatLon(dataset,-12.999095,-38.508166,"wrfout_d03_2018-04-27_00_00_00.nc")


#Faz o pareamento do eixo do tempo de uma variável entre dois dataframes
def dfParser(df1,df2,var,name):
    new_val1, new_val2 = [], []
    data1 = [[x,y] for x,y in zip(df1.index,df1[var])]
    data2 = [[x,y] for x,y in zip(df2.index,df2[var])]

    i = 0
    j = 0

    while i < len(data1) and j < len(data2):
        if data1[i][0] == data2[j][0] and not (np.isnan(data1[i][1]) or np.isnan(data2[j][1])):
            new_val1.append(data1[i][1])
            new_val2.append(data2[j][1])
            i+=1
            j+=1
        elif data1[i][0] < data2[i][0]:
            i+=1
        else:
            j+=1
    
    df_out = pd.DataFrame({'WRF':new_val1,name:new_val2})

    return df_out