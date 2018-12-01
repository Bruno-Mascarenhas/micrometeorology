import os
import sys
import glob
import netCDF4
import numpy as np
from math import sqrt

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def getFileNames(dataDir):
    os.chdir(dataDir)
    fileList = glob.glob('*.nc')
    fileList.sort()
    file = open(os.path.join(__location__,'files.dat'), 'w')
    files = []
    for item in fileList:
        if item[7:10] == "d03":
            item = dataDir + "\\" + item
            file.write("%s\n" % item)
            files.append(item)
        else:
            continue
    file.close()
    return files

#getFileNames("C:\\Users\\BrunoM\\Documents\\working now\\wrf-2018")
	
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