import netCDF4
import numpy as np
from datetime import datetime, timezone
from warnings import filterwarnings
import matplotlib.pyplot as plt
from math import sqrt
from mpl_toolkits.basemap import Basemap
import pandas as pd
filterwarnings('ignore')

def getLowHigh(variable):
    varflat = variable.flatten()
    varlow, varhigh = np.amin(varflat), np.amax(varflat)
    return varlow, varhigh

def print_points(dataset, lat, lon, name, day):
    nlat, nlon = None, None
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

        #print("LambMiM = lat = {} long = {}".format(latCoord,longCoord))
        print("{}\nfound: lat = {:0.4f} long = {:0.4f} | matrix coordinates = {} | {}".format(name,ncoord[0][0], ncoord[0][1], latCoord, longCoord))
        i, j = latCoord, longCoord
        nlat = [ncoord[0][0], matrix[i+3,j+2][0], matrix[i,j+3][0]]
        nlon = [ncoord[0][1], matrix[i+2,j+2][1], matrix[i,j+3][1]]

    except IndexError:
        print('File {i} is out of shape!'.format(i=name))

        xlat = dataset.variables['XLAT'][:]
        xlong = dataset.variables['XLONG'][:]
        xlat = xlat[0:1, :, :].squeeze()
        xlong = xlong[0:1, :, :].squeeze()
        id = xlat.shape
        print('Dim = {}\n'.format(id))
        return 0, 0

    data = dataset
    xlat, xlong = data.variables['XLAT'][:,:,:], data.variables['XLONG'][:,:,:]
    lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
    hlat, llat = np.amax(xlat), np.amin(xlat)
    hlong, llong = np.amax(xlong), np.amin(xlong)

    times_array = dataset.variables['Times'][:]

    LH = None; cont = 0
    for i,time in enumerate(times_array):
        currentTime = b''.join(time.tolist()).decode('UTF-8')
        current_date = datetime.strptime(currentTime, '%Y-%m-%d_%H:%M:%S')
        cd = current_date.replace(tzinfo=timezone.utc).astimezone(tz=None)
        if cd.day == day and cd.hour == 12:
            tmp = dataset.variables['LU_INDEX'][:,:,:].squeeze()
            #tmp = dataset.variables['LANDMASK'][:,:,:].squeeze()
            lh = tmp[i:i+1,:,:]
            if LH is None:
                LH = lh
            else:
                LH += lh
            cont+=1

    varmin, varmax = getLowHigh(LH)

    plt.figure(figsize=(8,6))
    #plt.suptitle("$^\circ\mathcal{C}$", fontsize=18, ha='center', x=0.79, y=0.75)
    plt.suptitle("LU_INDEX", fontsize=18, ha='center', x=0.79, y=0.75)
    plt.xlabel('Long (°)', fontsize=14, labelpad=25)
    plt.ylabel('Lat (°)', fontsize=14, labelpad=60)

    m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='f',projection='merc',llcrnrlon= llong, llcrnrlat= llat, urcrnrlon= hlong, urcrnrlat= hlat)

    x,y = m(lon, lat)

    m.drawcoastlines()

    m.drawparallels(np.arange(llat, hlat,abs(hlat-llat)/7), linewidth=0, labels=[1,0,0,0], color='r',zorder=0, fmt="%.1f", fontsize=14)
    m.drawmeridians(np.arange(llong, hlong,abs(hlong-llong)/5), linewidth=0, labels=[0,0,0,1], color='r',zorder=0, fmt="%.1f", fontsize=14)
    

    #m.contourf(x,y, np.squeeze(LH), alpha=0.4, cmap='jet', vmin=varmin, vmax=varmax)
    m.pcolormesh(x,y, np.squeeze(LH), alpha=0.4, cmap='jet_r', vmin=varmin, vmax=varmax)

    print(nlat)
    print(nlon)
    x,y = m(nlon,nlat)
    m.plot(x,y,'o',color='black')
    
    cb = plt.colorbar(shrink=0.5, pad=0.04)
    cb.ax.tick_params(labelsize=10)
    #plt.savefig(out_path+name+var+".png",bbox_inches='tight')
    plt.show()
    plt.close()

#netcdf = netCDF4.Dataset('/home/models/WRF/wrf-case/d-output/petrobras/sfcpblz1730_sst/wrfout_d04_2013-07-01')
#print_points(netcdf,-22.857222,-43.233716,'LAB',17)
