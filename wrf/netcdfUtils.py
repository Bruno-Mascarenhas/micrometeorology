import xarray as xr
import pandas as pd 
import numpy as np
import netCDF4
from datetime import datetime, timezone
from warnings import filterwarnings
from math import sqrt
import os
import sys
filterwarnings('ignore')

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def getFileNames(dataDir):
    """
    Function to generate 
    """
    grades = ['d01','d02','d03','d04']
    os.chdir(dataDir)
    fileList = os.listdir()
    file = open(os.path.join(__location__,'files.dat'), 'w')
    files = []
    date = []
    for item in fileList:
        if date in item and any(grade in item for grade in grades):
            item = dataDir + "/" + item
            file.write("%s\n" % item)
            files.append(item)
        else:
            continue
    file.close()
    return files

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

        #print("LambMiM = lat = {} long = {}".format(latCoord,longCoord))
        print("{}\nfound: lat = {:0.4f} long = {:0.4f} | matrix coordinates = {} | {}".format(name,ncoord[0][0], ncoord[0][1], latCoord, longCoord))

        return latCoord, longCoord

    except IndexError:
        print('File {i} is out of shape!'.format(i=name))

        xlat = dataset.variables['XLAT'][:]
        xlong = dataset.variables['XLONG'][:]
        xlat = xlat[0:1, :, :].squeeze()
        xlong = xlong[0:1, :, :].squeeze()
        id = xlat.shape
        print('Dim = {}\n'.format(id))
        return 0, 0

def merge(fpath,fprefix,name):
    """
    fpath = Files Path
    fprefix = Files Prefix
    name = Name to save the final netcdf
    """
    data = xr.open_mfdataset(fpath+fprefix,autoclose=True,parallel=True)
    data.to_netcdf(name)

def generateSeries(files,stations,foutput):
    """
    Files = absolute path of files
    Stations = dict like name:coords of stations
    foutput = Path to save
    """
    for name, coords in stations.items():
        serie = []
        for file in files:
            dataset = netCDF4.Dataset(file)
            latCoord, longCoord = getLatLon(dataset, coords[0], coords[1], name+' - '+file)

            temp = dataset.variables['T2'][:]
            u10 = dataset.variables['U10'][:]
            v10 = dataset.variables['V10'][:]
            p = dataset.variables['PSFC'][:]
            qv = dataset.variables['Q2'][:]
            hfx = dataset.variables['HFX'][:] # H
            lh = dataset.variables['LH'][:] # LH
            swdown = dataset.variables['SWDOWN'][:]
            grdflx = dataset.variables['GRDFLX'][:]
            glw = dataset.variables['GLW'][:]
            ustar = dataset.variables['UST'][:]
            tsm = dataset.variables['SST'][:]
            pblh = dataset.variables['PBLH'][:]
            times_array = dataset.variables['Times'][:]

            it = 0

            header = 'year,month,day,hour,T,q,pressure,u,v,Sw_dw,Lw_dw,H,LE,G,ustar,TSM,PBLH'.split(',')
            serie = {}
            for el in header:
                serie[el] = []

            for time in times_array:
                try:
                    currentTime = b''.join(time.tolist()).decode('UTF-8')
                    current_date = datetime.strptime(currentTime,'%Y-%m-%d_%H:%M:%S')
                    cd = current_date.replace(tzinfo=timezone.utc).astimezone(tz=None)
                except:
                    continue

                serie['year'].append(int(cd.year));    serie['month'].append(int(cd.month));    serie['day'].append(int(cd.day));    serie['hour'].append(int(cd.hour))
                
                temp_array = temp[:, latCoord:latCoord+1, longCoord:longCoord+1].squeeze()
                serie['T'].append(temp_array[it:it + 1][0]-273.15)
                

                u10_array = u10[:, latCoord:latCoord+1, longCoord:longCoord+1].squeeze()
                serie['u'].append(u10_array[it:it + 1][0])

                v10_array = v10[:, latCoord:latCoord+1, longCoord:longCoord+1].squeeze()
                serie['v'].append(v10_array[it:it + 1][0])


                p_array = p[:, latCoord:latCoord+1, longCoord:longCoord+1].squeeze()
                serie['pressure'].append(p_array[it:it + 1][0])


                qv_array = qv[:, latCoord:latCoord+1, longCoord:longCoord+1].squeeze()
                serie['q'].append(qv_array[it:it + 1][0]*1000)


                hfx_array = hfx[:, latCoord:latCoord+1, longCoord:longCoord+1].squeeze()
                serie['H'].append(hfx_array[it:it + 1][0])


                lh_array = lh[:, latCoord:latCoord+1, longCoord:longCoord+1].squeeze()
                serie['LE'].append(lh_array[it:it + 1][0])


                swdown_array = swdown[:, latCoord:latCoord+1, longCoord:longCoord+1].squeeze()
                serie['Sw_dw'].append(swdown_array[it:it + 1][0])

                grdflx_array = grdflx[:, latCoord:latCoord+1, longCoord:longCoord+1].squeeze()
                serie['G'].append(grdflx_array[it:it + 1][0])

                glw_array = glw[:, latCoord:latCoord+1, longCoord:longCoord+1].squeeze()
                serie['Lw_dw'].append(glw_array[it:it + 1][0])

                ustar_array = ustar[:, latCoord:latCoord+1, longCoord:longCoord+1].squeeze()
                serie['ustar'].append(ustar_array[it:it + 1][0])

                pblh_array = pblh[:, latCoord:latCoord+1, longCoord:longCoord+1].squeeze()
                serie['PBLH'].append(pblh_array[it:it + 1][0])

                tsm_array = tsm[:, latCoord:latCoord+1, longCoord:longCoord+1].squeeze()
                serie['TSM'].append(tsm_array[it:it + 1][0]-273.15)

                it+=1

        
        df = pd.DataFrame(serie)
        
        df['WS'] = np.sqrt(df['u']**2+df['v']**2)
        df['q'] = df['q']/1000
        df['e'] = (df['q']*df['pressure'])/(0.622*(1-df['q'])+df['q'])
        df['es'] = 611.2*np.exp((17.67*df['T'])/(df['T']+243.5))
        df['ur'] = (df['e']/df['es'])*100
        df['q'] *= 1000
        df['pressure'] /= 100
        df['e'] /= 100  
        df['WD'] = 180/np.pi*np.arctan2(df['u'],df['v'])+180

        df = df[header+'WS WD e es ur'.split()]
        df.to_csv(foutput+name+'.dat',sep=',',index=False,na_rep='nan',float_format='%.7f')