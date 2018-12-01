import numpy as np
import netCDF4
from math import sqrt
import utility
import os
from datetime import datetime
from datetime import timezone

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
filesPath = "C:\\Users\\BrunoM\\Documents\\working now\\wrf-2018"

def generateSeries(filesTxt):
    serie = []
    dataset = netCDF4.Dataset(filesTxt[0])
    latCoord, longCoord = utility.getLatLon(dataset, -12.999095, -38.508166, filesTxt[0])
    for file in filesTxt:
        dataset = netCDF4.Dataset(file)
        temp = dataset.variables['T2'][:]
        th2 = dataset.variables['TH2'][:]
        u10 = dataset.variables['U10'][:]
        v10 = dataset.variables['V10'][:]
        p = dataset.variables['PSFC'][:]
        qv = dataset.variables['Q2'][:]
        rainc = dataset.variables['RAINC'][:]
        hfx = dataset.variables['HFX'][:]
        lh = dataset.variables['LH'][:]
        swdown = dataset.variables['SWDOWN'][:]
        times_array = dataset.variables['Times'][:]

        #latCoord,longCoord = utility.getLatLon(dataset,-12.999095,-38.508166,file)

        it =0
        flag = True
        for time in times_array:
            if it == 24:
                break
            currentTime = b''.join(time.tolist()).decode('UTF-8')
            current_date = datetime.strptime(currentTime,'%Y-%m-%d_%H:%M:%S')
            cd = current_date.replace(tzinfo=timezone.utc).astimezone(tz=None)

            year = int(cd.year);    month = int(cd.month);    day = int(cd.day);    hour = int(cd.hour)

            temp_array = temp[:, latCoord:latCoord+1, longCoord:longCoord+1].squeeze()
            tempfix = temp_array[it:it + 1][0]

            th2_array = th2[:, latCoord:latCoord+1, longCoord:longCoord+1].squeeze()
            th2fix = th2_array[it:it + 1][0]


            u10_array = u10[:, latCoord:latCoord+1, longCoord:longCoord+1].squeeze()
            u10fix = u10_array[it:it + 1][0]


            v10_array = v10[:, latCoord:latCoord+1, longCoord:longCoord+1].squeeze()
            v10fix = v10_array[it:it + 1][0]


            p_array = p[:, latCoord:latCoord+1, longCoord:longCoord+1].squeeze()
            pfix = p_array[it:it + 1][0]


            qv_array = qv[:, latCoord:latCoord+1, longCoord:longCoord+1].squeeze()
            qvfix = qv_array[it:it + 1][0]


            rainc_array = rainc[:, latCoord:latCoord+1, longCoord:longCoord+1].squeeze()
            raincfix = rainc_array[it:it + 1][0]

            hfx_array = hfx[:, latCoord:latCoord+1, longCoord:longCoord+1].squeeze()
            hfxfix = hfx_array[it:it + 1][0]


            lh_array = lh[:, latCoord:latCoord+1, longCoord:longCoord+1].squeeze()
            lhfix = lh_array[it:it + 1][0]


            swdown_array = swdown[:, latCoord:latCoord+1, longCoord:longCoord+1].squeeze()
            swdownfix = swdown_array[it:it + 1][0]

            serie.append([year,month,day,hour,tempfix,u10fix,v10fix,pfix,qvfix,th2fix,raincfix,hfxfix,lhfix,swdownfix])
            it+=1

    #header = "year,month,day,hour,tempfix,u10fix,v10fix,pfix,qvfix,th2fix,raincfix,hfxfix,lhfix,swdownfix"
    np.savetxt(os.path.join(__location__,'series.dat'), serie, fmt='%.10g', delimiter=" ")

txt = utility.getFileNames(filesPath)
generateSeries(txt)