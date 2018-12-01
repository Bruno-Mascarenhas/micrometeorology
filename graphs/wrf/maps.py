import numpy as np
import netCDF4
from datetime import timezone
from datetime import datetime
import os
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import json

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
dataset = netCDF4.Dataset(os.path.join(__location__, 'new-wrfout_d03_2018-04-28_00_00_00.nc'))
with open(os.path.join(__location__, 'script.json')) as data:
    settings = json.load(data)

path = 'C:\\Users\\BrunoM\\Documents\\working now\\modelo\\'
times_array = dataset.variables['Times'][:]
dates = []; datesStr = []

for time in times_array:
    currentTime = b''.join(time.tolist()).decode('UTF-8')
    current_date = datetime.strptime(currentTime, '%Y-%m-%d_%H:%M:%S')
    cd = current_date.replace(tzinfo=timezone.utc).astimezone(tz=None)
    datesStr.append(cd.strftime("%Y-%m-%d_%H"))
    dates.append(cd)

def getLowHigh(variable):
    varflat = variable.flatten()
    varlow, varhigh = np.amin(varflat), np.amax(varflat)
    return varlow, varhigh

def getLowHighWind(variable1, variable2):
    varflat1, varflat2 = variable1.flatten(), variable2.flatten()
    speed = np.sqrt(varflat1*varflat1 + varflat2*varflat2)
    varlow , varhigh = np.amin(speed), np.amax(speed)
    return varlow, varhigh

def drawmap(tipo,dates,datesStr):
    if tipo == 'temperature':
        xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
        lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
        hlat, llat = np.amax(xlat), np.amin(xlat)
        hlong, llong = np.amax(xlong), np.amin(xlong)
        var = dataset.variables['T2'][:,:,:].squeeze()
        varmin, varmax = getLowHigh(var);
        varmax = varmax - 273;        varmin = varmin - 273

        for i in range(len(dates)):

            celcius = var[i:i+1,:,:] - 273.15
            colormap = settings[tipo]['colormap']

            plt.figure(figsize=(8,6))
            plt.title('Temperatura (2 m) ' + datesStr[i], fontsize=12)
            plt.suptitle("$^\circ\mathcal{C}$", fontsize=18, ha='center', x=0.79, y=0.75)
            plt.xlabel('Longitude', fontsize=11, labelpad=25)
            plt.ylabel('Latitude', fontsize=11, labelpad=60)

            #cylindrical ||mercator || miller cylindrical || gall stereo || prjection
            m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='h',projection='merc',llcrnrlon= llong, llcrnrlat= llat,
                                                                                            urcrnrlon= hlong, urcrnrlat= hlat)

            x,y = m(lon, lat)

            m.drawcoastlines()
            m.drawparallels(np.arange(hlat, llat, 0.06), linewidth=0, labels=[1,0,0,1], color='r', zorder=0, fmt="%.2f")
            m.drawmeridians(np.arange(hlong, llong, 0.07), linewidth=0, labels=[1,0,0,1], color='r', zorder=0, fmt="%.2f")

            m.contourf(x, y, np.squeeze(celcius), alpha=0.4, cmap=colormap, vmin=varmin, vmax=varmax)
            m.pcolor(x,y, np.squeeze(celcius), alpha=0.4, cmap=colormap, vmin=varmin, vmax=varmax)

            cb = plt.colorbar(shrink=0.5, pad=0.04)
            cb.ax.tick_params(labelsize=10)
            plt.savefig(path+'\\'+tipo+'\\'+ datesStr[i] +".png",bbox_inches='tight')
            plt.close()

    elif tipo == 'pressure':
        xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
        lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
        hlat, llat = np.amax(xlat), np.amin(xlat)
        hlong, llong = np.amax(xlong), np.amin(xlong)
        var = dataset.variables['PSFC'][:,:,:].squeeze()
        varmin, varmax = getLowHigh(var);
        varmax = varmax/100;        varmin = varmin/100

        for i in range(len(dates)):

            mbar = var[i:i+1,:,:]/100
            colormap = settings[tipo]['colormap']

            plt.figure(figsize=(8,6))
            plt.title('Pressão Atmosférica (Nível do Mar) ' + datesStr[i], fontsize=12)
            plt.suptitle("${mBar}$", fontsize=18, ha='center', x=0.79, y=0.75)
            plt.xlabel('Longitude', fontsize=11, labelpad=25)
            plt.ylabel('Latitude', fontsize=11, labelpad=60)

            #cylindrical ||mercator || miller cylindrical || gall stereo || prjection
            m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='h',projection='merc',llcrnrlon= llong, llcrnrlat= llat,
                                                                                            urcrnrlon= hlong, urcrnrlat= hlat)

            x,y = m(lon, lat)

            m.drawcoastlines()
            m.drawparallels(np.arange(hlat, llat, 0.06), linewidth=0, labels=[1,0,0,1], color='r', zorder=0, fmt="%.2f")
            m.drawmeridians(np.arange(hlong, llong, 0.07), linewidth=0, labels=[1,0,0,1], color='r', zorder=0, fmt="%.2f")

            m.contourf(x, y, np.squeeze(mbar), alpha=0.4, cmap=colormap, vmin=varmin, vmax=varmax)
            m.pcolor(x,y, np.squeeze(mbar), alpha=0.4, cmap=colormap, vmin=varmin, vmax=varmax)

            cb = plt.colorbar(shrink=0.5, pad=0.04)
            cb.ax.tick_params(labelsize=10)
            plt.savefig(path+'\\'+tipo+'\\'+ datesStr[i] +".png",bbox_inches='tight')
            plt.close()

    elif tipo == 'vapor':
        xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
        lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
        hlat, llat = np.amax(xlat), np.amin(xlat)
        hlong, llong = np.amax(xlong), np.amin(xlong)
        var = dataset.variables['Q2'][:,:,:].squeeze()
        varmin, varmax = getLowHigh(var);
        varmax = varmax*1000;        varmin = varmin*1000

        for i in range(len(dates)):

            gkg = var[i:i+1,:,:]*1000
            colormap = settings[tipo]['colormap']

            plt.figure(figsize=(8,6))
            plt.title('Umidade Específica ' + datesStr[i], fontsize=12)
            plt.suptitle("$g/kg \frac{m}{s}$", fontsize=18, ha='center', x=0.79, y=0.75)
            plt.xlabel('Longitude', fontsize=11, labelpad=25)
            plt.ylabel('Latitude', fontsize=11, labelpad=60)

            #cylindrical ||mercator || miller cylindrical || gall stereo || prjection
            m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='h',projection='merc',llcrnrlon= llong, llcrnrlat= llat,
                                                                                            urcrnrlon= hlong, urcrnrlat= hlat)

            x,y = m(lon, lat)

            m.drawcoastlines()
            m.drawparallels(np.arange(hlat, llat, 0.06), linewidth=0, labels=[1,0,0,1], color='r', zorder=0, fmt="%.2f")
            m.drawmeridians(np.arange(hlong, llong, 0.07), linewidth=0, labels=[1,0,0,1], color='r', zorder=0, fmt="%.2f")

            m.contourf(x, y, np.squeeze(gkg), alpha=0.4, cmap=colormap, vmin=varmin, vmax=varmax)
            m.pcolor(x,y, np.squeeze(gkg), alpha=0.4, cmap=colormap, vmin=varmin, vmax=varmax)

            cb = plt.colorbar(shrink=0.5, pad=0.04)
            cb.ax.tick_params(labelsize=10)
            plt.savefig(path+'\\'+tipo+'\\'+ datesStr[i] +".png",bbox_inches='tight')
            plt.close()

    elif tipo == 'wind':
        xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
        lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
        hlat, llat = np.amax(xlat), np.amin(xlat)
        hlong, llong = np.amax(xlong), np.amin(xlong)
        u10 = dataset.variables['U10'][:].squeeze()
        v10 = dataset.variables['V10'][:].squeeze()
        varmin, varmax = getLowHighWind(u10,v10);

        for i in range(len(dates)):

            u = u10[i:i+1,:,:].squeeze()
            v = v10[i:i+1,:,:].squeeze()
            colormap = settings[tipo]['colormap']

            plt.figure(figsize=(8,6))
            plt.title('Velocidade do Vento (10 m) ' + datesStr[i], fontsize=12)
            plt.xlabel('Longitude', fontsize=11, labelpad=25)
            plt.ylabel('Latitude', fontsize=11, labelpad=60)

            #cylindrical ||mercator || miller cylindrical || gall stereo || prjection
            m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='h',projection='merc',llcrnrlon= llong, llcrnrlat= llat,
                                                                                            urcrnrlon= hlong, urcrnrlat= hlat)
            x,y = m(lon, lat)

            yy = np.arange(0, y.shape[0], 3)
            xx = np.arange(0, x.shape[1], 3)

            speed = np.sqrt(u*u + v*v)
            points = np.meshgrid(yy, xx)

            m.contourf(x, y, np.sqrt(u*u + v*v), alpha=0.4, cmap=colormap, vmin=varmin, vmax=varmax)

            cs = m.pcolor(x,y,np.squeeze(speed), alpha=0.4, cmap=colormap, vmin=varmin, vmax=varmax)

            widths = np.linspace(0, 2, xx.size)

            Q = m.quiver(x[points], y[points], u[points], v[points], scale_units='xy', width= 0.0035)

            qk = plt.quiverkey(Q, 1.095,  0.78, 2, r'$2 \frac{m}{s}$', fontproperties={'size': 18}, labelpos='N')

            m.drawcoastlines(color='0.15')

            m.drawparallels(np.arange(hlat, llat, 0.06), linewidth=0, labels=[1,0,0,1], color='r', zorder=0, fmt="%.2f")
            m.drawmeridians(np.arange(hlong, llong, 0.07), linewidth=0, labels=[1,0,0,1], color='r', zorder=0, fmt="%.2f")

            cb = plt.colorbar(shrink=0.5, pad=0.04)
            cb.ax.tick_params(labelsize=10)

            plt.savefig(path+'\\'+tipo+'\\'+ datesStr[i] +".png",bbox_inches='tight')
            plt.close()

    elif tipo == 'rain':
        xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
        lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
        hlat, llat = np.amax(xlat), np.amin(xlat)
        hlong, llong = np.amax(xlong), np.amin(xlong)
        var = dataset.variables['RAINC'][:,:,:].squeeze()
        varmin, varmax = getLowHigh(var);

        for i in range(len(dates)):

            mm = var[i:i+1,:,:]
            colormap = settings[tipo]['colormap']

            plt.figure(figsize=(8,6))
            plt.title('Precipitação ' + datesStr[i], fontsize=12)
            plt.suptitle("$mm$", fontsize=18, ha='center', x=0.79, y=0.75)
            plt.xlabel('Longitude', fontsize=11, labelpad=25)
            plt.ylabel('Latitude', fontsize=11, labelpad=60)

            #cylindrical || mercator || miller cylindrical || gall stereo || prjection
            m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='h',projection='merc',llcrnrlon= llong, llcrnrlat= llat,
                                                                                            urcrnrlon= hlong, urcrnrlat= hlat)

            x,y = m(lon, lat)

            m.drawcoastlines()
            m.drawparallels(np.arange(hlat, llat, 0.06), linewidth=0, labels=[1,0,0,1], color='r', zorder=0, fmt="%.2f")
            m.drawmeridians(np.arange(hlong, llong, 0.07), linewidth=0, labels=[1,0,0,1], color='r', zorder=0, fmt="%.2f")

            m.contourf(x, y, np.squeeze(mm), alpha = 0.4, cmap = colormap, vmin=varmin, vmax=varmax)
            m.pcolor(x,y, np.squeeze(mm), alpha = 0.4, cmap = colormap, vmin=varmin, vmax=varmax)

            cb = plt.colorbar(shrink=0.5, pad=0.04)
            cb.ax.tick_params(labelsize=10)
            plt.savefig(path+'\\'+tipo+'\\'+ datesStr[i] +".png",bbox_inches='tight')
            plt.close()

    elif tipo == 'HFX':
        xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
        lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
        hlat, llat = np.amax(xlat), np.amin(xlat)
        hlong, llong = np.amax(xlong), np.amin(xlong)
        var = dataset.variables['HFX'][:,:,:].squeeze()
        varmin, varmax = getLowHigh(var);

        for i in range(len(dates)):

            mm = var[i:i+1,:,:]
            colormap = settings[tipo]['colormap']

            plt.figure(figsize=(8,6))
            plt.title('Calor Sensível ' + datesStr[i], fontsize=12)
            plt.suptitle("$ W m ^ {-2} $", fontsize=18, ha='center', x=0.79, y=0.75)
            plt.xlabel('Longitude', fontsize=11, labelpad=25)
            plt.ylabel('Latitude', fontsize=11, labelpad=60)

            #cylindrical ||mercator || miller cylindrical || gall stereo || prjection
            m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='h',projection='merc',llcrnrlon= llong, llcrnrlat= llat,
                                                                                            urcrnrlon= hlong, urcrnrlat= hlat)

            x,y = m(lon, lat)

            m.drawcoastlines()
            m.drawparallels(np.arange(hlat, llat, 0.06), linewidth=0, labels=[1,0,0,1], color='r', zorder=0, fmt="%.2f")
            m.drawmeridians(np.arange(hlong, llong, 0.07), linewidth=0, labels=[1,0,0,1], color='r', zorder=0, fmt="%.2f")

            m.contourf(x, y, np.squeeze(mm), alpha = 0.4, cmap = colormap, vmin=varmin, vmax=varmax)
            m.pcolor(x,y, np.squeeze(mm), alpha = 0.4, cmap = colormap, vmin=varmin, vmax=varmax)

            cb = plt.colorbar(shrink=0.5, pad=0.04)
            cb.ax.tick_params(labelsize=10)
            plt.savefig(path+'\\'+tipo+'\\'+ datesStr[i] +".png",bbox_inches='tight')
            plt.close()

    elif tipo == 'LH':
        xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
        lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
        hlat, llat = np.amax(xlat), np.amin(xlat)
        hlong, llong = np.amax(xlong), np.amin(xlong)
        var = dataset.variables['LH'][:,:,:].squeeze()
        varmin, varmax = getLowHigh(var);

        for i in range(len(dates)):

            mm = var[i:i+1,:,:]
            colormap = settings[tipo]['colormap']

            plt.figure(figsize=(8,6))
            plt.title('Calor Latente ' + datesStr[i], fontsize=12)
            plt.suptitle("$ W m ^ {-2} $", fontsize=18, ha='center', x=0.79, y=0.75)
            plt.xlabel('Longitude', fontsize=11, labelpad=25)
            plt.ylabel('Latitude', fontsize=11, labelpad=60)

            #cylindrical ||mercator || miller cylindrical || gall stereo || prjection
            m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='h',projection='merc',llcrnrlon= llong, llcrnrlat= llat,
                                                                                            urcrnrlon= hlong, urcrnrlat= hlat)

            x,y = m(lon, lat)

            m.drawcoastlines()
            m.drawparallels(np.arange(hlat, llat, 0.06), linewidth=0, labels=[1,0,0,1], color='r', zorder=0, fmt="%.2f")
            m.drawmeridians(np.arange(hlong, llong, 0.07), linewidth=0, labels=[1,0,0,1], color='r', zorder=0, fmt="%.2f")

            m.contourf(x, y, np.squeeze(mm), alpha = 0.4, cmap = colormap, vmin=varmin, vmax=varmax)
            m.pcolor(x,y, np.squeeze(mm), alpha = 0.4, cmap = colormap, vmin=varmin, vmax=varmax)

            cb = plt.colorbar(shrink=0.5, pad=0.04)
            cb.ax.tick_params(labelsize=10)
            plt.savefig(path+'\\'+tipo+'\\'+ datesStr[i] +".png",bbox_inches='tight')
            plt.close()

    elif tipo == 'SWDOWN':
        xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
        lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
        hlat, llat = np.amax(xlat), np.amin(xlat)
        hlong, llong = np.amax(xlong), np.amin(xlong)
        var = dataset.variables['SWDOWN'][:,:,:].squeeze()
        varmin, varmax = getLowHigh(var);

        for i in range(len(dates)):

            mm = var[i:i+1,:,:]
            colormap = settings[tipo]['colormap']

            plt.figure(figsize=(8,6))
            plt.title('Radiação Global ' + datesStr[i], fontsize=12)
            plt.suptitle("$ W m ^ {-2} $", fontsize=18, ha='center', x=0.79, y=0.75)
            plt.xlabel('Longitude', fontsize=11, labelpad=25)
            plt.ylabel('Latitude', fontsize=11, labelpad=60)

            #cylindrical ||mercator || miller cylindrical || gall stereo || prjection
            m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='h',projection='merc',llcrnrlon= llong, llcrnrlat= llat,
                                                                                            urcrnrlon= hlong, urcrnrlat= hlat)

            x,y = m(lon, lat)

            m.drawcoastlines()
            m.drawparallels(np.arange(hlat, llat, 0.06), linewidth=0, labels=[1,0,0,1], color='r', zorder=0, fmt="%.2f")
            m.drawmeridians(np.arange(hlong, llong, 0.07), linewidth=0, labels=[1,0,0,1], color='r', zorder=0, fmt="%.2f")

            m.contourf(x, y, np.squeeze(mm), alpha = 0.4, cmap = colormap, vmin=varmin, vmax=varmax)
            m.pcolor(x,y, np.squeeze(mm), alpha = 0.4, cmap = colormap, vmin=varmin, vmax=varmax)

            cb = plt.colorbar(shrink=0.5, pad=0.04)
            cb.ax.tick_params(labelsize=10)
            plt.savefig(path+'\\'+tipo+'\\'+ datesStr[i] +".png",bbox_inches='tight')
            plt.close()

drawmap('temperature',dates,datesStr)
drawmap('pressure',dates,datesStr)
drawmap('vapor',dates,datesStr)
drawmap('wind',dates,datesStr)
drawmap('rain',dates,datesStr)
drawmap('HFX',dates,datesStr)
drawmap('LH',dates,datesStr)
drawmap('SWDOWN',dates,datesStr)