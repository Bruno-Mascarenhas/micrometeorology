# -*- Coding: UTF-8 -*-
import glob
import os
from datetime import UTC, datetime
from itertools import product
from multiprocessing import Pool
from warnings import filterwarnings

import imageio
import matplotlib.pyplot as plt
import moviepy.editor as mp
import netCDF4
import numpy as np
from mpl_toolkits.basemap import Basemap
from utility import getFileNames

filterwarnings('ignore')

def toNum(arg):
    arg = str(arg)
    if len(arg) == 2:
        arg = '0'+arg
    else:
        arg = '00'+arg
    return arg

def getLowHigh(variable):
    varflat = variable.flatten()
    varlow, varhigh = np.amin(varflat), np.amax(varflat)
    return varlow, varhigh

def getLowHighWind(variable1, variable2):
    varflat1, varflat2 = variable1.flatten(), variable2.flatten()
    speed = np.sqrt(varflat1*varflat1 + varflat2*varflat2)
    varlow , varhigh = np.amin(speed), np.amax(speed)
    return varlow, varhigh

def drawmap(tipo,dataset):
    colormap = {'temperature':'jet','wind':'PuBu','vapor':'jet_r','pressure':'Blues','rain':'jet','HFX':'jet','LH':'jet','SWDOWN':'jet'}
    path = '/home/labmim/Build_WRF/d-output/figuras/'

    grade = dataset[39:42]; grade = grade.upper()
    dataset = netCDF4.Dataset(dataset)
    times_array = dataset.variables['Times'][:]
    dates = []; datesStr = []; names = []

    map = {'HFX':'HFX','LH':'LH','pressure':'PRES','rain':'RAIN','SWDOWN':'SWDOWN','temperature':'TEMP','vapor':'VAPOR','wind':'WIND'}
    week = {1:'Segunda',2:'Terça',3:'Quarta',4:'Quinta',5:'Sexta',6:'Sábado',7:'Domingo'}
    #desc = {'HFX':'Calor Sensível','pressure':'Pressão Atmosférica (Nível do Mar)','rain':'Precipitação','SWDOWN':'Radiação Global','temperature':'Temperatura (2 m)','vapor':'Umidade Específica','wind':'Velocidade do Vento (10 m)'}

    month = str(datetime.now().month); day = str(datetime.now().day)
    if len(month) == 1:
        month = '0'+str(month)
    if len(day) == 1:
        day = '0'+str(day)
    today = str(datetime.now().year)+month+day

    it = 0
    for time in times_array:
        currentTime = b''.join(time.tolist()).decode('UTF-8')
        current_date = datetime.strptime(currentTime, '%Y-%m-%d_%H:%M:%S')
        if it == 0:
            start = current_date.strftime("%d/%m/%Y %H") + " (UTC)\n"
            cd = current_date.replace(tzinfo=UTC).astimezone(tz=None)
            datesStr.append("\nInício Análise: "+start+"Previsão: "+cd.strftime("%d/%m/%Y %H")+"HL ("+week[cd.isoweekday()]+")")
            names.append(grade+"_"+map[tipo]+"_"+toNum(it))
            it+=1
            continue
        cd = current_date.replace(tzinfo=UTC).astimezone(tz=None)
        if tipo == 'SWDOWN' and (cd.hour < 6 or cd.hour > 18):
            datesStr.append("\nInício Análise: "+start+"Previsão: "+cd.strftime("%d/%m/%Y %H")+"HL ("+week[cd.isoweekday()]+")")
            names.append(grade+"_"+map[tipo]+"_"+toNum(it))
            it+=1
            continue
        datesStr.append("\nInício Análise: "+start+"Previsão: "+cd.strftime("%d/%m/%Y %H")+"HL ("+week[cd.isoweekday()]+")")
        names.append(grade+"_"+map[tipo]+"_"+toNum(it))
        dates.append((it,cd))
        it+=1

    #grafico da temperatura com as linhas de pressao
    if tipo == 'temperature':
        xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
        lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
        hlat, llat = np.amax(xlat), np.amin(xlat)
        hlong, llong = np.amax(xlong), np.amin(xlong)
        #temperatura
        var = dataset.variables['T2'][:,:,:].squeeze()
        varmin, varmax = getLowHigh(var)
        varmax = varmax - 273;        varmin = varmin - 273
        #pressao
        var2 = dataset.variables['PSFC'][:,:,:].squeeze()
        var2 /= 100

        for i,date in dates:

            celcius = var[i:i+1,:,:] - 273.15
            pressure = var2[i:i+1,:,:]

            plt.figure(figsize=(8,6))
            plt.title('Temperatura do Ar em 2m / Pressão Atm. Superfície ' + datesStr[i], fontsize=12)
            plt.suptitle(r"$^\circ\mathcal{C}$", fontsize=14, ha='center', x=0.79, y=0.75)
            plt.xlabel('Longitude', fontsize=11, labelpad=25)
            plt.ylabel('Latitude', fontsize=11, labelpad=60)

            m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='f',projection='merc',llcrnrlon= llong, llcrnrlat= llat,
                                                                                            urcrnrlon= hlong, urcrnrlat= hlat)

            x,y = m(lon, lat)

            m.drawcoastlines()

            m.drawparallels(np.arange(llat, hlat,abs(hlat-llat)/10), linewidth=0, labels=[1,0,0,0], color='r',zorder=0, fmt="%.2f")
            m.drawmeridians(np.arange(llong, hlong,abs(hlong-llong)/5), linewidth=0, labels=[0,0,0,1], color='r',zorder=0, fmt="%.2f")

            #temperatura
            m.contourf(x,y, np.squeeze(celcius), alpha=0.4, cmap=colormap[tipo], vmin=varmin, vmax=varmax)
            m.pcolor(x,y, np.squeeze(celcius), alpha=0.4, cmap=colormap[tipo], vmin=varmin, vmax=varmax)
            cb = plt.colorbar(shrink=0.5, pad=0.04)
            cb.ax.tick_params(labelsize=10)
            #pressao
            levels = [880,900,950,1000,1013]
            p = m.contour(x,y, np.squeeze(pressure), levels,linewidths=0.8, colors='black')
            plt.clabel(p,colors='black',fmt='%.0f')

            plt.savefig(path + names[i] +".png",bbox_inches='tight')
            plt.close()

    elif tipo == 'pressure':
        xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
        lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
        hlat, llat = np.amax(xlat), np.amin(xlat)
        hlong, llong = np.amax(xlong), np.amin(xlong)
        var = dataset.variables['PSFC'][:,:,:].squeeze()
        varmin, varmax = getLowHigh(var)
        varmax = varmax/100;        varmin = varmin/100

        for i,date in dates:

            mbar = var[i:i+1,:,:]/100

            plt.figure(figsize=(8,6))
            plt.title('Pressão Atmosférica ao Nível do Mar ' + datesStr[i], fontsize=12)
            plt.suptitle("${mBar}$", fontsize=14, ha='center', x=0.82, y=0.75)
            plt.xlabel('Longitude', fontsize=11, labelpad=25)
            plt.ylabel('Latitude', fontsize=11, labelpad=60)

            m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='f',projection='merc',llcrnrlon= llong, llcrnrlat= llat,
                                                                                            urcrnrlon= hlong, urcrnrlat= hlat)

            x,y = m(lon, lat)

            m.drawcoastlines()

            m.drawparallels(np.arange(llat, hlat,abs(hlat-llat)/10), linewidth=0, labels=[1,0,0,0], color='r',zorder=0, fmt="%.2f")
            m.drawmeridians(np.arange(llong, hlong,abs(hlong-llong)/5), linewidth=0, labels=[0,0,0,1], color='r',zorder=0, fmt="%.2f")

            m.contourf(x,y, np.squeeze(mbar), alpha=0.4, cmap=colormap[tipo], vmin=varmin, vmax=varmax)
            m.pcolor(x,y, np.squeeze(mbar), alpha=0.4, cmap=colormap[tipo], vmin=varmin, vmax=varmax)

            cb = plt.colorbar(shrink=0.5, pad=0.04)
            cb.ax.tick_params(labelsize=10)
            plt.savefig(path + names[i] +".png",bbox_inches='tight')
            plt.close()

    elif tipo == 'vapor':
        xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
        lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
        hlat, llat = np.amax(xlat), np.amin(xlat)
        hlong, llong = np.amax(xlong), np.amin(xlong)
        var = dataset.variables['Q2'][:,:,:].squeeze()
        varmin, varmax = getLowHigh(var)
        varmax = varmax*1000;        varmin = varmin*1000

        for i,date in dates:

            gkg = var[i:i+1,:,:]*1000

            plt.figure(figsize=(8,6))
            plt.title('Umidade Específica do Ar em 2m' + datesStr[i], fontsize=12)
            plt.suptitle("$g/kg \frac{m}{s}$", fontsize=14, ha='center', x=0.80, y=0.75)
            plt.xlabel('Longitude', fontsize=11, labelpad=25)
            plt.ylabel('Latitude', fontsize=11, labelpad=60)

            m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='f',projection='merc',llcrnrlon= llong, llcrnrlat= llat,
                                                                                            urcrnrlon= hlong, urcrnrlat= hlat)

            x,y = m(lon, lat)

            m.drawcoastlines()

            m.drawparallels(np.arange(llat, hlat,abs(hlat-llat)/10), linewidth=0, labels=[1,0,0,0], color='r',zorder=0, fmt="%.2f")
            m.drawmeridians(np.arange(llong, hlong,abs(hlong-llong)/5), linewidth=0, labels=[0,0,0,1], color='r',zorder=0, fmt="%.2f")

            m.contourf(x,y, np.squeeze(gkg), alpha=0.4, cmap=colormap[tipo], vmin=varmin, vmax=varmax)
            m.pcolor(x,y, np.squeeze(gkg), alpha=0.4, cmap=colormap[tipo], vmin=varmin, vmax=varmax)

            cb = plt.colorbar(shrink=0.5, pad=0.04)
            cb.ax.tick_params(labelsize=10)
            plt.savefig(path + names[i] +".png",bbox_inches='tight')
            plt.close()

    elif tipo == 'wind':
        xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
        lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
        hlat, llat = np.amax(xlat), np.amin(xlat)
        hlong, llong = np.amax(xlong), np.amin(xlong)
        u10 = dataset.variables['U10'][:].squeeze(); v10 = dataset.variables['V10'][:].squeeze()
        varmin, varmax = getLowHighWind(u10,v10)

        for i,date in dates:

            u = u10[i:i+1,:,:].squeeze()
            v = v10[i:i+1,:,:].squeeze()

            plt.figure(figsize=(8,6))
            plt.title('Velocidade do Vento em 10m ' + datesStr[i], fontsize=12)
            plt.xlabel('Longitude', fontsize=11, labelpad=25)
            plt.ylabel('Latitude', fontsize=11, labelpad=60)

            m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='f',projection='merc',llcrnrlon= llong, llcrnrlat= llat,urcrnrlon= hlong, urcrnrlat= hlat)
            x,y = m(lon, lat)

            yy = np.arange(0, y.shape[0], 3)
            xx = np.arange(0, x.shape[1], 3)

            speed = np.sqrt(u*u + v*v)
            points = np.meshgrid(yy, xx)

            m.contourf(x, y, np.sqrt(u*u + v*v), alpha = 0.4,cmap = 'Blues',vmin=varmin,vmax=varmax)
            cs = m.pcolor(x,y,np.squeeze(speed), alpha=0.4, cmap=colormap[tipo], vmin=varmin, vmax=varmax)

            widths = np.linspace(0, 2, xx.size)

            nx = x[points];            ny = y[points];            nu = u[points];            nv = v[points]
            #x[points], y[points], u[points], v[points]
            if 'D01' in grade:
                Q = m.quiver(nx[::2,::2],ny[::2,::2] ,nu[::2,::2] ,nv[::2,::2] , scale_units='xy', width=0.0035,scale=0.0001)
                qk = plt.quiverkey(Q,1.095, 0.78, 0.00019,label= r'$\frac{m}{s}$', fontproperties={'size': 18}, labelpos='N')

            elif 'D02' in grade:
                Q = m.quiver(nx[::,::],ny[::,::] ,nu[::,::] ,nv[::,::] , scale_units='xy', width=0.0035,scale=0.0003)
                qk = plt.quiverkey(Q,1.095, 0.78, 0.00039,label= r'$\frac{m}{s}$', fontproperties={'size': 18}, labelpos='N')

            else:
                Q = m.quiver(nx[::2,::2],ny[::2,::2] ,nu[::2,::2] ,nv[::2,::2] , scale_units='xy', width=0.0035,scale=0.001)
                qk = plt.quiverkey(Q,1.095, 0.78, 0.0019,label= r'$\frac{m}{s}$', fontproperties={'size': 18}, labelpos='N')

            m.drawcoastlines(color='0.15')

            m.drawparallels(np.arange(llat, hlat,abs(hlat-llat)/10), linewidth=0, labels=[1,0,0,0], color='r',zorder=0, fmt="%.2f")
            m.drawmeridians(np.arange(llong, hlong,abs(hlong-llong)/5), linewidth=0, labels=[0,0,0,1], color='r',zorder=0, fmt="%.2f")

            cb = plt.colorbar(shrink=0.5, pad=0.04)
            cb.ax.tick_params(labelsize=10)

            ##################################
            #nx, ny = m(-38.80,-12.5)
            #m.scatter(nx,ny,marker='D',color='m')
            ##################################

            plt.savefig(path + names[i] +".png",bbox_inches='tight')
            plt.close()

    elif tipo == 'rain':
        xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
        lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
        hlat, llat = np.amax(xlat), np.amin(xlat)
        hlong, llong = np.amax(xlong), np.amin(xlong)
        var = dataset.variables['RAINC'][:,:,:].squeeze()
        var2 = dataset.variables['RAINNC'][:,:,:].squeeze()
        var = var + var2
        varmin, varmax = getLowHigh(var)

        for i,date in dates:

            mm = var[i:i+1,:,:]

            plt.figure(figsize=(8,6))
            plt.title('Precipitação Acumulada em 6h' + datesStr[i], fontsize=12)
            plt.suptitle("$mm$", fontsize=14, ha='center', x=0.79, y=0.75)
            plt.xlabel('Longitude', fontsize=11, labelpad=25)
            plt.ylabel('Latitude', fontsize=11, labelpad=60)

            m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='f',projection='merc',llcrnrlon= llong, llcrnrlat= llat,
                                                                                            urcrnrlon= hlong, urcrnrlat= hlat)

            x,y = m(lon, lat)

            m.drawcoastlines()
            m.drawparallels(np.arange(llat, hlat,abs(hlat-llat)/10), linewidth=0, labels=[1,0,0,0], color='r',zorder=0, fmt="%.2f")
            m.drawmeridians(np.arange(llong, hlong,abs(hlong-llong)/5), linewidth=0, labels=[0,0,0,1], color='r',zorder=0, fmt="%.2f")

            m.contourf(x,y, np.squeeze(mm), alpha = 0.4, cmap = colormap[tipo], vmin=varmin, vmax=varmax)
            m.pcolor(x,y, np.squeeze(mm), alpha = 0.4, cmap = colormap[tipo], vmin=varmin, vmax=varmax)

            cb = plt.colorbar(shrink=0.5, pad=0.04)
            cb.ax.tick_params(labelsize=10)
            plt.savefig(path + names[i] +".png",bbox_inches='tight')
            plt.close()

    elif tipo == 'HFX':
        xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
        lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
        hlat, llat = np.amax(xlat), np.amin(xlat)
        hlong, llong = np.amax(xlong), np.amin(xlong)
        var = dataset.variables['HFX'][:,:,:].squeeze()
        varmin, varmax = getLowHigh(var)

        for i,date in dates:

            mm = var[i:i+1,:,:]

            plt.figure(figsize=(8,6))
            plt.title('Calor Sensível em Superfície ' + datesStr[i], fontsize=12)
            plt.suptitle("$ W m ^ {-2} $", fontsize=14, ha='center', x=0.82, y=0.75)
            plt.xlabel('Longitude', fontsize=11, labelpad=25)
            plt.ylabel('Latitude', fontsize=11, labelpad=60)

            m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='f',projection='merc',llcrnrlon= llong, llcrnrlat= llat,
                                                                                            urcrnrlon= hlong, urcrnrlat= hlat)

            x,y = m(lon, lat)

            m.drawcoastlines()
            m.drawparallels(np.arange(llat, hlat,abs(hlat-llat)/10), linewidth=0, labels=[1,0,0,0], color='r',zorder=0, fmt="%.2f")
            m.drawmeridians(np.arange(llong, hlong,abs(hlong-llong)/5), linewidth=0, labels=[0,0,0,1], color='r',zorder=0, fmt="%.2f")

            m.contourf(x,y, np.squeeze(mm), alpha = 0.4, cmap = colormap[tipo], vmin=varmin, vmax=varmax)
            m.pcolor(x,y, np.squeeze(mm), alpha = 0.4, cmap = colormap[tipo], vmin=varmin, vmax=varmax)

            cb = plt.colorbar(shrink=0.5, pad=0.04)
            cb.ax.tick_params(labelsize=10)
            plt.savefig(path + names[i] +".png",bbox_inches='tight')
            plt.close()

    elif tipo == 'LH':
        xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
        lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
        hlat, llat = np.amax(xlat), np.amin(xlat)
        hlong, llong = np.amax(xlong), np.amin(xlong)
        var = dataset.variables['LH'][:,:,:].squeeze()
        varmin, varmax = getLowHigh(var)

        for i,date in dates:

            mm = var[i:i+1,:,:]

            plt.figure(figsize=(8,6))
            plt.title('Calor Latente em Superfície' + datesStr[i], fontsize=12)
            plt.suptitle("$ W m ^ {-2} $", fontsize=14, ha='center', x=0.82, y=0.75)
            plt.xlabel('Longitude', fontsize=11, labelpad=25)
            plt.ylabel('Latitude', fontsize=11, labelpad=60)

            m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='f',projection='merc',llcrnrlon= llong, llcrnrlat= llat,
                                                                                            urcrnrlon= hlong, urcrnrlat= hlat)

            x,y = m(lon, lat)

            m.drawcoastlines()
            m.drawparallels(np.arange(llat, hlat,abs(hlat-llat)/10), linewidth=0, labels=[1,0,0,0], color='r',zorder=0, fmt="%.2f")
            m.drawmeridians(np.arange(llong, hlong,abs(hlong-llong)/5), linewidth=0, labels=[0,0,0,1], color='r',zorder=0, fmt="%.2f")

            m.contourf(x,y, np.squeeze(mm), alpha = 0.4, cmap = colormap[tipo], vmin=varmin, vmax=varmax)
            m.pcolor(x,y, np.squeeze(mm), alpha = 0.4, cmap = colormap[tipo], vmin=varmin, vmax=varmax)

            cb = plt.colorbar(shrink=0.5, pad=0.04)
            cb.ax.tick_params(labelsize=10)
            plt.savefig(path + names[i] +".png",bbox_inches='tight')
            plt.close()

    elif tipo == 'SWDOWN':
        xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
        lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
        hlat, llat = np.amax(xlat), np.amin(xlat)
        hlong, llong = np.amax(xlong), np.amin(xlong)
        var = dataset.variables['SWDOWN'][:,:,:].squeeze()
        varmin, varmax = getLowHigh(var)

        for i,date in dates:

            mm = var[i:i+1,:,:]

            plt.figure(figsize=(8,6))
            plt.title('Radiação Solar Global ' + datesStr[i], fontsize=12)
            plt.suptitle("$ W m ^ {-2} $", fontsize=14, ha='center', x=0.82, y=0.75)
            plt.xlabel('Longitude', fontsize=11, labelpad=25)
            plt.ylabel('Latitude', fontsize=11, labelpad=60)

            m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='f',projection='merc',llcrnrlon= llong, llcrnrlat= llat,
                                                                                            urcrnrlon= hlong, urcrnrlat= hlat)

            x,y = m(lon, lat)

            m.drawcoastlines()
            m.drawparallels(np.arange(llat, hlat,abs(hlat-llat)/10), linewidth=0, labels=[1,0,0,0], color='r',zorder=0, fmt="%.2f")
            m.drawmeridians(np.arange(llong, hlong,abs(hlong-llong)/5), linewidth=0, labels=[0,0,0,1], color='r',zorder=0, fmt="%.2f")

            m.contourf(x,y, np.squeeze(mm), alpha = 0.4, cmap = colormap[tipo], vmin=varmin, vmax=varmax)
            m.pcolor(x,y, np.squeeze(mm), alpha = 0.4, cmap = colormap[tipo], vmin=varmin, vmax=varmax)

            cb = plt.colorbar(shrink=0.5, pad=0.04)
            cb.ax.tick_params(labelsize=10)
            plt.savefig(path + names[i] +".png",bbox_inches='tight')
            plt.close()

def generateGifs(name, files, path_output):
    files = glob.glob(files+'*.png')
    files.sort(key=os.path.getmtime)

    images = [imageio.imread(x) for x in files]
    imageio.mimsave(path_output+name+'.gif',images,format='GIF',duration=0.5)

    clip = mp.VideoFileClip(path_output+name+'.gif')
    clip.write_videofile(path_output+name+'.webm')

    os.remove(path_output+name+'.gif')

if __name__ == '__main__':

    #graphs.png for each hour
    #args = ['temperature','pressure','vapor','wind','rain','HFX','LH','SWDOWN']
    args = ['temperature','vapor','wind','SWDOWN','rain']

    path = '/home/labmim/Build_WRF/d-output'
    files = getFileNames(path)

    final = product(args,files)
    processes = Pool(processes=20)

    processes.starmap(drawmap,final)

    #make a .webm with the graphs
    path = '/home/labmim/Build_WRF/d-output/figuras/'
    names = ['_RAIN','_SWDOWN','_TEMP','_VAPOR','_WIND']
    grade = ['D01','D02','D03']

    files = [(g+n[1:], path+g+n, path) for g in grade for n in names]

    processes.starmap(generateGifs,files)

