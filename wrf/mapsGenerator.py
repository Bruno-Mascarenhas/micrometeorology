# -*- Coding: UTF-8 -*-
#coding: utf-8
import numpy as np
import netCDF4
from datetime import timezone
from datetime import datetime
import os
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import json
from sys import exit
from warnings import filterwarnings
filterwarnings('ignore')

colormap = {'temperature':'jet','wind':'PuBu','vapor':'jet_r','pressure':'Blues','rain':'jet','HFX':'jet','LH':'jet','SWDOWN':'jet'}

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

#def drawmap(tipo,dataset):

def drawMapSeason(variables, files_path, out_path, name, hour):
    colormap = {'T':'jet','wind':'PuBu','q':'jet_r','pressure':'Blues','precip':'jet','hfx':'jet','lh':'jet','sw_dw':'jet'}
    #variables = T,q,wind
    
    data = netCDF4.Dataset(files_path[0])
    xlat, xlong = data.variables['XLAT'][:,:,:], data.variables['XLONG'][:,:,:]
    lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
    hlat, llat = np.amax(xlat), np.amin(xlat)
    hlong, llong = np.amax(xlong), np.amin(xlong)
    data.close()

    for var in variables:
        if var == 'T':
            temperatura = None; cont = 0
            for file in files_path:
                dataset = netCDF4.Dataset(file)
                
                times_array = dataset.variables['Times'][:]

                for i,time in enumerate(times_array):
                    currentTime = b''.join(time.tolist()).decode('UTF-8')
                    current_date = datetime.strptime(currentTime, '%Y-%m-%d_%H:%M:%S')
                    cd = current_date.replace(tzinfo=timezone.utc).astimezone(tz=None)
                    if cd.hour == hour:
                        tmp = dataset.variables['T2'][:,:,:].squeeze()
                        celcius = tmp[i:i+1,:,:] - 273.15
                        if temperatura is None:
                            temperatura = celcius
                        else:
                            temperatura += celcius
                        cont+=1

            temperatura = temperatura/cont
            varmin, varmax = getLowHigh(temperatura)

            plt.figure(figsize=(8,6))
            #plt.title('Temperatura (2 m)',fontsize=12)
            plt.suptitle("$^\circ\mathcal{C}$", fontsize=18, ha='center', x=0.79, y=0.75)
            plt.xlabel('Long (°)', fontsize=14, labelpad=25)
            plt.ylabel('Lat (°)', fontsize=14, labelpad=60)

            m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='f',projection='merc',llcrnrlon= llong, llcrnrlat= llat, urcrnrlon= hlong, urcrnrlat= hlat)

            x,y = m(lon, lat)

            m.drawcoastlines()

            m.drawparallels(np.arange(llat, hlat,abs(hlat-llat)/7), linewidth=0, labels=[1,0,0,0], color='r',zorder=0, fmt="%.1f", fontsize=14)
            m.drawmeridians(np.arange(llong, hlong,abs(hlong-llong)/5), linewidth=0, labels=[0,0,0,1], color='r',zorder=0, fmt="%.1f", fontsize=14)


            m.contourf(x,y, np.squeeze(temperatura), alpha=0.4, cmap=colormap['T'], vmin=14, vmax=32)
            m.pcolormesh(x,y, np.squeeze(temperatura), alpha=0.4, cmap=colormap['T'], vmin=14, vmax=32)

            cb = plt.colorbar(shrink=0.5, pad=0.04)
            cb.ax.tick_params(labelsize=10)
            plt.savefig(out_path+name+var+".png",bbox_inches='tight')
            plt.close()

        elif var == 'q':
            umidade = None; cont = 0
            for file in files_path:
                dataset = netCDF4.Dataset(file)
                
                times_array = dataset.variables['Times'][:]

                for i,time in enumerate(times_array):
                    currentTime = b''.join(time.tolist()).decode('UTF-8')
                    current_date = datetime.strptime(currentTime, '%Y-%m-%d_%H:%M:%S')
                    cd = current_date.replace(tzinfo=timezone.utc).astimezone(tz=None)
                    if cd.hour == hour:
                        tmp = dataset.variables['Q2'][:,:,:].squeeze()
                        q2 = tmp[i:i+1,:,:] * 1000
                        if umidade is None:
                            umidade = q2
                        else:
                            umidade += q2
                        cont+=1
            
            umidade = umidade/cont
            varmin, varmax = getLowHigh(umidade)

            plt.figure(figsize=(8,6))
            #plt.title('Umidade Específica ', fontsize=12)
            plt.suptitle("$g/kg \frac{m}{s}$", fontsize=18, ha='center', x=0.80, y=0.75)
            plt.xlabel('Long (°)', fontsize=14, labelpad=25)
            plt.ylabel('Lat (°)', fontsize=14, labelpad=60)

            m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='f',projection='merc',llcrnrlon= llong, llcrnrlat= llat, urcrnrlon= hlong, urcrnrlat= hlat)

            x,y = m(lon, lat)

            m.drawcoastlines()
            m.drawparallels(np.arange(llat, hlat,abs(hlat-llat)/7), linewidth=0, labels=[1,0,0,0], color='r',zorder=0, fmt="%.1f", fontsize=14)
            m.drawmeridians(np.arange(llong, hlong,abs(hlong-llong)/5), linewidth=0, labels=[0,0,0,1], color='r',zorder=0, fmt="%.1f", fontsize=14)

            m.contourf(x,y, np.squeeze(umidade), alpha=0.4, cmap=colormap['q'], vmin=8, vmax=20)
            m.pcolormesh(x,y, np.squeeze(umidade), alpha=0.4, cmap=colormap['q'], vmin=8, vmax=20)

            cb = plt.colorbar(shrink=0.5, pad=0.04)
            cb.ax.tick_params(labelsize=10)
            plt.savefig(out_path+name+var+".png",bbox_inches='tight')
            plt.close()

        elif var == 'wind':
            u = None; v = None; cont = 0
            for file in files_path:
                dataset = netCDF4.Dataset(file)
    
                times_array = dataset.variables['Times'][:]

                for i,time in enumerate(times_array):
                    currentTime = b''.join(time.tolist()).decode('UTF-8')
                    current_date = datetime.strptime(currentTime, '%Y-%m-%d_%H:%M:%S')
                    cd = current_date.replace(tzinfo=timezone.utc).astimezone(tz=None)
                    if cd.hour == hour:
                        var1 = dataset.variables['U10'][:,:,:].squeeze()
                        var2 = dataset.variables['V10'][:,:,:].squeeze()
                        ut = var1[i:i+1,:,:].squeeze(); vt = var2[i:i+1,:,:].squeeze()
                        if u is None:
                            u = ut; v = vt
                        else:
                            u += ut; v += vt
                        cont+=1

            u = u/cont; v = v/cont
            varmin, varmax = getLowHighWind(u,v)

            plt.figure(figsize=(8,6))
            #plt.title('Velocidade do Vento (10 m) ', fontsize=12)
            plt.xlabel('Long (°)', fontsize=14, labelpad=25)
            plt.ylabel('Lat (°)', fontsize=14, labelpad=60)

            m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='f',projection='merc',llcrnrlon= llong+0.02, llcrnrlat= llat+0.02, urcrnrlon= hlong-0.02, urcrnrlat= hlat-0.02)
            x,y = m(lon, lat)

            yy = np.arange(0, y.shape[0], 3)
            xx = np.arange(0, x.shape[1], 3)

            speed = np.sqrt(u*u + v*v)
            points = np.meshgrid(yy, xx)

            m.contourf(x,y,np.squeeze(speed), alpha=0.4, cmap=colormap['wind'], vmin=0, vmax=6)
            cs = m.pcolor(x,y,np.squeeze(speed), alpha=0.4, cmap=colormap['wind'], vmin=0, vmax=6)

            widths = np.linspace(0, 2, xx.size)
            nx = x[points];ny = y[points]; nu = u[points];nv = v[points]
            
            if 'd01' in name:
                Q = m.quiver(nx[::2,::2],ny[::2,::2] ,nu[::2,::2] ,nv[::2,::2] , scale_units='xy', width=0.0035,scale=0.00004)
                qk = plt.quiverkey(Q,1.095, 0.78, 0.00019,label= r'$\frac{m}{s}$', fontproperties={'size': 18}, labelpos='N')
            elif 'd02' in name:
                Q = m.quiver(nx[::2,::2],ny[::2,::2] ,nu[::2,::2] ,nv[::2,::2] , scale_units='xy', width=0.0035,scale=0.00007)
                qk = plt.quiverkey(Q,1.095, 0.78, 0.00039,label= r'$\frac{m}{s}$', fontproperties={'size': 18}, labelpos='N')
            elif 'd03' in name:
                Q = m.quiver(nx[::2,::2],ny[::2,::2] ,nu[::2,::2] ,nv[::2,::2] , scale_units='xy', width=0.0035,scale=0.0003)
                qk = plt.quiverkey(Q,1.095, 0.78, 0.0019,label= r'$\frac{m}{s}$', fontproperties={'size': 18}, labelpos='N')
            else:
                Q = m.quiver(nx[::2,::2],ny[::2,::2] ,nu[::2,::2] ,nv[::2,::2] , scale_units='xy', width=0.0035,scale=0.001)
                qk = plt.quiverkey(Q,1.095, 0.78, 0.0019,label= r'$\frac{m}{s}$', fontproperties={'size': 18}, labelpos='N')

            m.drawcoastlines(color='0.15')

            m.drawparallels(np.arange(llat, hlat,abs(hlat-llat)/7), linewidth=0, labels=[1,0,0,0], color='r',zorder=0, fmt="%.1f", fontsize=14)
            m.drawmeridians(np.arange(llong, hlong,abs(hlong-llong)/5), linewidth=0, labels=[0,0,0,1], color='r',zorder=0, fmt="%.1f", fontsize=14)

            cb = plt.colorbar(shrink=0.5, pad=0.04)
            cb.ax.tick_params(labelsize=10)

            plt.savefig(out_path+name+var+".png",bbox_inches='tight')
            plt.close()

def drawBreeze(file, out_path, name, qtd):
    colormap = {'T':'jet','wind':'PuBu','q':'jet_r','pressure':'Blues','precip':'jet','hfx':'jet','lh':'jet','sw_dw':'jet'}
    #variables = T,q,wind
    
    data = netCDF4.Dataset(file)
    xlat, xlong = data.variables['XLAT'][:,:,:], data.variables['XLONG'][:,:,:]
    lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
    hlat, llat = np.amax(xlat), np.amin(xlat)
    hlong, llong = np.amax(xlong), np.amin(xlong)
    
    u = None; v = None; cont = 0
    dataset = data

    times_array = dataset.variables['Times'][:]

    #1 semana = 168h
    for i,time in enumerate(times_array):
        if cont  == qtd:
            break

        currentTime = b''.join(time.tolist()).decode('UTF-8')
        current_date = datetime.strptime(currentTime, '%Y-%m-%d_%H:%M:%S')
        cd = current_date.replace(tzinfo=timezone.utc).astimezone(tz=None)
        var1 = dataset.variables['U10'][:,:,:].squeeze()
        var2 = dataset.variables['V10'][:,:,:].squeeze()
        u = var1[i:i+1,:,:].squeeze(); v = var2[i:i+1,:,:].squeeze()

        varmin, varmax = getLowHighWind(u,v)

        plt.figure(figsize=(8,6))
        plt.title(cd.strftime('%d/%m/%Y %H')+'h', fontsize=12)
        plt.xlabel('Long (°)', fontsize=14, labelpad=25)
        plt.ylabel('Lat (°)', fontsize=14, labelpad=60)

        m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='f',projection='merc',llcrnrlon= llong+0.02, llcrnrlat= llat+0.02, urcrnrlon= hlong-0.02, urcrnrlat= hlat-0.02)
        x,y = m(lon, lat)

        yy = np.arange(0, y.shape[0], 3)
        xx = np.arange(0, x.shape[1], 3)

        speed = np.sqrt(u*u + v*v)
        points = np.meshgrid(yy, xx)

        m.contourf(x,y,np.squeeze(speed), alpha=0.4, cmap=colormap['wind'], vmin=0, vmax=6)
        cs = m.pcolor(x,y,np.squeeze(speed), alpha=0.4, cmap=colormap['wind'], vmin=0, vmax=6)

        widths = np.linspace(0, 2, xx.size)
        nx = x[points];ny = y[points]; nu = u[points];nv = v[points]
        
        if 'd01' in name:
            Q = m.quiver(nx[::2,::2],ny[::2,::2] ,nu[::2,::2] ,nv[::2,::2] , scale_units='xy', width=0.0035,scale=0.00004)
            qk = plt.quiverkey(Q,1.095, 0.78, 0.00019,label= r'$\frac{m}{s}$', fontproperties={'size': 18}, labelpos='N')
        elif 'd02' in name:
            Q = m.quiver(nx[::2,::2],ny[::2,::2] ,nu[::2,::2] ,nv[::2,::2] , scale_units='xy', width=0.0035,scale=0.00007)
            qk = plt.quiverkey(Q,1.095, 0.78, 0.00039,label= r'$\frac{m}{s}$', fontproperties={'size': 18}, labelpos='N')
        elif 'd03' in name:
            Q = m.quiver(nx[::2,::2],ny[::2,::2] ,nu[::2,::2] ,nv[::2,::2] , scale_units='xy', width=0.0035,scale=0.0003)
            qk = plt.quiverkey(Q,1.095, 0.78, 0.0019,label= r'$\frac{m}{s}$', fontproperties={'size': 18}, labelpos='N')
        else:
            Q = m.quiver(nx[::2,::2],ny[::2,::2] ,nu[::2,::2] ,nv[::2,::2] , scale_units='xy', width=0.0035,scale=0.001)
            qk = plt.quiverkey(Q,1.095, 0.78, 0.0019,label= r'$\frac{m}{s}$', fontproperties={'size': 18}, labelpos='N')

        m.drawcoastlines(color='0.15')

        m.drawparallels(np.arange(llat, hlat,abs(hlat-llat)/7), linewidth=0, labels=[1,0,0,0], color='r',zorder=0, fmt="%.1f", fontsize=14)
        m.drawmeridians(np.arange(llong, hlong,abs(hlong-llong)/5), linewidth=0, labels=[0,0,0,1], color='r',zorder=0, fmt="%.1f", fontsize=14)

        cb = plt.colorbar(shrink=0.5, pad=0.04)
        cb.ax.tick_params(labelsize=10)

        plt.savefig(out_path+name+"-"+str(cont)+".png",bbox_inches='tight')
        plt.close()
        cont+=1
