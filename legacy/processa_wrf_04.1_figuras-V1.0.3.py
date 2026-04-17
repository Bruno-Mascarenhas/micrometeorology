# -*- Coding: UTF-8 -*-
import glob
import os
import sys
from datetime import UTC, datetime
from itertools import product
from multiprocessing import Pool
from warnings import filterwarnings

import imageio
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import moviepy.editor as mp
import netCDF4
import numpy as np
from mpl_toolkits.basemap import Basemap
from wrf import interplevel

#import pdb
filterwarnings('ignore')
#print (np.__version__)

#funçao que define o numero das imagens geradas como sendo de 3 algarismos
def toNum(arg):
    arg = str(arg)
    if len(arg) == 2:
        arg = '0'+arg
    else:
        arg = '00'+arg
    return arg

#funçao que retorna o menor e maior valor de um variavel especifica
def getLowHigh(variable):
    varflat = variable[1:,:].flatten()
    varlow, varhigh = np.amin(varflat), np.amax(varflat)
    return varlow, varhigh

#funcao que calcula a velocidade minima e maxima do vento
def getLowHighWind(variable1, variable2):
    varflat1, varflat2 = variable1[1:,:].flatten(), variable2[1:,:].flatten()
    speed = np.sqrt(varflat1*varflat1 + varflat2*varflat2)   #speed² = U² + V²
    varlow , varhigh = np.amin(speed), np.amax(speed)
    return varlow, varhigh

#funcao que calcula a velocidade minima e maxima da chuva
def getLowHighRain(variable):
    for i in range(2,variable.shape[0]):
       variable[i,:,:] = variable[i,:,:] - variable[i-1,:,:]
    varflat = variable[1:,:].flatten()
    varlow, varhigh = np.amin(varflat), np.amax(varflat)
    return varlow, varhigh

def cmap_saturado(cmap, saturation_factor):
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.linspace(0, 1, cmap.N))
    hsv_colors = mcolors.rgb_to_hsv(colors[:, :3])  # Converte para HSV
    hsv_colors[:, 1] *= saturation_factor           # Ajusta a saturação
    hsv_colors[:, 1] = np.clip(hsv_colors[:, 1], 0, 1)  # Garante que esteja no intervalo [0, 1]
    adjusted_colors = mcolors.hsv_to_rgb(hsv_colors)  # Converte de volta para RGB
    return mcolors.ListedColormap(adjusted_colors)


#funcao que desenha os mapas usando como argumento o tipo do mapa e o seu dataset correspondente
def drawmap(tipo,dataset):
    colormap = {'temperature':'hot_r','wind':'PuBu','vapor':'YlGnBu','pressure':'Blues','rain':'afmhot_r','HFX':'jet','LH':'jet','SWDOWN':'hot_r','weibull':'jet',f'poteolico{tipo[9:]}':'Blues'} #tipo de colormap usado em cada grafico
    #caminho de onde se localiza o python (diretorio de trabalho)
    dir_scriptpy = sys.argv[1]+'/'
    #caminho de output das figuras
    WRFoutput = sys.argv[2]+'/'
    grade = dataset[dataset.find('d0'):dataset.find('d0')+3]; grade = grade.upper()
    dataset = netCDF4.Dataset(dataset)
    times_array = dataset.variables['Times'][:]
    dates = []; datesStr = []; names = []

    map = {'HFX':'HFX','LH':'LH','pressure':'PRES','rain':'RAIN','SWDOWN':'SWDOWN','temperature':'TEMP','vapor':'VAPOR','wind':'WIND','weibull':'K_WEIB',f'poteolico{tipo[9:]}':f'POT_EOLICO_{tipo[9:]}M'}
    week = {1:'Segunda',2:'Terça',3:'Quarta',4:'Quinta',5:'Sexta',6:'Sábado',7:'Domingo'}
    #desc = {'HFX':'Calor Sensível','pressure':'Pressão Atmosférica (Nível do Mar)','rain':'Precipitação','SWDOWN':'Radiação Global','temperature':'Temperatura (2 m)','vapor':'Umidade Específica','wind':'Velocidade do Vento (10 m)'}

    month = str(datetime.now().month); day = str(datetime.now().day)
    day = sys.argv[3]
    month = sys.argv[4]
    year = sys.argv[5]
    #if len(month) == 1:
    #    month = '0'+str(month)
    #if len(day) == 1:
    #    day = '0'+str(day)
    #today = str(datetime.now().year)+month+day
    today = str(year+month+day)
    print(f'    + PYTHON - data do arquivo: {today}')

    #data do inicio da analise + data e horario da previsao
    it = 0
    for time in times_array:
        currentTime = b''.join(time.tolist()).decode('UTF-8')
        current_date = datetime.strptime(currentTime, '%Y-%m-%d_%H:%M:%S')
        if it == 0:
            start = current_date.strftime("%d/%m/%Y %H") + " (UTC)\n"   #data de inicio da analise
            cd = current_date.replace(tzinfo=UTC).astimezone(tz=None)   #data e horario da previsao
            datesStr.append("\nInício Análise: "+start+"Previsão: "+cd.strftime("%d/%m/%Y %H")+"HL ("+week[cd.isoweekday()]+")")
            names.append(grade+"_"+map[tipo]+"_"+toNum(it))
            it+=1
            continue
        cd = current_date.replace(tzinfo=UTC).astimezone(tz=None)
        #graficos de radiacao solar so sao gerados entre 6h e 18h
        if tipo == 'SWDOWN' and (cd.hour < 6 or cd.hour > 18):
            datesStr.append("\nInício Análise: "+start+"Previsão: "+cd.strftime("%d/%m/%Y %H")+"HL ("+week[cd.isoweekday()]+")")
            names.append(grade+"_"+map[tipo]+"_"+toNum(it))
            it+=1
            continue
        datesStr.append("\nInício Análise: "+start+"Previsão: "+cd.strftime("%d/%m/%Y %H")+"HL ("+week[cd.isoweekday()]+")")
        names.append(grade+"_"+map[tipo]+"_"+toNum(it))
        dates.append((it,cd))
        it+=1

    try:
        #grafico da temperatura com as linhas de pressao
        if tipo == 'temperature':
            xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
            lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
            hlat, llat = np.amax(xlat), np.amin(xlat)   #valor maximo e minimo da latitude (eixo y do grafico)
            hlong, llong = np.amax(xlong), np.amin(xlong)   #valor maximo e minimo da longitude (eixo x do grafico)
            var = dataset.variables['T2'][:,:,:].squeeze()   #variavel = temperatura
            varmin, varmax = getLowHigh(var)   #valor minimo e maximo da temperatura
            varmax = varmax - 273;        varmin = varmin - 273   #conversao de K para C°
            var2 = dataset.variables['PSFC'][:,:,:].squeeze()   #variavel = pressao
            var2 /= 100

            for i,date in dates:
                if i == it-1:
                    break

                celcius = var[i:i+1,:,:] - 273.15   #conversao da temperatura em K para C°
                pressure = var2[i:i+1,:,:]

                #plotagem do grafico
                fig, ax = plt.subplots(figsize=(8,6))
                plt.title('Temperatura do Ar em 2m / Pressão Atm. Superfície ' + datesStr[i], fontsize=12)
                plt.suptitle(r"$^\circ\mathcal{C}$", fontsize=14, ha='center', x=0.79, y=0.75)
                plt.xlabel('Longitude', fontsize=11, labelpad=25)
                plt.ylabel('Latitude', fontsize=11, labelpad=60)

                #cria o objeto basemap (mapa)
                m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='f', projection='merc',llcrnrlon= llong, llcrnrlat= llat,
                                                                                                urcrnrlon= hlong, urcrnrlat= hlat)

                x,y = m(lon, lat)

                #desenha o contorno da costa
                if 'D03' in grade:
                    m.drawcoastlines(linewidth=2)
                elif 'D04' in grade:
                    m.drawcoastlines(linewidth=3)
                else:
                    m.drawcoastlines(linewidth=1)

                #desenha as divisoes dos estados
                if 'D03' in grade:
                    m.drawstates(linewidth=2)
                else:
                    m.drawstates(linewidth=1)

                #desenha as divisoes dos municipios
                if 'D03' in grade or 'D04' in grade:
                    m.readshapefile(dir_scriptpy + 'shapes_BR_cities/BRMUE250GC_SIR',name='mapa')

                #desenha os paralelos e meridianos
                m.drawparallels(np.arange(llat, hlat,abs(hlat-llat)/10), linewidth=0.001, labels=[1,0,0,0], color='r',zorder=0, fmt="%.2f")
                m.drawmeridians(np.arange(llong, hlong,abs(hlong-llong)/5), linewidth=0.001, labels=[0,0,0,1], color='r',zorder=0, fmt="%.2f")

                #cria o mapa de cor da temperatura
                m.contourf(x,y, np.squeeze(celcius), alpha=0.4, cmap=cmap_saturado(colormap[tipo],2), vmin=varmin, vmax=varmax)
                m.pcolor(x,y, np.squeeze(celcius), alpha=0.4, cmap=cmap_saturado(colormap[tipo],2), vmin=varmin, vmax=varmax)
                cb = plt.colorbar(shrink=0.5, pad=0.04)
                cb.ax.tick_params(labelsize=10)

                #desenha as linhas de pressao
                levels = [880,900,950,1000,1013]
                p = m.contour(x,y, np.squeeze(pressure), levels,linewidths=0.8, colors='black')
                plt.clabel(p,colors='black',fmt='%.0f')

                #salva a imagem
                #plt.savefig(path + names[i] +".png",bbox_inches='tight')
                plt.savefig(WRFoutput + names[i] +".png")
                print('    + PYTHON - ' + WRFoutput + names[i] +".png")
                plt.close()

        #grafico da pressao
        elif tipo == 'pressure':
            xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
            lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
            hlat, llat = np.amax(xlat), np.amin(xlat)   #valor maximo e minimo da latitude (eixo y do grafico)
            hlong, llong = np.amax(xlong), np.amin(xlong)   #valor maximo e minimo da longitude (eixo x do grafico)
            var = dataset.variables['PSFC'][:,:,:].squeeze()   #variavel = pressao
            varmin, varmax = getLowHigh(var)   #valor minimo e maximo da pressao
            varmax = varmax/100;        varmin = varmin/100

            for i,date in dates:
                if i == it-1:
                    break

                mbar = var[i:i+1,:,:]/100

                #plotagem do grafico
                fig, ax = plt.subplots(figsize=(8,6))
                plt.title('Pressão Atmosférica ao Nível do Mar ' + datesStr[i], fontsize=12)
                plt.suptitle("${mBar}$", fontsize=14, ha='center', x=0.82, y=0.75)
                plt.xlabel('Longitude', fontsize=11, labelpad=25)
                plt.ylabel('Latitude', fontsize=11, labelpad=60)

                #cria o objeto basemap (mapa)
                m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='f', projection='merc',llcrnrlon= llong, llcrnrlat= llat,
                                                                                                urcrnrlon= hlong, urcrnrlat= hlat)

                x,y = m(lon, lat)

                #desenha o contorno da costa
                if 'D03' in grade:
                    m.drawcoastlines(linewidth=2)
                elif 'D04' in grade:
                    m.drawcoastlines(linewidth=3)
                else:
                    m.drawcoastlines(linewidth=1)

                #desenha as divisoes dos estados
                if 'D03' in grade:
                    m.drawstates(linewidth=2)
                else:
                    m.drawstates(linewidth=1)

                #desenha as divisoes dos municipios
                if 'D03' in grade or 'D04' in grade:
                    m.readshapefile(dir_scriptpy + 'shapes_BR_cities/BRMUE250GC_SIR',name='mapa')

                #desenha os paralelos e meridianos
                m.drawparallels(np.arange(llat, hlat,abs(hlat-llat)/10), linewidth=0.001, labels=[1,0,0,0], color='r',zorder=0, fmt="%.2f")
                m.drawmeridians(np.arange(llong, hlong,abs(hlong-llong)/5), linewidth=0.001, labels=[0,0,0,1], color='r',zorder=0, fmt="%.2f")

                #cria o mapa de cor da pressao
                m.contourf(x,y, np.squeeze(mbar), alpha=0.4, cmap=cmap_saturado(colormap[tipo],2), vmin=varmin, vmax=varmax)
                m.pcolor(x,y, np.squeeze(mbar), alpha=0.4, cmap=cmap_saturado(colormap[tipo],2), vmin=varmin, vmax=varmax)
                cb = plt.colorbar(shrink=0.5, pad=0.04)
                cb.ax.tick_params(labelsize=10)

                #salva a imagem
                #plt.savefig(path + names[i] +".png",bbox_inches='tight')
                plt.savefig(WRFoutput + names[i] +".png")
                print('    + PYTHON - ' + WRFoutput + names[i] +".png")
                plt.close()

        #grafico da umidade especifica
        elif tipo == 'vapor':
            xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
            lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
            hlat, llat = np.amax(xlat), np.amin(xlat)   #valor maximo e minimo da latitude (eixo y do grafico)
            hlong, llong = np.amax(xlong), np.amin(xlong)   #valor maximo e minimo da longitude (eixo x do grafico)
            var = dataset.variables['Q2'][:,:,:].squeeze()   #variavel = umidade especifica
            varmin, varmax = getLowHigh(var)   #valor minimo e maximo da umidade especifica
            varmax = varmax*1000;        varmin = varmin*1000

            for i,date in dates:
                if i == it-1:
                    break

                gkg = var[i:i+1,:,:]*1000

                #plotagem do grafico
                fig, ax = plt.subplots(figsize=(8,6))
                plt.title('Umidade Específica do Ar em 2m' + datesStr[i], fontsize=12)
                plt.suptitle("$g/kg \frac{m}{s}$", fontsize=14, ha='center', x=0.80, y=0.75)
                plt.xlabel('Longitude', fontsize=11, labelpad=25)
                plt.ylabel('Latitude', fontsize=11, labelpad=60)

                #cria o objeto basemap (mapa)
                m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='f', projection='merc',llcrnrlon= llong, llcrnrlat= llat,
                                                                                                urcrnrlon= hlong, urcrnrlat= hlat)

                x,y = m(lon, lat)

                #desenha o contorno da costa
                if 'D03' in grade:
                    m.drawcoastlines(linewidth=2)
                elif 'D04' in grade:
                    m.drawcoastlines(linewidth=3)
                else:
                    m.drawcoastlines(linewidth=1)

                #desenha as divisoes dos estados
                if 'D03' in grade:
                    m.drawstates(linewidth=2)
                else:
                    m.drawstates(linewidth=1)

                #desenha as divisoes dos municipios
                if 'D03' in grade or 'D04' in grade:
                    m.readshapefile(dir_scriptpy + 'shapes_BR_cities/BRMUE250GC_SIR',name='mapa')

                #desenha os paralelos e meridianos
                m.drawparallels(np.arange(llat, hlat,abs(hlat-llat)/10), linewidth=0.001, labels=[1,0,0,0], color='r',zorder=0, fmt="%.2f")
                m.drawmeridians(np.arange(llong, hlong,abs(hlong-llong)/5), linewidth=0.001, labels=[0,0,0,1], color='r',zorder=0, fmt="%.2f")

                plt.contourf(x,y, np.squeeze(gkg), alpha=0.4, cmap=cmap_saturado(colormap[tipo],2), vmin=varmin, vmax=varmax)
                plt.pcolor(x,y, np.squeeze(gkg), alpha=0.4, cmap=cmap_saturado(colormap[tipo],2), vmin=varmin, vmax=varmax)

                cb = plt.colorbar(shrink=0.5, pad=0.04)
                cb.ax.tick_params(labelsize=10)
                #salva a imagem
                #plt.savefig(path + names[i] +".png",bbox_inches='tight')
                plt.savefig(WRFoutput + names[i] +".png")
                print('    + PYTHON - ' + WRFoutput + names[i] +".png")
                plt.close()

        #grafico da velocidade do vento
        elif tipo == 'wind':
            xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
            lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
            hlat, llat = np.amax(xlat), np.amin(xlat)   #valor maximo e minimo da latitude (eixo y do grafico)
            hlong, llong = np.amax(xlong), np.amin(xlong)   #valor maximo e minimo da longitude (eixo x do grafico)
            u10 = dataset.variables['U10'][:].squeeze(); v10 = dataset.variables['V10'][:].squeeze()
            varmin, varmax = getLowHighWind(u10,v10)

            for i,date in dates:
                if i == it-1:
                    break

                u = u10[i:i+1,:,:].squeeze()
                v = v10[i:i+1,:,:].squeeze()

                #plotagem do grafico
                fig, ax = plt.subplots(figsize=(8,6))
                plt.title('Velocidade do Vento em 10m ' + datesStr[i], fontsize=12)
                plt.suptitle(r'$\frac{m}{s}$', fontsize=17, ha='center', x=0.79, y=0.75)
                plt.xlabel('Longitude', fontsize=11, labelpad=25)
                plt.ylabel('Latitude', fontsize=11, labelpad=60)

                #cria o objeto basemap (mapa)
                m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='f', projection='merc',llcrnrlon= llong, llcrnrlat= llat,urcrnrlon= hlong, urcrnrlat= hlat,ax=ax)

                x,y = m(lon, lat)
                yy = np.arange(0, y.shape[0], 3)
                xx = np.arange(0, x.shape[1], 3)

                speed = np.sqrt(u*u + v*v)
                points = np.meshgrid(yy, xx)

                #mapa de cor da velocidade do vento
                plt.contourf(x, y, np.sqrt(u*u + v*v), alpha = 0.4,cmap = 'Blues',vmin=varmin,vmax=varmax)
                cs = plt.pcolor(x,y,np.squeeze(speed), alpha=0.4, cmap=cmap_saturado(colormap[tipo],2), vmin=varmin, vmax=varmax)
                cb = plt.colorbar(shrink=0.5, pad=0.04)
                cb.ax.tick_params(labelsize=10)

                widths = np.linspace(0, 2, xx.size)

                points = tuple(np.meshgrid(yy, xx))
                nx = x[points];            ny = y[points];            nu = u[points];            nv = v[points]
                #x[points], y[points], u[points], v[points]

                #mapa vetorial (setas) indicando direcao e velocidade do vento
                if 'D01' in grade:
                    Q = m.quiver(nx[::2,::2],ny[::2,::2] ,nu[::2,::2] ,nv[::2,::2] , scale_units='xy', width=0.0035,scale=0.0001)
                    qk = plt.quiverkey(Q,1.095, 1, 10,label= r'10 $\frac{m}{s}$', fontproperties={'size': 11}, labelpos='E')

                #no caso dos graficos do dataset D02 ha mais vetores de vento
                elif 'D02' in grade:
                    Q = m.quiver(nx[::,::],ny[::,::] ,nu[::,::] ,nv[::,::] , scale_units='xy', width=0.0035,scale=0.0003)
                    qk = plt.quiverkey(Q,1.095, 1, 10,label= r'10 $\frac{m}{s}$', fontproperties={'size': 11}, labelpos='E')

                elif 'D03' in grade:
                    Q = m.quiver(nx[::2,::2],ny[::2,::2] ,nu[::2,::2] ,nv[::2,::2] , scale_units='xy', width=0.0035,scale=0.001)
                    qk = plt.quiverkey(Q,1.095, 1, 8,label= r'8 $\frac{m}{s}$', fontproperties={'size': 11}, labelpos='E')

                else:
                    Q = m.quiver(nx[::2,::2],ny[::2,::2] ,nu[::2,::2] ,nv[::2,::2] , scale_units='xy', width=0.0035,scale=0.001)
                    qk = plt.quiverkey(Q,1.095, 1, 5,label= r'5 $\frac{m}{s}$', fontproperties={'size': 11}, labelpos='E')

                #desenha o contorno da costa
                if 'D03' in grade:
                    m.drawcoastlines(linewidth=2)
                elif 'D04' in grade:
                    m.drawcoastlines(linewidth=3)
                else:
                    m.drawcoastlines(linewidth=1)

                #desenha as divisoes dos estados
                if 'D03' in grade:
                    m.drawstates(linewidth=2)
                else:
                    m.drawstates(linewidth=1)

                #desenha as divisoes dos municipios
                if 'D03' in grade or 'D04' in grade:
                    m.readshapefile(dir_scriptpy + 'shapes_BR_cities/BRMUE250GC_SIR',name='mapa')

                #desenha os paralelos e meridianos
                m.drawparallels(np.arange(llat, hlat,abs(hlat-llat)/10), linewidth=0.001, labels=[1,0,0,0], color='r',zorder=0, fmt="%.2f")
                m.drawmeridians(np.arange(llong, hlong,abs(hlong-llong)/5), linewidth=0.001, labels=[0,0,0,1], color='r',zorder=0, fmt="%.2f")

                ##################################
                #nx, ny = m(-38.80,-12.5)
                #m.scatter(nx,ny,marker='D',color='m')
                ##################################

                #salva a imagem
                #plt.savefig(path + names[i] +".png",bbox_inches='tight')
                plt.savefig(WRFoutput + names[i] +".png")
                print('    + PYTHON - ' + WRFoutput + names[i] +".png")
                plt.close()

        #grafico da precipitacao acumulada
        elif tipo == 'rain':
            xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
            lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
            hlat, llat = np.amax(xlat), np.amin(xlat)   #valor maximo e minimo da latitude (eixo y do grafico)
            hlong, llong = np.amax(xlong), np.amin(xlong)   #valor maximo e minimo da longitude (eixo x do grafico)
            var = dataset.variables['RAINC'][:,:,:].squeeze()
            var2 = dataset.variables['RAINNC'][:,:,:].squeeze()
            var = var + var2
            varmin, varmax = getLowHighRain(var)

            for i,date in dates:
                if i == it-1:
                    break

                # mm = var[i:i+1,:,:]

                if i == 1:
                    mm = var[i:i+1,:,:]
                else:
                    mm = var[i:i+1,:,:]-var[i-1:i,:,:]

                #plotagem do grafico
                fig, ax = plt.subplots(figsize=(8,6))
                plt.title('Precipitação Acumulada em 1h' + datesStr[i], fontsize=12)
                plt.suptitle("$mm$", fontsize=14, ha='center', x=0.79, y=0.75)
                plt.xlabel('Longitude', fontsize=11, labelpad=25)
                plt.ylabel('Latitude', fontsize=11, labelpad=60)

                #cria o objeto basemap (mapa)
                m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='f', projection='merc',llcrnrlon= llong, llcrnrlat= llat,
                                                                                                urcrnrlon= hlong, urcrnrlat= hlat)

                x,y = m(lon, lat)

                #desenha o contorno da costa
                if 'D03' in grade:
                    m.drawcoastlines(linewidth=2)
                elif 'D04' in grade:
                    m.drawcoastlines(linewidth=3)
                else:
                    m.drawcoastlines(linewidth=1)

                #desenha as divisoes dos estados
                if 'D03' in grade:
                    m.drawstates(linewidth=2)
                else:
                    m.drawstates(linewidth=1)

                #desenha as divisoes dos municipios
                if 'D03' in grade or 'D04' in grade:
                    m.readshapefile(dir_scriptpy + 'shapes_BR_cities/BRMUE250GC_SIR',name='mapa')

                #desenha os paralelos e meridianos
                m.drawparallels(np.arange(llat, hlat,abs(hlat-llat)/10), linewidth=0.001, labels=[1,0,0,0], color='r',zorder=0, fmt="%.2f")
                m.drawmeridians(np.arange(llong, hlong,abs(hlong-llong)/5), linewidth=0.001, labels=[0,0,0,1], color='r',zorder=0, fmt="%.2f")

                #mapa de cor da precipitacao
                m.contourf(x,y, np.squeeze(mm), alpha = 0.4, cmap = cmap_saturado(colormap[tipo],2), vmin=varmin, vmax=varmax)
                m.pcolor(x,y, np.squeeze(mm), alpha = 0.4, cmap = cmap_saturado(colormap[tipo],2), vmin=varmin, vmax=varmax)
                cb = plt.colorbar(shrink=0.5, pad=0.04)
                cb.ax.tick_params(labelsize=10)

                #salva a imagem
                #plt.savefig(path + names[i] +".png",bbox_inches='tight')
                plt.savefig(WRFoutput + names[i] +".png")
                print('    + PYTHON - ' + WRFoutput + names[i] +".png")   #breno esteve aqui
                plt.close()

        #grafico do calor sensivel
        elif tipo == 'HFX':
            xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
            lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
            hlat, llat = np.amax(xlat), np.amin(xlat)   #valor maximo e minimo da latitude (eixo y do grafico)
            hlong, llong = np.amax(xlong), np.amin(xlong)   #valor maximo e minimo da longitude (eixo x do grafico)
            var = dataset.variables['HFX'][:,:,:].squeeze()
            varmin, varmax = getLowHigh(var)   #valor minimo e maximo da

            for i,date in dates:
                if i == it-1:
                    break

                mm = var[i:i+1,:,:]

                #plotagem do grafico
                fig, ax = plt.subplots(figsize=(8,6))
                plt.title('Calor Sensível em Superfície ' + datesStr[i], fontsize=12)
                plt.suptitle("$ W m ^ {-2} $", fontsize=14, ha='center', x=0.82, y=0.75)
                plt.xlabel('Longitude', fontsize=11, labelpad=25)
                plt.ylabel('Latitude', fontsize=11, labelpad=60)

                #cria o objeto basemap (mapa)
                m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='f', projection='merc',llcrnrlon= llong, llcrnrlat= llat,
                                                                                                urcrnrlon= hlong, urcrnrlat= hlat)

                x,y = m(lon, lat)

                #desenha o contorno da costa
                if 'D03' in grade:
                    m.drawcoastlines(linewidth=2)
                elif 'D04' in grade:
                    m.drawcoastlines(linewidth=3)
                else:
                    m.drawcoastlines(linewidth=1)

                #desenha as divisoes dos estados
                if 'D03' in grade:
                    m.drawstates(linewidth=2)
                else:
                    m.drawstates(linewidth=1)

                #desenha as divisoes dos municipios
                if 'D03' in grade or 'D04' in grade:
                    m.readshapefile(dir_scriptpy + 'shapes_BR_cities/BRMUE250GC_SIR',name='mapa')

                #desenha os paralelos e meridianos
                m.drawparallels(np.arange(llat, hlat,abs(hlat-llat)/10), linewidth=0.001, labels=[1,0,0,0], color='r',zorder=0, fmt="%.2f")
                m.drawmeridians(np.arange(llong, hlong,abs(hlong-llong)/5), linewidth=0.001, labels=[0,0,0,1], color='r',zorder=0, fmt="%.2f")

                #mapa de cor do calor sensivel
                m.contourf(x,y, np.squeeze(mm), alpha = 0.4, cmap = cmap_saturado(colormap[tipo],2), vmin=varmin, vmax=varmax)
                m.pcolor(x,y, np.squeeze(mm), alpha = 0.4, cmap = cmap_saturado(colormap[tipo],2), vmin=varmin, vmax=varmax)
                cb = plt.colorbar(shrink=0.5, pad=0.04)
                cb.ax.tick_params(labelsize=10)

                #salva a imagem
                #plt.savefig(path + names[i] +".png",bbox_inches='tight')]
                plt.savefig(WRFoutput + names[i] +".png")
                print('    + PYTHON - ' + WRFoutput + names[i] +".png")
                plt.close()

        #grafico do calor latente
        elif tipo == 'LH':
            xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
            lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
            hlat, llat = np.amax(xlat), np.amin(xlat)   #valor maximo e minimo da latitude (eixo y do grafico)
            hlong, llong = np.amax(xlong), np.amin(xlong)   #valor maximo e minimo da longitude (eixo x do grafico)
            var = dataset.variables['LH'][:,:,:].squeeze()
            varmin, varmax = getLowHigh(var)   #valor minimo e maximo da

            for i,date in dates:
                if i == it-1:
                    break

                mm = var[i:i+1,:,:]

                #plotagem do grafico
                fig, ax = plt.subplots(figsize=(8,6))
                plt.title('Calor Latente em Superfície' + datesStr[i], fontsize=12)
                plt.suptitle("$ W m ^ {-2} $", fontsize=14, ha='center', x=0.82, y=0.75)
                plt.xlabel('Longitude', fontsize=11, labelpad=25)
                plt.ylabel('Latitude', fontsize=11, labelpad=60)

                #cria o objeto basemap (mapa)
                m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='f', projection='merc',llcrnrlon= llong, llcrnrlat= llat,
                                                                                                urcrnrlon= hlong, urcrnrlat= hlat)

                x,y = m(lon, lat)

                #desenha o contorno da costa
                if 'D03' in grade or 'DO4' in grade:
                    m.drawcoastlines(linewidth=2)
                else:
                    m.drawcoastlines(linewidth=1)

                #desenha as divisoes dos estados
                if 'D03' in grade:
                    m.drawstates(linewidth=2)
                else:
                    m.drawstates(linewidth=1)

                #desenha os paralelos e meridianos
                m.drawparallels(np.arange(llat, hlat,abs(hlat-llat)/10), linewidth=0.001, labels=[1,0,0,0], color='r',zorder=0, fmt="%.2f")
                m.drawmeridians(np.arange(llong, hlong,abs(hlong-llong)/5), linewidth=0.001, labels=[0,0,0,1], color='r',zorder=0, fmt="%.2f")

                #desenha as divisoes dos municipios
                if 'D03' in grade or 'D04' in grade:
                    m.readshapefile(dir_scriptpy + 'shapes_BR_cities/BRMUE250GC_SIR',name='mapa')

                #mapa de cor do calor latente
                m.contourf(x,y, np.squeeze(mm), alpha = 0.4, cmap = cmap_saturado(colormap[tipo],2), vmin=varmin, vmax=varmax)
                m.pcolor(x,y, np.squeeze(mm), alpha = 0.4, cmap = cmap_saturado(colormap[tipo],2), vmin=varmin, vmax=varmax)
                cb = plt.colorbar(shrink=0.5, pad=0.04)
                cb.ax.tick_params(labelsize=10)

                #salva a imagem
                #plt.savefig(path + names[i] +".png",bbox_inches='tight')
                plt.savefig(WRFoutput + names[i] +".png")
                print('    + PYTHON - ' + WRFoutput + names[i] +".png")

                plt.close()

        #grafico da radiacao solar
        elif tipo == 'SWDOWN':
            xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
            lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
            hlat, llat = np.amax(xlat), np.amin(xlat)   #valor maximo e minimo da latitude (eixo y do grafico)
            hlong, llong = np.amax(xlong), np.amin(xlong)   #valor maximo e minimo da longitude (eixo x do grafico)
            var = dataset.variables['SWDOWN'][:,:,:].squeeze()
            varmin, varmax = getLowHigh(var)   #valor minimo e maximo da

            for i,date in dates:
                if i == it-1:
                    break

                mm = var[i:i+1,:,:]

                #plotagem do grafico
                fig, ax = plt.subplots(figsize=(8,6))
                plt.title('Radiação Solar Global ' + datesStr[i], fontsize=12)
                plt.suptitle(r'$\frac{W}{m²}$', fontsize=14, ha='center', x=0.82, y=0.75)
                plt.xlabel('Longitude', fontsize=11, labelpad=25)
                plt.ylabel('Latitude', fontsize=11, labelpad=60)

                #cria o objeto basemap (mapa)
                m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='f', projection='merc',llcrnrlon= llong, llcrnrlat= llat,
                                                                                                urcrnrlon= hlong, urcrnrlat= hlat)

                x,y = m(lon, lat)

                #desenha o contorno da costa
                if 'D03' in grade:
                    m.drawcoastlines(linewidth=2)
                elif 'D04' in grade:
                    m.drawcoastlines(linewidth=3)
                else:
                    m.drawcoastlines(linewidth=1)

                #desenha as divisoes dos estados
                if 'D03' in grade:
                    m.drawstates(linewidth=2)
                else:
                    m.drawstates(linewidth=1)

                #desenha as divisoes dos municipios
                if 'D03' in grade or 'D04' in grade:
                    m.readshapefile(dir_scriptpy + 'shapes_BR_cities/BRMUE250GC_SIR',name='mapa')

                #desenha os paralelos e meridianos
                m.drawparallels(np.arange(llat, hlat,abs(hlat-llat)/10), linewidth=0.001, labels=[1,0,0,0], color='r',zorder=0, fmt="%.2f")
                m.drawmeridians(np.arange(llong, hlong,abs(hlong-llong)/5), linewidth=0.001, labels=[0,0,0,1], color='r',zorder=0, fmt="%.2f")

                #mapa de cor da radiacao solar
                m.contourf(x,y, np.squeeze(mm), alpha = 0.4, cmap = cmap_saturado(colormap[tipo],2), vmin=varmin, vmax=varmax)
                m.pcolor(x,y, np.squeeze(mm), alpha = 0.4, cmap = cmap_saturado(colormap[tipo],2), vmin=varmin, vmax=varmax)
                cb = plt.colorbar(shrink=0.5, pad=0.04)
                cb.ax.tick_params(labelsize=10)

                #salva a imagem
                #plt.savefig(path + names[i] +".png",bbox_inches='tight')
                plt.savefig(WRFoutput + names[i] +".png")
                print('    + PYTHON - ' + WRFoutput + names[i] +".png")   #breno esteve aqui

                plt.close()

        #grafico fator k de weibull
        elif tipo == 'weibull':
            xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
            lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
            hlat, llat = np.amax(xlat), np.amin(xlat)   #valor maximo e minimo da latitude (eixo y do grafico)
            hlong, llong = np.amax(xlong), np.amin(xlong)   #valor maximo e minimo da longitude (eixo x do grafico)
            U = dataset.variables['U'][:].squeeze(); V = dataset.variables['V'][:].squeeze()  #velocidades horizontais e verticais do vento na borda das grades
            #interpolacao das velocidades para representarem os valores nos centros das grades
            U_central = (U[:, :, :, :-1] + U[:, :, :, 1:])/2; V_central = (V[:, :, :-1, :] + V[:, :, 1:, :])/2
            speed4d = np.empty_like(U_central) #matriz das velocidades resultantes (4d)
            speed3d = np.empty_like(U_central[:,0,:,:]) #matriz das velocidades resultantes para uma altura especifica (3d)
            fator_k = np.empty_like(U_central[0,0,:,:]) #matriz do fator K de weibull

            ph = dataset.variables['PH'][:].squeeze(); phb = dataset.variables['PHB'][:].squeeze()  #alturas potencias
            hgt = dataset.variables['HGT'][:].squeeze() #altura do terreno

            geopot_total = ph + phb      #geopotencial total (media + variação)
            altura = geopot_total / 9.81        #altura das extremidades dos niveis (em relaçao ao nivel do mar)
            #interpolacao das velocidades para representarem os valores no meio dos niveis
            altura_central = (altura[:, :-1, :, :] + altura[:, 1:, :, :])/2

            #ajustando a altura para considerar o nivel do terreno
            altura_ajustada = np.empty_like(altura_central)
            for i in range(altura_ajustada.shape[0]):
                for j in range(altura_ajustada.shape[1]):
                    altura_ajustada[i,j,:,:] = altura_central[i,j,:,:] - hgt[i,:,:]

            #calculo do mapa de velocidade resultante do vento para cada hora analisada em todas os niveis de altura
            for i in range(speed4d.shape[0]):
                u = U_central[i:i+1,:,:,:].squeeze()
                v = V_central[i:i+1,:,:,:].squeeze()
                speed4d[i,:,:,:] = np.sqrt(u*u + v*v)

            #calculo do mapa de velocidade resultante do vento para cada hora analisada em um nivel de altura especificado
            for i in range(speed3d.shape[0]):
                speed3d[i,:,:] = interplevel(speed4d[i,:,:,:], altura_ajustada[i,:,:,:], 100)

            #corrigindo falhas na interpolaçao
            for i in range(speed3d.shape[0]):
                for k in range(speed3d.shape[1]):
                    for j in range(speed3d.shape[2]):
                        if np.isnan(speed3d[i,k,j]) == True:
                            for a in range(altura_ajustada.shape[1]):
                                if altura_desejada > altura_ajustada[i,a,k,j]:
                                    nivel_inf_altura = a
                            speed3d[i,k,j] = speed4d[i,nivel_inf_altura,k,j] + (altura_desejada-altura_ajustada[i,nivel_inf_altura,k,j])*\
                            (speed4d[i,nivel_inf_altura+1,k,j]-speed4d[i,nivel_inf_altura,k,j])/(altura_ajustada[i,nivel_inf_altura+1,k,j]-altura_ajustada[i,nivel_inf_altura,k,j])

            #calculo do fator k de weibull em cada ponto do mapa para todo o periodo analisado
            for k in range(fator_k.shape[0]):
                for j in range(fator_k.shape[1]):
                    fator_k[k,j] = (np.std(speed3d[1:,k,j])/np.mean(speed3d[1:,k,j]))**-1.086

            #plotagem do grafico
            fig, ax = plt.subplots(figsize=(8,6))
            plt.title('Fator de forma de Weibull' + datesStr[-1], fontsize=12)
            plt.suptitle('k', fontsize=15, x=0.79, y=0.75)
            plt.xlabel('Longitude', fontsize=11, labelpad=25)
            plt.ylabel('Latitude', fontsize=11, labelpad=60)

            #cria o objeto basemap (mapa)
            m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='f', projection='merc',llcrnrlon= llong, llcrnrlat= llat,urcrnrlon= hlong, urcrnrlat= hlat,ax=ax)

            x,y = m(lon, lat)

            #mapa de cor do fator k
            plt.contourf(x, y, fator_k, alpha = 0.4,cmap = cmap_saturado(colormap[tipo],2), vmin = np.min(fator_k), vmax = np.max(fator_k))
            cs = plt.pcolor(x,y,fator_k, alpha=0.4, cmap= cmap_saturado(colormap[tipo],2), vmin = np.min(fator_k), vmax = np.max(fator_k))
            cb = plt.colorbar(shrink=0.5, pad=0.04)
            cb.ax.tick_params(labelsize=10)

            p = m.contour(x,y, fator_k,linewidths=0.8, colors='black')
            plt.clabel(p,colors='black',fmt='%.1f')

            #desenha o contorno da costa
            if 'D03' in grade:
                m.drawcoastlines(linewidth=2)
            elif 'D04' in grade:
                m.drawcoastlines(linewidth=3)
            else:
                m.drawcoastlines(linewidth=1)

            #desenha as divisoes dos estados
            if 'D03' in grade:
                m.drawstates(linewidth=2)
            else:
                m.drawstates(linewidth=1)

            #desenha as divisoes dos municipios
            if 'D03' in grade or 'D04' in grade:
                m.readshapefile(dir_scriptpy + 'shapes_BR_cities/BRMUE250GC_SIR',name='mapa')

            #desenha os paralelos e meridianos
            m.drawparallels(np.arange(llat, hlat,abs(hlat-llat)/10), linewidth=0.001, labels=[1,0,0,0], color='r',zorder=0, fmt="%.2f")
            m.drawmeridians(np.arange(llong, hlong,abs(hlong-llong)/5), linewidth=0.001, labels=[0,0,0,1], color='r',zorder=0, fmt="%.2f")

            ##################################
            #nx, ny = m(-38.80,-12.5)
            #m.scatter(nx,ny,marker='D',color='m')
            ##################################

            #salva a imagem
            #plt.savefig(path + names[i] +".png",bbox_inches='tight')
            plt.savefig(WRFoutput + grade + "_K_WEIBULL.png")
            print('    + PYTHON - ' + WRFoutput + grade + "_K_WEIBULL.png")
            plt.close()

        #grafico potencial eolico
        elif 'poteolico' in tipo:
            xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
            lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
            hlat, llat = np.amax(xlat), np.amin(xlat)   #valor maximo e minimo da latitude (eixo y do grafico)
            hlong, llong = np.amax(xlong), np.amin(xlong)   #valor maximo e minimo da longitude (eixo x do grafico)

            U = dataset.variables['U'][:].squeeze(); V = dataset.variables['V'][:].squeeze()  #velocidades horizontais e verticais do vento na borda das grades
            U_central = (U[:, :, :, :-1] + U[:, :, :, 1:])/2; V_central = (V[:, :, :-1, :] + V[:, :, 1:, :])/2   #interpolacao das velocidades para representarem os valores nos centros das grades

            speed4d = np.empty_like(U_central) #matriz das velocidades resultantes (4d)
            speed3d = np.empty_like(U_central[:,0,:,:]) #matriz das velocidades resultantes para uma altura especifica (3d)
            pot_eolico = np.empty_like(U_central[0,0,:,:]) #matriz das velocidades resultantes medias semanais (potencial eolico)


            ph = dataset.variables['PH'][:].squeeze(); phb = dataset.variables['PHB'][:].squeeze()  #alturas potencias
            hgt = dataset.variables['HGT'][:].squeeze() #altura do terreno
            geopot_total = ph + phb    #geopotencial total (media + variação)
            altura = geopot_total / 9.81      #altura das extremidades dos niveis (em relaçao ao nivel do mar)
            altura_central = (altura[:, :-1, :, :] + altura[:, 1:, :, :])/2    #interpolacao das velocidades para representarem os valores no meio dos niveis


            #ajustando a altura para considerar o nivel do terreno
            altura_ajustada = np.empty_like(altura_central)
            for i in range(altura_ajustada.shape[0]):
                for j in range(altura_ajustada.shape[1]):
                    altura_ajustada[i,j,:,:] = altura_central[i,j,:,:] - hgt[i,:,:]

            #altura media de todos os pontos de cada nivel (referente ao meio do nivel)
            altura_media = [np.mean(altura_ajustada[:,i,:,:]) for i in range(altura_ajustada.shape[1])]
            # print(altura_media,np.size(altura_media))

            altura_desejada = int(tipo[9:]) #altura escolhida no comando de rodar o script

            #valores minimo e maximo da escala das figuras
            for i in range(np.size(altura_media)):
                if altura_desejada > altura_media[i]:
                    nivel_inf_altura = i
            # varmin,_ = getLowHighWind(U_central[:,nivel_inf_altura,:,:],V_central[:,nivel_inf_altura,:,:])
            # _,varmax = getLowHighWind(U_central[:,nivel_inf_altura+1,:,:],V_central[:,nivel_inf_altura+1,:,:])

            speed4d = np.sqrt(U_central**2+V_central**2)
            # calculo do mapa de velocidade resultante do vento para cada hora analisada em um nivel de altura especificado
            for i in range(speed3d.shape[0]):
                speed3d[i,:,:] = interplevel(speed4d[i,:,:,:], altura_ajustada[i,:,:,:], altura_desejada)
            speed3d_corrigida = speed3d[~np.isnan(speed3d)]
            varmin, varmax = np.min(speed3d_corrigida), np.max(speed3d_corrigida)

            for i,date in dates:
                if i == it-1:
                    break

                #calculo do mapa de velocidade resultante do vento para cada hora analisada em todas os niveis de altura
                u = U_central[i:i+1,:,:,:].squeeze()
                v = V_central[i:i+1,:,:,:].squeeze()
                speed3d = np.sqrt(u*u + v*v)

                #calculo do mapa de velocidade resultante do vento para cada hora analisada em um nivel de altura especificado (interpolaçao)
                pot_eolico = interplevel(speed3d, altura_ajustada[i,:,:,:], altura_desejada)

                #corrigindo falhas na interpolaçao
                for k in range(pot_eolico.shape[0]):
                    for j in range(pot_eolico.shape[1]):
                        if np.isnan(pot_eolico[k,j]) == True:
                            for a in range(altura_ajustada.shape[1]):
                                if altura_desejada > altura_ajustada[i,a,k,j]:
                                    nivel_inf_altura = a
                            pot_eolico[k,j] = speed3d[nivel_inf_altura,k,j] + (altura_desejada-altura_ajustada[nivel_inf_altura,k,j])*\
                            (speed3d[nivel_inf_altura+1,k,j]-speed3d[nivel_inf_altura,k,j])/(altura_ajustada[nivel_inf_altura+1,k,j]-altura_ajustada[nivel_inf_altura,k,j])


                #plotagem do grafico
                fig, ax = plt.subplots(figsize=(8,6))
                plt.title(f'Potencial eólico a {altura_desejada}m de altura' + datesStr[i], fontsize=12)
                plt.suptitle(r'$\frac{m}{s}$', fontsize=15, x=0.79, y=0.75)
                plt.xlabel('Longitude', fontsize=11, labelpad=25)
                plt.ylabel('Latitude', fontsize=11, labelpad=60)


                #cria o objeto basemap (mapa)
                m = Basemap(rsphere=(6378137.00,6356752.3142),resolution='f', projection='merc',llcrnrlon= llong, llcrnrlat= llat,urcrnrlon= hlong, urcrnrlat= hlat,ax=ax)

                x,y = m(lon, lat)

                #mapa do potencial eolico
                plt.contourf(x, y, pot_eolico, alpha = 0.4,cmap = cmap_saturado(colormap[tipo],2),vmin=varmin,vmax=varmax)
                cs = plt.pcolor(x, y, pot_eolico, alpha=0.4, cmap = cmap_saturado(colormap[tipo],2),vmin=varmin,vmax=varmax)
                cb = plt.colorbar(shrink=0.5, pad=0.04)
                cb.ax.tick_params(labelsize=10)

                #desenha o contorno da costa
                if 'D03' in grade:
                    m.drawcoastlines(linewidth=2)
                elif 'D04' in grade:
                    m.drawcoastlines(linewidth=3)
                else:
                    m.drawcoastlines(linewidth=1)

                #desenha as divisoes dos estados
                if 'D03' in grade:
                    m.drawstates(linewidth=2)
                else:
                    m.drawstates(linewidth=1)

                #desenha as divisoes dos municipios
                if 'D03' in grade or 'D04' in grade:
                    m.readshapefile(dir_scriptpy + 'shapes_BR_cities/BRMUE250GC_SIR',name='mapa')

                #desenha os paralelos e meridianos
                m.drawparallels(np.arange(llat, hlat,abs(hlat-llat)/10), linewidth=0.001, labels=[1,0,0,0], color='r',zorder=0, fmt="%.2f")
                m.drawmeridians(np.arange(llong, hlong,abs(hlong-llong)/5), linewidth=0.001, labels=[0,0,0,1], color='r',zorder=0, fmt="%.2f")

                ##################################
                #nx, ny = m(-38.80,-12.5)
                #m.scatter(nx,ny,marker='D',color='m')
                ##################################

                #salva a imagem
                #plt.savefig(path + names[i] +".png",bbox_inches='tight')
                plt.savefig(WRFoutput + names[i] +".png")
                print('    + PYTHON - ' + WRFoutput + names[i] +".png")
                plt.close()
    except Exception as err:
        _, _, exc_tb = sys.exc_info()
        print("error line: ", exc_tb.tb_lineno)
        print("figuras err: ", err, "\n tipo: ", tipo, "\n arquivo: ", files)


def generateGifs(name, files, path_output):
    try:
        files = glob.glob(files+'*.png')
        files.sort(key=os.path.getmtime)
        print("peguei o glob")

        images = [imageio.imread(x) for x in files]

        imageio.mimsave(path_output+name+'.gif',images,format='GIF',duration=0.5)
        print("peguei o gif")

        clip = mp.VideoFileClip(path_output+name+'.gif')
        clip.write_videofile(path_output+name+'.webm',audio=False,threads=1)
        print("peguei o video")


        os.remove(path_output+name+'.gif')
    except Exception as err:
        print("gif err: ", err)

if __name__ == '__main__':

    #graphs.png for each hour (define quais tipos (variaveis) de graficos seram gerados)
    #args = ['temperature','pressure','vapor','wind','rain','HFX','LH','SWDOWN']
    args = [arg for arg in sys.argv[13:]] #variaveis de saida
    #caminho de input dos dados wrf
    #pathX = ['/home/murilo/leal/mapas/wrfout_d0X_2024-06-19_00_00_00']


    #sys.argv[1]  sys.argv[2]  sys.argv[3]  sys.argv[4]  sys.argv[5]    sys.argv[6]   sys.argv[7]   sys.argv[8]   sys.argv[9]   sys.argv[10]   sys.argv[11]
    #$dir_script  $WRFoutput   $day $month  $year        $num_min_dom   $num_max_dom  $path1        $path2        $path3        $path4         $path5



    #dir_scriptpy $WRFoutput $day $month $year $FIGini $FIGfim $path1 $path2 $path3 $path4 $path5

    print(' ')
    print('    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐')
    print('    + PYTHON - EXECUTANDO O PYTHON')
    print('    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘')
    print(' ')


    print(f'    + PYTHON - dir_scriptpy = {sys.argv[1]}')
    print(f'    + PYTHON - WRFoutput    = {sys.argv[2]}')
    print(f'    + PYTHON - day          = {sys.argv[3]}')
    print(f'    + PYTHON - month        = {sys.argv[4]}')
    print(f'    + PYTHON - year         = {sys.argv[5]}')
    print(f'    + PYTHON - num_min_dom  = {sys.argv[6]}')
    print(f'    + PYTHON - num_max_dom  = {sys.argv[7]}')
    print(f'    + PYTHON - path1        = {sys.argv[8]}')
    print(f'    + PYTHON - path2        = {sys.argv[9]}')
    print(f'    + PYTHON - path3        = {sys.argv[10]}')
    print(f'    + PYTHON - path4        = {sys.argv[11]}')
    print(f'    + PYTHON - path5        = {sys.argv[12]}')
    [print(f'    + PYTHON - variavel_saida_{i+1}        = {args[i]}') for i in range(np.size(args))]

    #define de quais arquivos wrf serao gerados as imagens
    num_min_dom= int(sys.argv[6]) #inicio - wrf_d+num_min_dom
    num_max_dom= int(sys.argv[7]) #fim - wrf_d+num_max_dom

    #num_min_dom <= num_max_dom
    if num_min_dom > num_max_dom:
        print('    + PYTHON - O primeiro argumento wrf deve ser menor ou igual ao segundo')
        exit()
    #se num_min_dom == num_max_dom, gera os graficos correspondente a apenas um arquivo wrf
    elif num_min_dom == num_max_dom:
        files = [sys.argv[7+num_min_dom]]
        #files = path
        print(f'    + PYTHON -  {files}')
    #gera os graficos correspondentes ao wrf_d+num_min_dom ate wrf_d+num_max_dom
    else:
        files = [sys.argv[7+i] for i in range(num_min_dom,num_max_dom+1)]
        #files = path
        print(f'    + PYTHON -  {files}')

    final = product(args,files)

    processes = Pool()

    processes.starmap(drawmap,final)   #executa a funcao de criacao das imagens

    #make a .webm with the graphs
    WRFoutput  = sys.argv[2]+'/' #caminho de output dos videos

    #possiveis variaveis de saida
    map = {'HFX':'HFX','LH':'LH','pressure':'PRES','rain':'RAIN','SWDOWN':'SWDOWN','temperature':'TEMP','vapor':'VAPOR','wind':'WIND','weibull':'K_WEIB'}
    for i in range(np.size(args)):
        if 'poteolico' in args[i]:
            map[f'poteolico{args[i][9:]}'] = f'POT_EOLICO_{args[i][9:]}M'

    # names = ['_RAIN','_SWDOWN','_TEMP','_VAPOR','_WIND']
    names = ['_' + map[arg] for arg in args] #variaveis para quais seram geradas os .webm
    grade = ['D01','D02','D03','D04','D05']  #grades escolhidas

    files = [(g+n[1:], WRFoutput+g+n, WRFoutput) for g in grade for n in names]

    try:
        processes.starmap(generateGifs,files)   #executa a funcao de criacao dos videos
        processes.close()
    except Exception as err:
        print(err)

