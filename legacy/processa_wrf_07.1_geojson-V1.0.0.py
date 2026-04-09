# -*- Coding: UTF-8 -*-
#coding: utf-8
from multiprocessing import Pool
from itertools import product
import numpy as np
import netCDF4
from datetime import timezone, datetime
import os
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
#from utility import getFileNames
from warnings import filterwarnings
#import imageio
#import moviepy as mp
import glob
import sys
from wrf import interplevel
import matplotlib.colors as mcolors
# usar json nativo para serialização mais leve
from collections import OrderedDict
import json
#import pdb
filterwarnings('ignore')
#print (np.__version__)


#funçao que define o numero das imagens geradas como sendo de 3 algarismos
def toNum(arg):
    arg = str(arg)
    if len(arg) == 2:
        arg = '0'+arg
    elif len(arg) == 1:
        arg = '00'+arg
    return arg

#funçao que retorna o menor e maior valor de um variavel especifica
def getLowHigh(variable):
    varflat = variable[1:,:].flatten()
    varlow, varhigh = np.amin(varflat), np.percentile(varflat,98)
    return varlow, varhigh

#funcao que calcula a velocidade minima e maxima do vento
def getLowHighWind(variable1, variable2):
    varflat1, varflat2 = variable1[1:,:].flatten(), variable2[1:,:].flatten()
    speed = np.sqrt(varflat1*varflat1 + varflat2*varflat2)   #speed² = U² + V²
    varlow , varhigh = np.amin(speed), np.amax(speed) 
    return varlow, varhigh

#funcao que calcula a velocidade minima e maxima da chuva
def getLowHighRain(variable):
    # Não mutar o array original: calcular diferenças ao longo do eixo temporal
    # Assume-se que 'variable' é acumulado no tempo; diffs dá a precipitação incremental
    if variable.ndim < 3:
        # Formato inesperado; cair back para comportamento robusto
        arr = np.asarray(variable)
        varflat = arr.flatten()
        return float(np.nanmin(varflat)), float(np.nanmax(varflat))
    diffs = np.diff(np.asarray(variable), axis=0)  # shape: (t-1, y, x)
    if diffs.size == 0:
        return 0.0, 0.0
    varflat = diffs.flatten()
    varlow, varhigh = np.nanmin(varflat), np.nanmax(varflat)
    return float(varlow), float(varhigh)

def cmap_saturado(cmap, saturation_factor):
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.linspace(0, 1, cmap.N))
    hsv_colors = mcolors.rgb_to_hsv(colors[:, :3])  # Converte para HSV
    hsv_colors[:, 1] *= saturation_factor           # Ajusta a saturação
    hsv_colors[:, 1] = np.clip(hsv_colors[:, 1], 0, 1)  # Garante que esteja no intervalo [0, 1]
    adjusted_colors = mcolors.hsv_to_rgb(hsv_colors)  # Converte de volta para RGB
    return mcolors.ListedColormap(adjusted_colors)

#cria arquivo geoJSON
def create_grid_geojson(lon,lat,resolution_x,resolution_y,colormap):
    # Criar GeoJSON como dicionários nativos (mais leve que objetos do pacote geojson)
    features = []
    n_rows, n_cols = lon.shape
    for i in range(n_rows):
        for j in range(n_cols):
            # calcular cantos do polígono
            if i == 0:
                lat_top = float(lat[i, j] + (lat[i, j] - lat[i+1, j]) / 2)
                lat_bottom = float((lat[i, j] + lat[i+1, j]) / 2)
            elif i == n_rows - 1:
                lat_top = float((lat[i-1, j] + lat[i, j]) / 2)
                lat_bottom = float(lat[i, j] - (lat[i-1, j] - lat[i, j]) / 2)
            else:
                lat_top = float((lat[i-1, j] + lat[i, j]) / 2)
                lat_bottom = float((lat[i, j] + lat[i+1, j]) / 2)

            if j == 0:
                lon_left = float(lon[i, j] - (lon[i, j+1] - lon[i, j]) / 2)
                lon_right = float((lon[i, j] + lon[i, j+1]) / 2)
            elif j == n_cols - 1:
                lon_left = float((lon[i, j-1] + lon[i, j]) / 2)
                lon_right = float(lon[i, j] + (lon[i, j] - lon[i, j-1]) / 2)
            else:
                lon_left = float((lon[i, j-1] + lon[i, j]) / 2)
                lon_right = float((lon[i, j] + lon[i, j+1]) / 2)

            polygon_coords = [[
                [lon_left, lat_bottom],
                [lon_right, lat_bottom],
                [lon_right, lat_top],
                [lon_left, lat_top],
                [lon_left, lat_bottom]
            ]]

            # Calcular índice plano: cada célula é um elemento no array JSON
            cell_index = i * n_cols + j

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": polygon_coords
                },
                "properties": {
                    "linear_index": int(cell_index)
                }
            }
            features.append(feature)

    metadata = {
        "colormap": colormap,
        "resolucao_m": [float(resolution_x), float(resolution_y)]
    }

    geojson_obj = OrderedDict([
        ("type", "FeatureCollection"),
        ("metadata", metadata),
        ("features", features)
    ])

    return geojson_obj

def create_values_json(var, scale_min, scale_max, date_time, wind_data=None):
    # Normalizar entrada para array numpy; suportar MaskedArray
    if isinstance(var, np.ma.MaskedArray):
        arr = var.filled(np.nan)
    else:
        arr = np.asarray(var)

    flat = arr.flatten()

    # Converter NaNs para None para serialização JSON ou arredondar valores válidos
    values_rounded = []
    for v in flat:
        try:
            if np.isnan(v):
                values_rounded.append(None)
            else:
                values_rounded.append(float(np.round(v, 2)))
        except Exception:
            # valor não numérico; tentar converter
            try:
                values_rounded.append(float(v))
            except Exception:
                values_rounded.append(None)

    # Tratar date_time com segurança
    if date_time is None:
        date_time_str = "N/A"
    else:
        try:
            dt = date_time.replace(minute=0, second=0, microsecond=0, tzinfo=None)
            date_time_str = dt.strftime("%d/%m/%Y %H:%M:%S")
        except Exception:
            try:
                date_time_str = str(date_time)
            except Exception:
                date_time_str = "N/A"

    scale_values = [float(round(x, 2)) for x in np.linspace(scale_min, scale_max, 6)]

    metadata = {
        "scale_values": scale_values,
        "date_time": date_time_str
    }
    
    # Adicionar dados de vento se fornecidos
    if wind_data is not None:
        metadata["wind"] = wind_data

    return {
        "metadata": metadata,
        "values": values_rounded
    }


def ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def read_grid(dataset):
    xlat = dataset.variables['XLAT'][:,:,:]
    xlong = dataset.variables['XLONG'][:,:,:]
    lon = xlong[:1, :,:].squeeze()
    lat = xlat[:1, :, :].squeeze()
    DX = dataset.getncattr("DX")
    DY = dataset.getncattr("DY")
    return lon, lat, DX, DY


def save_geojson(output_dir, filename_prefix, lon, lat, DX, DY, colormap):
    ensure_dir(output_dir)
    geojson_obj = create_grid_geojson(lon, lat, DX, DY, colormap)
    out_path = os.path.join(output_dir, filename_prefix + ".geojson")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson_obj, f, indent=2, ensure_ascii=False)
    print(f'    + PYTHON - ' + out_path)


def save_values_file(output_dir, name, json_obj):
    ensure_dir(output_dir)
    out_path = os.path.join(output_dir, name + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)
    print(f'    + PYTHON - ' + out_path)


def vertical_interpolate_to_target(speed_levels, height_levels, target_height):
    """Interpolação linear vertical para `target_height`.
    speed_levels, height_levels: arrays with shape (levels, ny, nx).
    Retorna array (ny, nx) com valores interpolados (float) ou np.nan quando não possível.
    Implementação vetorizada: ordena níveis por altura e interpola por coluna.
    """
    speed_levels = np.asarray(speed_levels)
    height_levels = np.asarray(height_levels)
    if speed_levels.ndim != 3 or height_levels.ndim != 3:
        raise ValueError('speed_levels and height_levels must be 3D arrays')

    levels, ny, nx = speed_levels.shape
    N = ny * nx

    h = height_levels.reshape(levels, N)
    s = speed_levels.reshape(levels, N)

    # ordenar por altura (NaNs vão para o final)
    order = np.argsort(h, axis=0)
    h_sorted = np.take_along_axis(h, order, axis=0)
    s_sorted = np.take_along_axis(s, order, axis=0)

    # máscara de validade (altura e velocidade válidas)
    valid = ~np.isnan(h_sorted) & ~np.isnan(s_sorted)
    valid_count = np.sum(valid, axis=0)

    result = np.full(N, np.nan, dtype=float)

    # caso com apenas 1 valor válido -> usar esse valor
    single_mask = valid_count == 1
    if np.any(single_mask):
        # índice do único válido
        idx_single = np.argmax(valid, axis=0)
        cols = np.where(single_mask)[0]
        result[cols] = s_sorted[idx_single[cols], cols]

    # casos com 2 ou mais valores válidos -> interpolar
    multi_mask = valid_count >= 2
    if np.any(multi_mask):
        cols = np.where(multi_mask)[0]
        h_m = h_sorted[:, cols]
        s_m = s_sorted[:, cols]

        # localizar o primeiro nível com altura > target
        greater = h_m > target_height
        any_greater = np.any(greater, axis=0)
        first_gt = np.argmax(greater, axis=0)

        # índice inferior para interpolação
        lower_idx = np.where(any_greater, first_gt - 1, valid_count[cols] - 2)
        lower_idx = np.clip(lower_idx, 0, levels - 2)

        col_indices = np.arange(cols.size)
        h1 = h_m[lower_idx, col_indices]
        h2 = h_m[lower_idx + 1, col_indices]
        s1 = s_m[lower_idx, col_indices]
        s2 = s_m[lower_idx + 1, col_indices]

        # evitar divisão por zero
        denom = (h2 - h1)
        with np.errstate(invalid='ignore', divide='ignore'):
            frac = (target_height - h1) / denom
        frac = np.where(np.isfinite(frac), frac, 0.0)

        interp_vals = s1 + frac * (s2 - s1)
        result[cols] = interp_vals

    return result.reshape(ny, nx)

def compute_wind_vectors_at_height(U_central, V_central, altura_ajustada, target_height, downsampling=4):
    """Computa vetores de vento interpolados para altura desejada com downsampling.
    
    Retorna dicionário com:
    - angles: lista com ângulos do vento (graus) - 0=de Norte, 90=de Leste
    - magnitudes: lista com magnitudes (m/s)
    - downsampled_angles: ângulos subamostrados para renderização
    - downsampled_magnitudes: magnitudes subamostradas
    - downsampled_linear_indices: índices lineares (row-major) dos vetores subamostrados
    """
    levels, _, ny, nx = U_central.shape
    
    # Interpolar componentes U e V para altura alvo
    U_target = np.full((ny, nx), np.nan)
    V_target = np.full((ny, nx), np.nan)
    
    for t in range(U_central.shape[0]):
        try:
            U_t = vertical_interpolate_to_target(U_central[t], altura_ajustada[t], target_height)
            V_t = vertical_interpolate_to_target(V_central[t], altura_ajustada[t], target_height)
            # Usar média temporal para suavização
            U_target = np.nanmean([U_target, U_t], axis=0) if not np.all(np.isnan(U_target)) else U_t
            V_target = np.nanmean([V_target, V_t], axis=0) if not np.all(np.isnan(V_target)) else V_t
        except:
            pass
    
    # Calcular magnitude e ângulo (convenção meteorológica: 0=Norte, 90=Leste)
    magnitude = np.sqrt(U_target**2 + V_target**2)
    angle = np.arctan2(U_target, V_target) * 180 / np.pi
    angle = np.where(angle < 0, angle + 360, angle)
    
    # Downsampling para reduzir poluição visual
    downsampled_angles = []
    downsampled_magnitudes = []
    downsampled_linear_indices = []
    all_angles = []
    all_magnitudes = []
    
    for i in range(0, ny, downsampling):
        for j in range(0, nx, downsampling):
            if not np.isnan(angle[i, j]):
                # Calcular índice linear em row-major order: i * nx + j
                linear_idx = int(i * nx + j)
                downsampled_linear_indices.append(linear_idx)
                downsampled_angles.append(float(angle[i, j]))
                downsampled_magnitudes.append(float(magnitude[i, j]))
    
    # Converter arrays completos para listas (JSON serializable)
    all_angles = angle.flatten().tolist()
    all_magnitudes = magnitude.flatten().tolist()
    
    return {
        "angles": all_angles,
        "magnitudes": all_magnitudes,
        "downsampled_angles": downsampled_angles,
        "downsampled_magnitudes": downsampled_magnitudes,
        "downsampled_linear_indices": downsampled_linear_indices
    }

#funcao que desenha os mapas usando como argumento o tipo do mapa e o seu dataset correspondente
def drawmap(tipo,dataset):
    #paleta de cor de cada variavel
    colormap = {'temperature':'hot_r','wind':'PuBu','vapor':'YlGnBu','pressure':'Blues','rain':'afmhot_r','HFX':'jet','LH':'jet','SWDOWN':'hot_r','weibull':'jet',f'poteolico{tipo[9:]}':'Blues'} #tipo de colormap usado em cada grafico
    #caminho de onde se localiza o python (diretorio de trabalho)
    dir_scriptpy = sys.argv[1]+'/'
    #caminho de output das figuras
    WRFoutput = sys.argv[2]+'/'
    grade = dataset[dataset.find('d0'):dataset.find('d0')+3]; grade = grade.upper()
    dataset = netCDF4.Dataset(dataset)
    times_array = dataset.variables['Times'][:] 
    DX = dataset.getncattr("DX"); DY = dataset.getncattr("DY")  #resolucao espacial em metros 
    dates = []; datesStr = [];   names = []
    #datas      #texto da data   #identificacao do mapa 
                                 #(grade+variavel+numero)

    map = {'HFX':'HFX','LH':'LH','pressure':'PRES','rain':'RAIN','SWDOWN':'SWDOWN','temperature':'TEMP','vapor':'VAPOR','wind':'WIND','weibull':'K_WEIB',f'poteolico{tipo[9:]}':f'POT_EOLICO_{tipo[9:]}M'}
    week = {1:'Segunda',2:'Terça',3:'Quarta',4:'Quinta',5:'Sexta',6:'Sábado',7:'Domingo'}
    #desc = {'HFX':'Calor Sensível','pressure':'Pressão Atmosférica (Nível do Mar)','rain':'Precipitação','SWDOWN':'Radiação Global','temperature':'Temperatura (2 m)','vapor':'Umidade Específica','wind':'Velocidade do Vento (10 m)'}
    
    month = str(datetime.now().month); day = str(datetime.now().day)
    day = sys.argv[4]
    month = sys.argv[5]
    year = sys.argv[6]
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
            start = current_date.strftime("%d/%m/%Y %H") + " (UTC)\n"   #data de inicio da analise UTC
        if it < 3:
            cd = current_date.replace(tzinfo=timezone.utc).astimezone(tz=None)   #data e horario da previsao
            datesStr.append("\nInício Análise: "+start+"Previsão: "+cd.strftime("%d/%m/%Y %H")+"HL ("+week[cd.isoweekday()]+")")
            names.append(grade+"_"+map[tipo]+"_"+toNum(it))
            it+=1
            continue
        cd = current_date.replace(tzinfo=timezone.utc).astimezone(tz=None)
        #graficos de radiacao solar so sao gerados entre 6h e 18h 
        if tipo == 'SWDOWN' and (cd.hour < 6 or cd.hour > 18):
            datesStr.append("\nInício Análise: "+start+"Previsão: "+cd.strftime("%d/%m/%Y %H")+"HL ("+week[cd.isoweekday()]+")")
            names.append(grade+"_"+map[tipo]+"_"+toNum(it))
            it+=1
            continue
        datesStr.append("\nInício Análise: "+start+"Previsão: "+cd.strftime("%d/%m/%Y %H")+"HL ("+week[cd.isoweekday()]+")")
        names.append(grade+"_"+map[tipo]+"_"+toNum(it-2))
        dates.append((it,cd)) #primeiros 3 indices temporais '0,1,2' são excluidos na geracao dos raster/json 
        it+=1

    #grafico da temperatura com as linhas de pressao
    if tipo == 'temperature':
        xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
        lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
        hlat, llat = np.amax(xlat), np.amin(xlat)   #valor maximo e minimo da latitude (eixo y do grafico)
        hlong, llong = np.amax(xlong), np.amin(xlong)   #valor maximo e minimo da longitude (eixo x do grafico)
        var = dataset.variables['T2'][:,:,:].squeeze()   #variavel = temperatura
        varmin, varmax = getLowHigh(var);   #valor minimo e maximo da temperatura
        varmax = varmax - 273;        varmin = varmin - 273   #conversao de K para C°
        var2 = dataset.variables['PSFC'][:,:,:].squeeze()   #variavel = pressao
        var2 /= 100
        try:
            save_geojson(WRFoutput, grade + "_" + map[tipo], lon, lat, DX, DY, colormap[tipo])
        except Exception as e:
            raise Exception(f'Erro ao criar o arquivo GeoJSON {grade + "_" + map[tipo]}: {e}')
        
        for i,date in dates:

            celsius = var[i:i+1,:,:].squeeze() - 273.15   #conversao da temperatura em K para C°
            pressure = var2[i:i+1,:,:].squeeze()

            try:
                json_obj = create_values_json(celsius, varmin, varmax, date) 
                save_values_file(sys.argv[3], names[i], json_obj)
            except Exception as e:
                raise Exception(f'Erro ao criar o arquivo JSON {names[i]}: {e}')             

    #grafico da pressao 
    elif tipo == 'pressure':
        xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
        lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
        hlat, llat = np.amax(xlat), np.amin(xlat)   #valor maximo e minimo da latitude (eixo y do grafico)
        hlong, llong = np.amax(xlong), np.amin(xlong)   #valor maximo e minimo da longitude (eixo x do grafico)
        var = dataset.variables['PSFC'][:,:,:].squeeze()   #variavel = pressao
        varmin, varmax = getLowHigh(var);   #valor minimo e maximo da pressao
        varmax = varmax/100;        varmin = varmin/100
        try:
            save_geojson(WRFoutput, grade + "_" + map[tipo], lon, lat, DX, DY, colormap[tipo])
        except Exception as e:
            raise Exception(f'Erro ao criar o arquivo GeoJSON {grade + "_" + map[tipo]}: {e}')
        
        for i,date in dates:

            mbar = var[i:i+1,:,:].squeeze()/100

            try:
                json_obj = create_values_json(mbar, varmin, varmax, date) 
                save_values_file(sys.argv[3], names[i], json_obj)
            except Exception as e:
                raise Exception(f'Erro ao criar o arquivo JSON {names[i]}: {e}')
    
    #grafico da umidade especifica
    elif tipo == 'vapor':
        xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
        lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
        hlat, llat = np.amax(xlat), np.amin(xlat)   #valor maximo e minimo da latitude (eixo y do grafico)
        hlong, llong = np.amax(xlong), np.amin(xlong)   #valor maximo e minimo da longitude (eixo x do grafico)
        var = dataset.variables['Q2'][:,:,:].squeeze()   #variavel = umidade especifica
        varmin, varmax = getLowHigh(var);   #valor minimo e maximo da umidade especifica
        varmax = varmax*1000;        varmin = varmin*1000
        try:
            save_geojson(WRFoutput, grade + "_" + map[tipo], lon, lat, DX, DY, colormap[tipo])
        except Exception as e:
            raise Exception(f'Erro ao criar o arquivo GeoJSON {grade + "_" + map[tipo]}: {e}')
        
        for i,date in dates:

            gkg = var[i:i+1,:,:].squeeze()*1000

            try:
                json_obj = create_values_json(gkg, varmin, varmax, date) 
                save_values_file(sys.argv[3], names[i], json_obj)
            except Exception as e:
                raise Exception(f'Erro ao criar o arquivo JSON {names[i]}: {e}')
    
    #grafico da velocidade do vento
    elif tipo == 'wind':
        xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
        lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
        hlat, llat = np.amax(xlat), np.amin(xlat)   #valor maximo e minimo da latitude (eixo y do grafico)
        hlong, llong = np.amax(xlong), np.amin(xlong)   #valor maximo e minimo da longitude (eixo x do grafico)
        u10 = dataset.variables['U10'][:].squeeze(); v10 = dataset.variables['V10'][:].squeeze()
        varmin, varmax = getLowHighWind(u10,v10)
        try:
            save_geojson(WRFoutput, grade + "_" + map[tipo], lon, lat, DX, DY, colormap[tipo])
        except Exception as e:
            raise Exception(f'Erro ao criar o arquivo GeoJSON {grade + "_" + map[tipo]}: {e}')
        
        for i,date in dates:

            u = u10[i:i+1,:,:].squeeze()
            v = v10[i:i+1,:,:].squeeze()
            speed = np.sqrt(u*u + v*v)

            try:
                json_obj = create_values_json(speed, varmin, varmax, date) 
                save_values_file(sys.argv[3], names[i], json_obj)
            except Exception as e:
                raise Exception(f'Erro ao criar o arquivo JSON {names[i]}: {e}')

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
        try:
            save_geojson(WRFoutput, grade + "_" + map[tipo], lon, lat, DX, DY, colormap[tipo])
        except Exception as e:
            raise Exception(f'Erro ao criar o arquivo GeoJSON {grade + "_" + map[tipo]}: {e}')
        
        for i,date in dates:
            
            # mm = var[i:i+1,:,:]

            if i == 1:
               mm = var[i:i+1,:,:].squeeze()
            else:
               mm = var[i:i+1,:,:].squeeze()-var[i-1:i,:,:].squeeze()

            try:
                json_obj = create_values_json(mm, varmin, varmax, date) 
                save_values_file(sys.argv[3], names[i], json_obj)
            except Exception as e:
                raise Exception(f'Erro ao criar o arquivo JSON {names[i]}: {e}')
    
    #grafico do calor sensivel
    elif tipo == 'HFX':
        xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
        lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
        hlat, llat = np.amax(xlat), np.amin(xlat)   #valor maximo e minimo da latitude (eixo y do grafico)
        hlong, llong = np.amax(xlong), np.amin(xlong)   #valor maximo e minimo da longitude (eixo x do grafico)
        var = dataset.variables['HFX'][:,:,:].squeeze()
        varmin, varmax = getLowHigh(var);   #valor minimo e maximo da 
        try:
            save_geojson(WRFoutput, grade + "_" + map[tipo], lon, lat, DX, DY, colormap[tipo])
        except Exception as e:
            raise Exception(f'Erro ao criar o arquivo GeoJSON {grade + "_" + map[tipo]}: {e}')
        
        for i,date in dates:

            mm = var[i:i+1,:,:].squeeze()

            try:
                json_obj = create_values_json(mm, varmin, varmax, date) 
                # Salvando em um arquivo
                with open(f"{sys.argv[3]}/{names[i]}.json", "w") as f:
                    json.dump(json_obj, f, indent=2)
                print(f'    + PYTHON - ' + sys.argv[3] + "/"  + names[i] +".json")
            except Exception as e:
                raise Exception(f'Erro ao criar o arquivo JSON {names[i]}: {e}')
    
    #grafico do calor latente
    elif tipo == 'LH':
        xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
        lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
        hlat, llat = np.amax(xlat), np.amin(xlat)   #valor maximo e minimo da latitude (eixo y do grafico)
        hlong, llong = np.amax(xlong), np.amin(xlong)   #valor maximo e minimo da longitude (eixo x do grafico)
        var = dataset.variables['LH'][:,:,:].squeeze()
        varmin, varmax = getLowHigh(var);   #valor minimo e maximo da 
        try:
            save_geojson(WRFoutput, grade + "_" + map[tipo], lon, lat, DX, DY, colormap[tipo])
        except Exception as e:
            raise Exception(f'Erro ao criar o arquivo GeoJSON {grade + "_" + map[tipo]}: {e}')
        
        for i,date in dates:

            mm = var[i:i+1,:,:].squeeze()

            try:
                json_obj = create_values_json(mm, varmin, varmax, date) 
                save_values_file(sys.argv[3], names[i], json_obj)
            except Exception as e:
                raise Exception(f'Erro ao criar o arquivo JSON {names[i]}: {e}')

    #grafico da radiacao solar
    elif tipo == 'SWDOWN':
        xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
        lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
        hlat, llat = np.amax(xlat), np.amin(xlat)   #valor maximo e minimo da latitude (eixo y do grafico)
        hlong, llong = np.amax(xlong), np.amin(xlong)   #valor maximo e minimo da longitude (eixo x do grafico)
        var = dataset.variables['SWDOWN'][:,:,:].squeeze()
        varmin, varmax = getLowHigh(var);   #valor minimo e maximo da 
        try:
            save_geojson(WRFoutput, grade + "_" + map[tipo], lon, lat, DX, DY, colormap[tipo])
        except Exception as e:
            raise Exception(f'Erro ao criar o arquivo GeoJSON {grade + "_" + map[tipo]}: {e}')
        
        for i,date in dates:

            mm = var[i:i+1,:,:].squeeze()

            try:
                json_obj = create_values_json(mm, varmin, varmax, date) 
                save_values_file(sys.argv[3], names[i], json_obj)
            except Exception as e:
                raise Exception(f'Erro ao criar o arquivo JSON {names[i]}: {e}')
    
    #grafico fator k de weibull
    elif tipo == 'weibull':
        xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
        lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
        hlat, llat = np.amax(xlat), np.amin(xlat)   #valor maximo e minimo da latitude (eixo y do grafico)
        hlong, llong = np.amax(xlong), np.amin(xlong)   #valor maximo e minimo da longitude (eixo x do grafico)
        U = dataset.variables['U'][:].squeeze(); V = dataset.variables['V'][:].squeeze()  #velocidades horizontais e verticais do vento na borda das grades
        #interpolacao das velocidades para representarem os valores nos centros das grades
        U_central = (U[:, :, :, :-1] + U[:, :, :, 1:])/2; V_central = (V[:, :, :-1, :] + V[:, :, 1:, :])/2;       
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
        
        # Interpolação vertical para 100 m usando função vetorizada (mais robusta e rápida)
        for ti in range(speed3d.shape[0]):
            try:
                speed3d[ti,:,:] = vertical_interpolate_to_target(speed4d[ti,:,:,:], altura_ajustada[ti,:,:,:], 100)
            except Exception:
                speed3d[ti,:,:] = np.asarray(interplevel(speed4d[ti,:,:,:], altura_ajustada[ti,:,:,:], 100))

        # Cálculo vetorizado do fator k de Weibull (ignora NaNs e protege divisão por zero)
        with np.errstate(invalid='ignore', divide='ignore'):
            std = np.nanstd(speed3d[1:,...], axis=0)
            mean = np.nanmean(speed3d[1:,...], axis=0)
            ratio = np.where(mean > 0, std / mean, np.nan)
            fator_k = np.power(ratio, -1.086)
        
        try:
            save_geojson(WRFoutput, grade + "_" + map[tipo], lon, lat, DX, DY, colormap[tipo])
        except Exception as e:
            raise Exception(f'Erro ao criar o arquivo GeoJSON {grade + "_" + map[tipo]}: {e}')

        try:
            varmin = float(np.nanmin(fator_k))
            varmax = float(np.nanmax(fator_k))
            json_obj = create_values_json(fator_k, varmin, varmax, None)
            save_values_file(sys.argv[3], grade + "_" + map[tipo], json_obj)
        except Exception as e:
            raise Exception(f'Erro ao criar o arquivo JSON {grade + "_" + map[tipo]}: {e}')

    #grafico potencial eolico  
    elif 'poteolico' in tipo:
        xlat, xlong = dataset.variables['XLAT'][:,:,:], dataset.variables['XLONG'][:,:,:]
        lon, lat = xlong[:1, :,:].squeeze(), xlat[:1, :, :].squeeze()
        hlat, llat = np.amax(xlat), np.amin(xlat)   #valor maximo e minimo da latitude (eixo y do grafico)
        hlong, llong = np.amax(xlong), np.amin(xlong)   #valor maximo e minimo da longitude (eixo x do grafico)

        U = dataset.variables['U'][:].squeeze(); V = dataset.variables['V'][:].squeeze()  #velocidades horizontais e verticais do vento na borda das grades
        U_central = (U[:, :, :, :-1] + U[:, :, :, 1:])/2; V_central = (V[:, :, :-1, :] + V[:, :, 1:, :])/2;   #interpolacao das velocidades para representarem os valores nos centros das grades     

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
        for ti in range(speed3d.shape[0]):
            try:
                speed3d[ti,:,:] = vertical_interpolate_to_target(speed4d[ti,:,:,:], altura_ajustada[ti,:,:,:], altura_desejada)
            except Exception:
                speed3d[ti,:,:] = np.asarray(interplevel(speed4d[ti,:,:,:], altura_ajustada[ti,:,:,:], altura_desejada))
        arr_speed3d = np.asarray(speed3d)
        speed3d_corrigida = arr_speed3d[~np.isnan(arr_speed3d)]
        if speed3d_corrigida.size:
            varmin = float(np.min(speed3d_corrigida))
            varmax = float(np.percentile(speed3d_corrigida, 98))
        else:
            varmin, varmax = 0.0, 0.0
        try:
            save_geojson(WRFoutput, grade + "_" + map[tipo], lon, lat, DX, DY, colormap[tipo])
        except Exception as e:
            raise Exception(f'Erro ao criar o arquivo GeoJSON {grade + "_" + map[tipo]}: {e}')
        
        for i,date in dates:
             
            #calculo do mapa de velocidade resultante do vento para cada hora analisada em todas os niveis de altura 
            u = U_central[i:i+1,:,:,:].squeeze()
            v = V_central[i:i+1,:,:,:].squeeze()
            speed3d = np.sqrt(u*u + v*v)
            
            #calculo do mapa de velocidade resultante do vento para cada hora analisada em um nivel de altura especificado (interpolaçao)
            pot_eolico = interplevel(speed3d, altura_ajustada[i,:,:,:], altura_desejada)

            # preencher NaNs por interpolação vertical vetorizada
            try:
                pot_eolico = vertical_interpolate_to_target(speed3d, altura_ajustada[i,:,:,:], altura_desejada)
            except Exception:
                pot_eolico = np.asarray(interplevel(speed3d, altura_ajustada[i,:,:,:], altura_desejada))
            pot_eolico = np.asarray(pot_eolico)

            # Calcular vetores de vento para os dados do timestamp atual
            try:
                wind_data = compute_wind_vectors_at_height(
                    U_central[i:i+1], V_central[i:i+1], 
                    altura_ajustada[i:i+1], altura_desejada, downsampling=4
                )
            except Exception as e:
                print(f'    + PYTHON - Aviso: Falha ao calcular vetores de vento: {e}')
                wind_data = None

            try:
                json_obj = create_values_json(pot_eolico, varmin, varmax, date, wind_data=wind_data)
                save_values_file(sys.argv[3], names[i], json_obj)
            except Exception as e:
                raise Exception(f'Erro ao criar o arquivo JSON {names[i]}: {e}')


# def generateGifs(name, files, path_output):
#     try:
#         files = glob.glob(files+'*.png')
#         files.sort(key=os.path.getmtime)
#         print("peguei o glob")

#         images = [imageio.imread(x) for x in files]

#         imageio.mimsave(path_output+name+'.gif',images,format='GIF',duration=0.5)
#         print("peguei o gif")

#         clip = mp.video.io.VideoFileClip(path_output+name+'.gif')
#         clip.write_videofile(path_output+name+'.webm',audio=False,threads=1)
#         print("peguei o video")


#         os.remove(path_output+name+'.gif')
#     except Exception as err:
#         print(err)

if __name__ == '__main__':
    
    #graphs.png for each hour (define quais tipos (variaveis) de graficos seram gerados)
    #args = ['temperature','pressure','vapor','wind','rain','HFX','LH','SWDOWN'] 
    args = [arg for arg in sys.argv[13:]] #variaveis de saida
    #caminho de input dos dados wrf 
    #pathX = ['/home/murilo/leal/mapas/wrfout_d0X_2024-06-19_00_00_00']


    #sys.argv[1]  sys.argv[2]  sys.argv[3]  sys.argv[4]  sys.argv[5]    sys.argv[6]   sys.argv[7]   sys.argv[8]   sys.argv[9]   sys.argv[10]   sys.argv[11]
    #$dir_script  $WRFoutput   $day $month  $year        $num_min_dom   $num_max_dom  $path1        $path2        $path3        $path4         $path5



    #dir_scriptpy $WRFoutput $day $month $year $FIGini $FIGfim $path1 $path2 $path3 $path4 $path5

    print(f' ')
    print(f'    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐')
    print(f'    + PYTHON - EXECUTANDO O PYTHON')
    print(f'    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘')
    print(f' ')


    print(f'    + PYTHON - dir_scriptpy = {sys.argv[1]}')
    print(f'    + PYTHON - WRFoutputGEOJSON    = {sys.argv[2]}')
    print(f'    + PYTHON - WRFoutputJSON    = {sys.argv[3]}')
    print(f'    + PYTHON - day          = {sys.argv[4]}')
    print(f'    + PYTHON - month        = {sys.argv[5]}')
    print(f'    + PYTHON - year         = {sys.argv[6]}')
    print(f'    + PYTHON - num_min_dom  = {sys.argv[7]}')
    print(f'    + PYTHON - num_max_dom  = {sys.argv[8]}')
    print(f'    + PYTHON - path1        = {sys.argv[9]}')
    print(f'    + PYTHON - path2        = {sys.argv[10]}')
    print(f'    + PYTHON - path3        = {sys.argv[11]}')
    print(f'    + PYTHON - path4        = {sys.argv[12]}')
    
    if np.size(args) == 0:
        print('Nenhuma variável foi selecionada')
        exit()
    else:
        [print(f'    + PYTHON - variavel_saida_{i+1}        = {args[i]}') for i in range(np.size(args))]

    #define de quais arquivos wrf serao gerados as imagens 
    num_min_dom= int(sys.argv[7]) #inicio - wrf_d+num_min_dom
    num_max_dom= int(sys.argv[8]) #fim - wrf_d+num_max_dom
    
    #num_min_dom <= num_max_dom
    if num_min_dom > num_max_dom:
        print('    + PYTHON - O primeiro argumento wrf deve ser menor ou igual ao segundo')
        exit()
    #se num_min_dom == num_max_dom, gera os graficos correspondente a apenas um arquivo wrf 
    elif num_min_dom == num_max_dom:
        files = [sys.argv[8+num_min_dom]]
        #files = path
        print(f'    + PYTHON -  {files}')
    #gera os graficos correspondentes ao wrf_d+num_min_dom ate wrf_d+num_max_dom
    else:
        files = [sys.argv[8+i] for i in range(num_min_dom,num_max_dom+1)]
        #files = path
        print(f'    + PYTHON -  {files}')

    final = product(args,files)

    # executar drawmap em paralelo com context manager para garantir fechamento
    with Pool() as processes:
        processes.starmap(drawmap, final)

    # make a .webm with the graphs
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
        #processes.starmap(generateGifs,files)   #executa a funcao de criacao dos videos 
        pass
    except Exception as err:
        print(err)

