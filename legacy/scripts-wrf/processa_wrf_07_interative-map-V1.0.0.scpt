#!/bin/bash

echo " "
echo '    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
echo "    + INTERATIVEMAP - EXECUTANDO SCRIPT CHAMANDO O INTERATIVE MAP"
echo '    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
echo " "

#local de trabalho
#-----------------
dir_local=$( grep dir_local /home/models/WRF/wrf-op/.FUNCTIONS_WRF_environment | awk '{print $2}' )
dir_python=$( grep dir_python /home/models/WRF/wrf-op/.FUNCTIONS_WRF_environment | awk '{print $2}' )
export PATH=$PATH:${dir_local}:.


#definindo o vetor de variaveis para passagem de dados
#           0                     1           2         3            4
#vetintmap=($processa_wrf_geoJSON $WRFoutput1 $yyyymmdd $num_max_dom ${INTERATIVEMAP[@]}})


#lendo as variaveis passadas pelo SYSTEM
#---------------------------------------
vetintmap=("$@") #recebendo o vetor das variaveis

processa_wrf_geoJSON=${vetintmap[0]}
WRFoutput=${vetintmap[1]}
yyyymmdd=${vetintmap[2]}
num_max_dom=${vetintmap[3]}
INTERATIVEMAP=${vetintmap[@]:4:${#vetintmap[*]}-1}

year=$(echo ${yyyymmdd}  | cut -b 1-4)
month=$(echo ${yyyymmdd} | cut -b 5-6)
day=$(echo ${yyyymmdd}   | cut -b 7-8)

#diretorios de trabalho
path_maps=$dir_local/d-interative-map/wrf-map-utils
path_geojson=$path_maps/geoJSON
path_json=$path_maps/JSON


#entrando no d-leaflet
cd $path_maps

#-----------------------------------------------------
echo "    + INTERATIVEMAP - recebendo ${#vetintmap[*]} variaveis do processa_wrf_00_system.scpt:"
echo "    + INTERATIVEMAP - local de trabalho (dir_local)              = $dir_local"
echo "    + INTERATIVEMAP - local do python (dir_python)               = $dir_python"
echo "    + INTERATIVEMAP - local das figuras (dir_fig)                = $path_maps"
echo "    + INTERATIVEMAP - local do WRF (wrf_simulacao)               = $WRFoutput"
echo "    + INTERATIVEMAP - figuras serao geradas para o(s) dominio(s) = d01 a d0$num_max_dom"
#-----------------------------------------------------


echo "    + INTERATIVEMAP - apagando arquivos antigos em $path_maps/geoJSON"
\rm -f $path_maps/geoJSON/*
echo "    + INTERATIVEMAP - apagando arquivos antigos em $path_maps/JSON"
\rm -f $path_maps/JSON/*


echo "    + INTERATIVEMAP - ativa o ambiente virtual INTERATIVEMAP"
eval "$($dir_python/conda shell.bash hook)"
conda activate rastermap
$dir_python/conda env list


#local do script python para figuras do mapa interativo
echo "    + INTERATIVEMAP - executando o $processa_wrf_geoJSON"
python ${path_maps}/processa_wrf_07.1_geojson-V1.0.0.py $path_maps $path_geojson $path_json $day $month $year 1 $num_max_dom $WRFoutput/wrfout_d01_$year-$month-${day}_00:00:00 $WRFoutput/wrfout_d02_$year-$month-${day}_00:00:00 $WRFoutput/wrfout_d03_$year-$month-${day}_00:00:00 $WRFoutput/wrfout_d04_$year-$month-${day}_00:00:00 ${INTERATIVEMAP[@]}


echo "    + INTERATIVEMAP - desativando o ambiente virtual INTERATIVEMAP"
#------------------------------------------------------------
conda deactivate

cd ..
