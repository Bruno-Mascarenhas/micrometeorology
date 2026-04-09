#!/bin/bash

echo " "
echo '    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
echo "    + PYTHON - EXECUTANDO SCRIPT CHAMANDO O PYTHON"
echo '    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
echo " "


#local de trabalho
#-----------------
dir_local=$( grep dir_local /home/models/WRF/wrf-op/.FUNCTIONS_WRF_environment | awk '{print $2}' )
dir_python=$( grep dir_python /home/models/WRF/wrf-op/.FUNCTIONS_WRF_environment | awk '{print $2}' )
export PATH=$PATH:${dir_local}:.


#           0                     1         2           3       4       5
#vetpython=($processa_wrf_figuras $yyyymmdd $WRFoutput1 $DOMini $DOMfim ${GRAFICOS_WRF[@]}) #fazendo as figuras de grade em grade pois o python consome muita memoria


#lendo as variaveis passadas pelo SYSTEM
#---------------------------------------
vetpython=("$@") #recebendo o vetor das variaveis

processa_wrf_figuras=${vetpython[0]}
yyyymmdd=${vetpython[1]}
WRFoutput=${vetpython[2]}
DOMini=${vetpython[3]}
DOMfim=${vetpython[4]}
GRAFICOS_WRF=${vetpython[@]:5:${#vetpython[*]}-1}

year=$(echo ${yyyymmdd}  | cut -b 1-4)
month=$(echo ${yyyymmdd} | cut -b 5-6)
day=$(echo ${yyyymmdd}   | cut -b 7-8)

#-----------------------------------------------------       
echo "    + PYTHON - no. de variaveis recebidas                 = ${#vetpython[*]}"
echo "    + PYTHON - local de trabalho (dir_local)              = $dir_local"
echo "    + PYTHON - local do python (dir_python)               = $dir_python"
echo "    + PYTHON - local das figuras (dir_fig)                = $WRFoutput"
echo "    + PYTHON - local do WRF (wrf_simulacao)               = $WRFoutput"
echo "    + PYTHON - figuras serao geradas para o(s) dominio(s) = d0$DOMini ao d0$DOMfim"
echo "    + PYTHON - ativa o ambiente virtual basemap"
#-----------------------------------------------------    


#local do script python para figuras do wrf
dir_scriptpy=$dir_local/d-python/wrfpy  
eval "$($dir_python/conda shell.bash hook)"
conda activate basemap
$dir_python/conda env list

path1=$WRFoutput/wrfout_d01_${year}-${month}-${day}_00:00:00
path2=$WRFoutput/wrfout_d02_${year}-${month}-${day}_00:00:00
path3=$WRFoutput/wrfout_d03_${year}-${month}-${day}_00:00:00
path4=$WRFoutput/wrfout_d04_${year}-${month}-${day}_00:00:00
path5=$WRFoutput/wrfout_d05_${year}-${month}-${day}_00:00:00

echo "    + PYTHON - para executar PYTHON:">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
echo "    + PYTHON - python3 $processa_wrf_figuras $dir_scriptpy $WRFoutput $day $month $year $DOMini $DOMfim $path1 $path2 $path3 $path4 $path5 ${GRAFICOS_WRF[@]}">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
echo "    + PYTHON - ----------------------------">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
#excecuta o python    

python3 $dir_scriptpy/$processa_wrf_figuras $dir_scriptpy $WRFoutput $day $month $year $DOMini $DOMfim $path1 $path2 $path3 $path4 $path5 ${GRAFICOS_WRF[@]}

#se o python nao apagar *.gif, apaga por aqui
\rm -f ${WRFoutput}/*.png

#se o python nao apagar *.gif, apaga por aqui
\rm -f ${WRFoutput}/*.gif

#python3 $dir_scriptpy/main.py
echo "    + PYTHON - desativando o ambiente virtual basemap"
#-----------------------------------------------------------       
conda deactivate

echo "    + PYTHON - voltando ao local de trabalho"
#--------------------------------------------------       
cd ${dir_local}
