#!/bin/ksh

#script para extrair dados do WRF
#
#autor:
#MAXSUEL M R PEREIRA 
#
#datas:
#27/01/2020 - codigo original  - maxsuel m r pereira
#26/08/2023 - correcao de bugs - maxsuel m r pereira
#08/07/2023 - integracao ao wrf operacional - maxsuel m r pereira
#15/03/2026 - corrigindo bugs de passagem de dados

echo " "
echo '    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
echo "    + READWRFNC - EXECUTANDO O READWRFNC (TIME SERIES - EXTRACAO DE DADOS)"
echo '    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
echo " "


#local de trabalho
#-----------------
dir_local=$( grep dir_local /home/models/WRF/wrf-op/.FUNCTIONS_WRF_environment | awk '{print $2}' )
export PATH=$PATH:${dir_local}:.


#definindo o vetor de variaveis para passagem para o processa_wrf_readwrfnc
#              0           1         2             3     4     5         6      7               8     9                           10                         11                         12
#vetreadwrfnc=($WRFoutput1 $yyyymmdd $nome_estacao $lat1 $lon1 $timezone $TSgrade $h_estacao $projeto ${#TIME_SERIES_SOUNDING[*]} ${#TIME_SERIES_SURFACE[*]} ${TIME_SERIES_SOUNDING[@]} ${TIME_SERIES_SURFACE[@]})


#lendo as variaveis passadas pelo SYSTEM
#---------------------------------------
vetreadwrfnc=("$@") #recebendo o vetor das variaveis

#pegando dados passados por processa_wrf_00.scpt
WRFoutput=${vetreadwrfnc[0]}
yyyymmdd=${vetreadwrfnc[1]}
nome_estacao=${vetreadwrfnc[2]}
lat1=${vetreadwrfnc[3]}
lon1=${vetreadwrfnc[4]}
timezone=${vetreadwrfnc[5]}
WRFgrade=${vetreadwrfnc[6]}
altura_estacao=${vetreadwrfnc[7]}
projeto=${vetreadwrfnc[8]}
nTSSOUNDING=${vetreadwrfnc[9]}
nTSSURFACE=${vetreadwrfnc[10]}
TIME_SERIES_SOUNDING=${vetreadwrfnc[@]:11:$nTSSOUNDING}
TIME_SERIES_SURFACE=${vetreadwrfnc[@]:11+$nTSSOUNDING:$nTSSURFACE}

LC_NUMERIC=C  #para aceita numeros nao inteiros

echo '    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
echo " "
echo '    + READWRFNC - READING INPUT DATA'
echo "    + READWRFNC - recebendo ${#vetreadwrfnc[*]} variaveis do processa_wrf_00_system.scpt"


#diretorio base
echo '    + READWRFNC - dir_local       = '$dir_local
#diretorio de input (sempre em READ_WRF_NC)
ReadWRFnc=$dir_local/READ_WRF_NC
echo "    + READWRFNC - dados de input  = "$ReadWRFnc

#diretorio de output (em d-output/yyyymmdd/wrfXX - muda todo dia)
echo "    + READWRFNC - dados de output = "$WRFoutput

if [ "$lat1" -le 0. ]; then
    lat2=$((-1*$lat1))
else
    lat2=$lat1
fi

if [ "$lon1" -lt 0. ]; then
    lon2=$((-1*$lon1))
else
    lon2=$lon1
fi

#transformando graus decimais para graus, minutos e segundos
#somente para exibir na tela. nada mais  :-)
lat_deg=$(echo $lat2 | cut -d. -f1)
lat_min1=$(($lat2-$lat_deg))
lat_min2=$(($lat_min1*60))
lat_min=$(echo $lat_min2 | cut -d. -f1)
lat_seg1=$(($lat_min2-$lat_min))
lat_seg=$(($lat_seg1*60))
lon_deg=$(echo $lon2 | cut -d. -f1)
lon_min1=$(($lon2-$lon_deg))
lon_min2=$(($lon_min1*60))
lon_min=$(echo $lon_min2 | cut -d. -f1)
lon_seg1=$(($lon_min2-$lon_min))
lon_seg=$(($lon_seg1*60))
if [ "$lat1" -lt 0 ]; then;
    lat_deg=$((-1*$lat_deg))
fi
if [ "$lon1" -lt 0 ]; then
    lon_deg=$((-1*$lon_deg))
fi

#data inicial
#------------
year=$(echo ${yyyymmdd}  | cut -b 1-4)
month=$(echo ${yyyymmdd} | cut -b 5-6)
day=$(echo ${yyyymmdd}   | cut -b 7-8)


echo "    + READWRFNC - data            = $yyyymmdd - ${year}/${month}/${day}"
echo "    + READWRFNC - localidade      = $nome_estacao"
echo "    + READWRFNC - projeto         = $projeto"
echo "    + READWRFNC - lat,lon         = ($lat1, $lon1)"
#echo '    + lat2,lon2       = '$lat2', '$lon2
echo '    + READWRFNC - lat,lon         = ('$lat_deg'°'$lat_min"'"$lat_seg'", '$lon_deg'°'$lon_min"'"$lon_seg'")'
echo "    + READWRFNC - timezone        = $timezone"
echo "    + READWRFNC - WRFgrade        = d0$WRFgrade"
echo "    + READWRFNC - altura estacao  = $altura_estacao"
echo " "
echo '    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
echo " "


#fazendo tudo desntro no READ_WRF_NC
cd $ReadWRFnc

#criando o link simbolico, e.g., wrfout_d01_2023-09-09_00
rm -f wrfout_d0${WRFgrade}_${year}-${month}*                   #apagando os links antigos caso existam
ln -sf $WRFoutput/wrfout_d0${WRFgrade}_${year}-${month}* .     #criando o link para os anos (1,2,3,4,5,6) diferentes de zero

    
#extraindo as series temporais e de perfis verticais wrfout_d04_2013-04-01
#-------------------------------------------
#OBS:
# geopotencial total               = PH + PHB
# altura geopotencial_m            = (PH + PHB)/9.81
# temperatura potencial total_K    = T + 300
# pressao total_mb                 = (P + PB)/0.01
# caminho dos dados de input (sem a barra, /, no final)
   
#echo "variaveis lidas:"
#echo "-----------------------------------------------"
#echo "VENTO ZONAL                  - U      (m/s    )"
#echo "VENTO MERIDIONAL             - V      (m/s    )"
#echo "VENTO NA VERTICAL            - W      (m/s    )"
#echo "PERTUBACAO DA PRESSAO        - P      (Pa     )"
#echo "PRESSAO DO ESTADO BÁSICO     - PB     (Pa     )"
#echo "PERTUBACAO GEOPOTENCIAL      - PH     (m²/s²  )"
#echo "ESTADO BASE GEOPOTENCIAL     - PHB    (m²/s²  )"
#echo "ALTURA DO TERRRENO           - HGT    (m      )"
#echo "ALTURA DA CLP                - PBLH   (m      )"
#echo "VENTO ZONAL A 10 m           - U10    (m/s    )"
#echo "VENTO MERIDIONAL A 10 m      - v10    (m/s    )"
#echo "PRECIPITACAO TOTAL ACUMULADA - RAINNC (mm     )"
#echo "PRESSAO NA SUPERFICIE        - PSFC   (Pa     )"
#echo "TEMPERATURA A 2 m            - T2     (K      )"
#echo "TEMPERATURA                  - T      (K      )"
#echo "WATER VAPOR MIXING RATIO     - QVAPOR (kg kg-1)"
#echo "RAIN WATER MIXING RATIO      - QRAIN  (kg kg-1)"
#echo "RAIN WATER MIXING RATIO      - QCLOUD (kg kg-1)"
#echo "QV AT 2 m                    - Q2     (kg kg-1)"
#echo "RAdayCAO GLOBAL              - SWDOWN (W/m2   )"
#echo "CLOUD FRACTION               - CLDFRA          "
#echo "-----------------------------------------------"


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#PH and PHB are staggered in the Z direction, meaning that they are at the tops and bottoms of grid cells. 
#  That is fine for the heights of vertical winds, but for horizontal winds, other prognostic variables, 
#  and most diagnostic variables, the center of the cell should be used. You can average the top and bottom 
#  to get the center. For the cell at vertical level 'k,' height above sea level is:

#*(((PH(k)+PH(k+1)) / 2) + ((PHB(k)+(PHB(k+1)) / 2) / 9.81**

#Call me lazy if you like (everyone else does), but WRF already makes part of that calculation and puts it 
#  in an unstaggered variable for the geopotential height at the cell center. That variable is PHP, 
#  so height above sea level for middle of cell(j,i) at level 'k' is PHP(k,j,i)/9.81. In the subroutine 
#  calc_php, it uses this code:

#*php(i,k,j) = 0.5*(phb(i,k,j)+phb(i,k+1,j)+ph(i,k,j)+ph(i,k+1,j))*

#Height in m asl (metres above mean sea level-MAMSL, or simply metres above sea level-MASL or asl) 
#  is (PH+PHB)/9.81. Subtract terrain height HGT to get height in m agl (above ground level).
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#apagando os arquivos antigos caso existam
rm -f *.csv
rm -f *.zip
rm -f *.out 

echo " "
echo '    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
echo " "

echo "    + READWRFNC - criando o arquivo como TIME_SERIES_SURFACE_$d0{WRFgrade}_$year-$month-${day}-$nome_estacao.csv"
./read_wrf_nc.exe wrfout_d0${WRFgrade}_${year}-${month}* -ts ll $lat1 $lon1 ${TIME_SERIES_SURFACE[@]} -lev 1
mv -u TIME_SERIES.out TIME_SERIES_SURFACE_d0${WRFgrade}_$year-$month-${day}-$nome_estacao.csv

#apagando o arquivo caso exista
rm -f TIME_SERIES.out

echo "    + READWRFNC - gravando arquivo como TIME_SERIES_SOUNDING_$d0{WRFgrade}_$year-$month-${day}-$nome_estacao.csv"
./read_wrf_nc.exe wrfout_d0${WRFgrade}_${year}-${month}* -ts ll $lat1 $lon1 ${TIME_SERIES_SOUNDING[@]}
mv -u TIME_SERIES.out TIME_SERIES_SOUNDING_d0${WRFgrade}_$year-$month-${day}-$nome_estacao.csv

echo " "
echo '    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
echo " "

echo " "
echo '    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
echo " "
    
echo "    + READWRFNC - compactando os arquivos como TIME_SERIES_${yyyymmdd}.zip"
#   -----------------------------------------------------------------------------------------------------
rm -f *.zip
/usr/bin/zip TIME_SERIES_${yyyymmdd}-$nome_estacao.zip *.csv

echo "    + READWRFNC - movendo TIME_SERIES_${yyyymmdd}-$nome_estacao.zip para $WRFoutput"

#pegando o aquivo da lista.json de d-namelist
dir_namelist=$dir_local/d-namelist
echo "    + READWRFNC - copiando a $projeto-lista.json como lista.json"
\cp -f $dir_namelist/$projeto-lista.json lista.json

echo "    + READWRFNC - preenchedo a lista.json"
echo "    + READWRFNC - exclui as 10 primeiras linhas da lista.json"
sed -i '1,10d' lista.json
echo "    + READWRFNC - inserindo a primeira linha em branco na lista.json"
sed -i '1s/^/\n/' lista.json
echo "    + READWRFNC - inserindo [{ na primeira linha da lista.json"
sed -i '1s/^/[{/' lista.json 
echo "    + READWRFNC - apagando }] na ultima linha"
sed -i '/}]/d' lista.json
echo "    + READWRFNC - preenchendo a ultima linha"
echo "}," >> lista.json
echo "{" >> lista.json
echo '    "nome":"TIME_SERIES_'${yyyymmdd}-${nome_estacao}'.zip",' >>lista.json
echo '    "ano":"'${year}'",' >>lista.json
echo '    "mes":"'${month}'",' >>lista.json
echo '    "dia":"'${day}'",' >>lista.json
echo '    "local":"'${nome_estacao}'",' >>lista.json
echo '    "lat":"'${lat1}'",' >>lista.json
echo '    "lon":"'${lon1}'"' >>lista.json
echo "}]" >> lista.json

echo "    + READWRFNC - enviando *.${nome_estacao}.zip e lista.json para $WRFoutput"
#---------------------------------------------------------------------------------
\cp -f *${nome_estacao}.zip $WRFoutput
\cp -f lista.json $WRFoutput

#enviando a lista.json para d-namelist
echo "    + READWRFNC - copiando a lista.json para $projeto-lista.json"
\cp -f lista.json $dir_namelist/$projeto-lista.json

echo '    + READWRFNC - voltado ao processa_wrf_00_system.scpt'
echo " "
echo '    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
echo " "

cd $dir_local
