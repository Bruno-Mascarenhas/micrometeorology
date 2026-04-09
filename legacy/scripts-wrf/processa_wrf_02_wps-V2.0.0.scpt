#!/bin/ksh

echo " "
echo '    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
echo "    + WPS    - PREENCHENDO O NAMELIST.WPS E EXECUTANDO O WPS"
echo '    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
echo " "


#definindo o vetor de variaveis para passagem para o processa_wrf_02_wps.scpt
#        0      1       2     3      4      5       6     7      8            9            10           11       12         13        14            15        16             17       18                 19       20
#vetwps=($year1 $month1 $day1 $hour1 $year2 $month2 $day2 $hour2 $num_max_dom $lat_central $lon_central $numproc $WPSoutput1 $DATAgeog $DATAhistory $DATAtype $geog_data_res $dir_wps $interval_seconds  $projeto $wudapt_op)


#path das funcoes
#----------------
export PATH=$PATH:/home/models/WRF/wrf-op/d-functions
export LD_LIBRARY_PATH=/usr/local/wrflibrary/lib:$LD_LIBRARY_PATH


#local de trabalho
#-----------------
dir_local=$( grep dir_local /home/models/WRF/wrf-op/.FUNCTIONS_WRF_environment | awk '{print $2}' )
export PATH=$PATH:${dir_local}:.


#lendo as variaveis passadas pelo SYSTEM
#---------------------------------------
vetwps=("$@") #recebendo o vetor das variaveis
year1=${vetwps[0]}
month1=${vetwps[1]}
day1=${vetwps[2]}
hour1=${vetwps[3]}
year2=${vetwps[4]}
month2=${vetwps[5]}
day2=${vetwps[6]}
hour2=${vetwps[7]}
num_max_dom=${vetwps[8]}
lat_central=${vetwps[9]}
lon_central=${vetwps[10]}
numproc=${vetwps[11]}
WPSoutput=${vetwps[12]}
output_geogrid=$WPSoutput
output_metgrid=$WPSoutput
DATAgeog=${vetwps[13]}
DATAhistory=${vetwps[14]}
DATAtype=${vetwps[15]}
geog_data_res=${vetwps[16]}
dir_wps=${vetwps[17]}
dir_namelist=$dir_local/d-namelist
interval_seconds=${vetwps[18]}
projeto=${vetwps[19]}
wudapt_op=${vetwps[20]}


echo "    + WPS    - recebendo ${#vetwps[*]} variaveis do processa_wrf_00_system.scpt:"
echo "    + WPS    - (projeto)          - a quem se destina os dados             = $projeto"
echo "    + WPS    -                    - data inicial da simulacao              = $year1/$month1/$day1 as $hour1 h"
echo "    + WPS    -                    - data final   da simulacao              = $year2/$month2/$day2 as $hour2 h"
echo "    + WPS    - (num_max_dom)      - numero de maximo de dominios do wrf    = $num_max_dom"
echo "    + WPS    - (lat_central)      - latitude  central do wrf (deg)         = $lat_central"
echo "    + WPS    - (lon_central)      - longitude central do wrf (deg)         = $lon_central"
echo "    + WPS    - (numproc)          - numero de processadores utilizados     = $numproc"
echo "    + WPS    - (output_geogrid)   - local do output geogrid (WPS)          = $output_geogrid"
echo "    + WPS    - (output_metgrid)   - local do output metgrid (WPS)          = $output_metgrid"
echo "    + WPS    - (DATAgeog)         - local dos dados geograficos (WPS)      = $DATAgeog"
echo "    + WPS    - (DATAtype)         - modelo global                          = $DATAtype"
echo "    + WPS    - (DATAhistory)      - local dos dados do modelo GFS          = $DATAhistory"
echo "    + WPS    - (geog_data_res)    - opcao dos dados geograficos (WPS)      = $geog_data_res"
echo "    + WPS    - (dir_wps)          - local do WPS                           = $dir_wps"
echo "    + WPS    - (dir_namelist)     - local do namelist.wps                  = $dir_namelist"
echo "    + WPS    - (interval_seconds) - intervalo de leitura do modelo global  = $interval_seconds s"
echo "    + WPS    - (wudapt_op)        - WUDAPT                                 = $wudapt_op"


echo "    + WPS    - criando as pastas de output do WPS caso nao existam:"
if [ ! -d $output_geogrid ]; then; echo "    + WPS    - criando diretorios $output_geogrid"; \mkdir -p $output_geogrid; else; echo "    + WPS    - diretorio $output_geogrid ja existe"; fi
if [ ! -d $output_metgrid ]; then; echo "    + WPS    - criando diretorios $output_metgrid"; \mkdir -p $output_metgrid; else; echo "    + WPS    - diretorio $output_metgrid ja existe"; fi


echo "    + WPS    - apagando arquivos antigos em $dir_wps"
\rm -f $dir_wps/namelist.wps
\rm -f $dir_wps/GRIBFILE.*
\rm -f $dir_wps/PFILE*
\rm -f $dir_wps/FILE*
\rm -f $dir_wps/SST*
\rm -f $dir_wps/met_em*
\rm -f $dir_wps/geo_em*
\rm -f $dir_wps/metgrid.lo*
\rm -f $dir_wps/geogrid.lo*


echo "    + WPS    - apagando arquivos antigos em $output_geogrid"
\rm -f $DATAhistory/namelist.wps
\rm -f $DATAhistory/met_em*
\rm -f $DATAhistory/geo_em*


echo "    + WPS    - montando o namelist.wps para $projeto"
cd $dir_namelist


echo "    + WPS    - apagando namelist.wps anterior"
\rm -f namelist.wps     
sed "s/xnum_max_domx/$num_max_dom/g" $projeto-namelist.wps-cat > namelist.wps


#igual para todos namelists
#--------------------------
sed -i "s/xyear1x/$year1/g" namelist.wps
sed -i "s/xmonth1x/$month1/g" namelist.wps
sed -i "s/xday1x/$day1/g" namelist.wps
sed -i "s/xhour1x/$hour1/g" namelist.wps
sed -i "s/xyear2x/$year2/g" namelist.wps
sed -i "s/xmonth2x/$month2/g" namelist.wps
sed -i "s/xday2x/$day2/g" namelist.wps
sed -i "s/xhour2x/$hour2/g" namelist.wps
sed -i "s/xinterval_secondsx/$interval_seconds/g" namelist.wps


#importante: no sed com path, mudar os delimitadores
sed -i "s|xoutput_geogridx|$output_geogrid|g" namelist.wps
sed -i "s/xgeog_data_resx/$geog_data_res/g" namelist.wps
sed -i "s/xlat_centralx/$lat_central/g" namelist.wps
sed -i "s/xlon_centralx/$lon_central/g" namelist.wps


#add=2 #2 graus para + e para -
#(( truelat1  = lat_central - add ))
#(( truelat2  = lat_central + add ))
#(( stand_lon = lon_central - add ))


truelat1=$( echo $lat_central | cut -b 1-5)
truelat2=$( echo $lat_central | cut -b 1-5)
stand_lon=$(echo $lon_central | cut -b 1-5)


sed -i "s/xlat_central1x/$truelat1/g" namelist.wps
sed -i "s/xlat_central2x/$truelat2/g" namelist.wps
sed -i "s/xlon_central1x/$stand_lon/g" namelist.wps


#importante: no sed com path, mudar os delimitadores de / para |
sed -i "s|xgeogdatapathx|$DATAgeog|g" namelist.wps
sed -i "s|xoutput_geogridx|$output_geogrid|g" namelist.wps
sed -i "s|xoutput_metgridx|$output_metgrid|g" namelist.wps
sed -i "s/xFILEx/'FILE'/g" namelist.wps
#--------------------------
#ateh aqui


echo "    + WPS    - copiando namelist.wps (GFS) para o WPS"
\cp -f namelist.wps $dir_wps/namelist.wps
\cp -f namelist.wps $output_geogrid/namelist.wps


echo "    + WPS    - saindo de d-namelist e entrado no WPS"
cd $dir_wps


##se wudapt Vtable deve ser modificado

echo "    + WPS    - entrando em d-namelist (opcao WUDAPT)"
if [[ $wudapt_op != 'S' ]]; then

    echo "    + WPS    - ─────────────────────────────────────────────────────────────────────────────────"
    echo "    + WPS    - ERRO na execucao do WUDAPT: corrija o namelist.processa"
    echo "    + WPS    - TROCAR wudapt_op = $wudapt_op para wudapt_op = S"
    echo "    + WPS    - FIM DA SIMULACAO"
    echo "    + WPS    - ─────────────────────────────────────────────────────────────────────────────────"
    echo "    + WPS    - Unsuccessful">$dir_local/d-log/WUDAPT.LOG
    exit 1

else

    echo "    + WPS    - Successful">$dir_local/d-log/WUDAPT.LOG

fi


echo "    + WPS    - ─────────────────────────────────────────────────────────────────────────────────"
echo "    + WPS    - ajustando o geogrid (normal ou wudapt)"
echo "    + WPS    - ─────────────────────────────────────────────────────────────────────────────────"
cd geogrid
if [[ $wudapt_op == 'S' ]]; then

    \ln -sf GEOGRID.TBL.ARW_LCZ GEOGRID.TBL

else

    \ln -sf GEOGRID.TBL.ARW GEOGRID.TBL

fi
#saindo do geogrid
cd ..


echo "    + WPS    - rodando o GEOGRID.EXE"
#mpirun -np $numproc ./geogrid.exe #WPS rodando em paralelo (tem que compilar)
$dir_wps/geogrid.exe                      #WPS rodando em serial


#verificando se o GEOGRID rodou com sucesso, caso contrario, acaba aqui
success=$( grep Successful $dir_wps/geogrid.log | awk '{print $5}' )
echo "    + WPS    - $success">$dir_local/d-log/wps.log


if [[ $success != "Successful" ]]; then

    echo "    + WPS    - WPS (geogrid.exe) nao rodou completamente">>$dir_local/d-log/wps.log
    echo "    + WPS    - Veja wps.log e geogrid.log na pasta d-log"
    echo "    + WPS    - Parou em: ${year1}/${month1}/${day1} -- ${year2}/${month2}/${day2}">>$dir_local/d-log/wps.log
    cp $dir_wps/geogrid.log $dir_local/d-log  #somente um log, para ver o erro da rodada
    exit

fi 


\rm -f Vtable
\ln -sf $dir_wps/ungrib/Variable_Tables/Vtable.GFS Vtable 


echo "    + WPS    - criando o link para ${DATAhistory}/${DATAtype}/"
$dir_wps/link_grib.csh ${DATAhistory}/${DATAtype}/


echo "    + WPS    - rodando o UNGRID.EXE"
#mpirun -np $numproc ./ungrib.exe #WPS rodando em paralelo (tem que compilar)
$dir_wps/ungrib.exe                      #WPS rodando em serial


echo "    + WPS    - rodando o METGRID.EXE"
#mpirun -np $numproc ./metgrid.exe #WPS rodando em paralelo (tem que compilar)
$dir_wps/metgrid.exe                     #WPS rodando em serial


#verificando se o METGRID rodou com sucesso, caso contrario, acaba aqui
success=$( grep Successful $dir_wps/metgrid.log | awk '{print $5}' )
echo "    + $success">>$dir_local/d-log/wps.log


if [[ $success != "Successful" ]]; then

    echo "    + WPS    - WPS (metgrid.exe nao rodou completamente">>$dir_local/d-log/wps.log
    echo "    + WPS    - Veja WPS.LOG e metgrid.log na pasta d-log"
    echo "    + WPS    - Parou em: ${year1}/${month1}/${day1} -- ${year2}/${month2}/${day2}">>$dir_local/d-log/wps.log
    cp $dir_wps/metgrid.log $dir_local/d-log  #somente um log, para ver o erro da rodada
    exit

fi 


echo "    + WPS    - saindo do diretorio do WPS"
cd ..
