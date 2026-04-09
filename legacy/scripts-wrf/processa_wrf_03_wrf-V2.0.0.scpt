#!/bin/ksh

echo " "
echo '    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
echo "    + WRF    - EXECUTANDO O WRF"
echo '    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
echo " "

#        0      1       2     3      4      5       6     7      8            9        10         11          12       13       14
#vetwrf=($year1 $month1 $day1 $hour1 $year2 $month2 $day2 $hour2 $num_max_dom $numproc $time_step $WPSoutput1 $dir_wrf              $projeto $wudapt_op)
#        2026   01      10    00     2026   01      10    12     1            36       160        /home/models/WRF/wrf-op/WRF-4.7.1 LEAL     S

#local de trabalho
#-----------------
dir_local=$( grep dir_local /home/models/WRF/wrf-op/.FUNCTIONS_WRF_environment | awk '{print $2}' )
export PATH=$PATH:${dir_local}:.


#lendo as variaveis passadas pelo SYSTEM
#---------------------------------------
vetwrf=("$@") #recebendo o vetor das variaveis


year1=${vetwrf[0]}
month1=${vetwrf[1]}
day1=${vetwrf[2]}
hour1=${vetwrf[3]}
year2=${vetwrf[4]}
month2=${vetwrf[5]}
day2=${vetwrf[6]}
hour2=${vetwrf[7]}
num_max_dom=${vetwrf[8]}
numproc=${vetwrf[9]}
time_step=${vetwrf[10]}
WPSoutput=${vetwrf[11]}
dir_wrf=${vetwrf[12]}
projeto=${vetwrf[13]}
wudapt_op=${vetwrf[14]}


dir_namelist=$dir_local/d-namelist


#local do mpich
dir_mpich=$( grep dir_mpich /home/models/WRF/wrf-op/.FUNCTIONS_WRF_environment | awk '{print $2}' )


#local do netcdf
dir_netcdf=$( grep dir_netcdf /home/models/WRF/wrf-op/.FUNCTIONS_WRF_environment | awk '{print $2}' )


#path das funcoes
export PATH=$PATH:/home/models/WRF/wrf-op/d-functions
export NETCDF=$NETCDF:$dir_netcdf
export PATH=$PATH:$dir_netcdf/lib


#local de trabalho
dir_local=$( grep dir_local /home/models/WRF/wrf-op/.FUNCTIONS_WRF_environment | awk '{print $2}' )
export PATH=$PATH:${dir_local}:.


echo "    + WRF    - recebendo ${#vetwrf[*]} variaveis do processa_wrf_00_system.scpt:"
echo "    + WRF    - (projeto)          - a quem se destina os dados            = $projeto"
echo "    + WRF    - (year1/month1/day1)- data inicial da simulacao             = $year1/$month1/$day1 as $hour1 h"
echo "    + WRF    - (year2/month2/day2)- data final   da simulacao             = $year2/$month2/$day2 as $hour2 h"
echo "    + WRF    - (num_max_dom)      - numero de maximo de dominios do wrf   = $num_max_dom"
echo "    + WRF    - (numproc)          - numero de processadores utilizados    = $numproc"
echo "    + WRF    - (time_step)        - passo no tempo do WRF                 = $time_step"
echo "    + WRF    - (WPSoutput)        - local do output do WPS                = $WPSoutput"
echo "    + WRF    - (dir_wrf)          - local do WRF                          = $dir_wrf"
echo "    + WRF    - (dir_namelist)     - local do namelist.WRF                 = $dir_namelist"
echo "    + WRF    - (dir_mpich)        - local do mpich                        = $dir_mpich"
echo "    + WRF    - (dir_netcdf)       - local do netcdf                       = $dir_netcdf"
echo "    + WRF    - (wudapt_op)        - opcao wudapt                          ? $wudapt_op"


echo "    + WRF    - (dir_wrf)          - apagando arquivos antigos em          = $dir_wrf/test/em_real"
\rm -f $dir_wrf/test/em_real/namelist.input
\rm -f $dir_wrf/test/em_real/met_em.d0*
\rm -f $dir_wrf/test/em_real/rsl.error.*
\rm -f $dir_wrf/test/em_real/rsl.out.*
\rm -f $dir_wrf/test/em_real/wrfinput_d0*
\rm -f $dir_wrf/test/em_real/wrfbdy_d*
\rm -f $dir_wrf/test/em_real/wrfrst_d*
\rm -f $dir_wrf/test/em_real/wrfout_d*


#entrado em WPSoutput para pegar variaveis em comum
cd $WPSoutput
parentid=`cat namelist.wps | grep parent_id`
parentgridratio=`cat namelist.wps | grep parent_grid_ratio`
iparentstart=`cat namelist.wps | grep i_parent_start`
jparentstart=`cat namelist.wps | grep j_parent_start`
intervalseconds=`cat namelist.wps | grep interval_seconds`
ewe=`cat namelist.wps | grep e_we`
esn=`cat namelist.wps | grep e_sn`
#pegando num_metgrid_levels do primero arquivo met_em* gerado pelo WPS
nummetgridlev=`ncdump -c "met_em.d01.${year1}-${month1}-${day1}_${hour1}:00:00.nc" |grep "num_metgrid_levels =" | awk '{print $3}'`
num_land_cat=`ncdump -c "met_em.d01.${year1}-${month1}-${day1}_${hour1}:00:00.nc" |grep "NUM_LAND_CAT" | awk '{print $3}'`


echo "    + WRF    - (parentid)         - namelist.wps:                         =$parentid"
echo "    + WRF    - (parentgridratio)  - namelist.wps:                         =$parentgridratio"
echo "    + WRF    - (iparentstart)     - namelist.wps:                         =$iparentstart"
echo "    + WRF    - (jparentstart)     - namelist.wps:                         =$jparentstart"
echo "    + WRF    - (intervalseconds)  - namelist.wps:                         =$intervalseconds"
echo "    + WRF    - (esn)              - namelist.wps:                         =$esn"
echo "    + WRF    - (ewe)              - namelist.wps:                         =$ewe"
echo "    + WRF    - (nummetgridlev)    - met_em*: numero de niveis             = $nummetgridlev"


#entrado no namelist
cd $dir_namelist


echo "    + WRF    - preenchendo o namelist.input em                            = $dir_namelist"
sed "s/xyear1x/$year1/g" $projeto-namelist.input-cat > namelist.input


#igual para todos namelists.input
#--------------------------------
echo "    + WRF    - completando &time_control em                               = namelist.input"
sed -i "s/xmonth1x/$month1/g" namelist.input
sed -i "s/xday1x/$day1/g" namelist.input
sed -i "s/xhour1x/$hour1/g" namelist.input
sed -i "s/xinterval_secondsx/$intervalseconds/g" namelist.input
sed -i "s/xyear2x/$year2/g" namelist.input
sed -i "s/xmonth2x/$month2/g" namelist.input
sed -i "s/xday2x/$day2/g" namelist.input
sed -i "s/xhour2x/$hour2/g" namelist.input
sed -i "s/xflagx/.false./g" namelist.input   #sem restart
sed -i "s/xtime_stepx/$time_step/g" namelist.input
sed -i "s/xmaxdomx/$num_max_dom/g" namelist.input
sed -i "s/xi_parent_startx/$iparentstart/g" namelist.input
sed -i "s/xj_parent_startx/$jparentstart/g" namelist.input
sed -i "s/xe_wex/$ewe/g" namelist.input
sed -i "s/xe_snx/$esn/g" namelist.input
sed -i "s/xparent_idx/$parentid/g" namelist.input
sed -i "s/xparent_grid_ratiox/$parentgridratio/g" namelist.input


echo "    + WRF    - completando &domains em                                    = namelist.input"
sed -i "s/xnum_metgrid_levx/$nummetgridlev/g" namelist.input


echo "    + WRF    - completando &physics em                                    = namelist.input"
sed -i "s/xnum_land_catx/$num_land_cat/g" namelist.input

if [[ $wudapt_op == 'S' ]]; then

    wudapt_op1=1
    sed -i "s/xwudapt_opx/$wudapt_op1/g" namelist.input

else

    wudapt_op1=0
    sed -i "s/xwudapt_opx/$wudapt_op1/g" namelist.input

fi
#--------------------------------
#ateh aqui

echo "    + WRF    - copiando namelist.input para o                             = WRF"
\cp -f namelist.input $dir_wrf/test/em_real/namelist.input


echo "    + WRF    - saindo de d-namelist e entrando no                         = WRF"
cd $dir_wrf/test/em_real


echo "    + WRF    - criando o link dos dados do                                = WPS"
\ln -sf $WPSoutput/met_em.d0* .


echo "    + WRF    - executa o                                                  = REAL.EXE"
$dir_mpich/mpirun -np $numproc ./real.exe


#verificando se o REAL rodou com sucesso, caso contrario, acaba aqui
success=$( grep SUCCESS $dir_wrf/test/em_real/rsl.error.0000 | awk '{print $4}' )
echo "    + WRF    - real.exe: $success">$dir_local/d-log/wrf.log

if [[ $success != "SUCCESS" ]]; then

    echo "    + WRF    - REAL nao rodou completamente">$dir_local/d-log/wrf.log
    echo "    + WRF    - ver wrf.log e rsl.error.0000 na pasta d-log">>$dir_local/d-log/wrf.log
    echo "    + WRF    - $success">>$dir_local/d-log/wrf.log
    echo "    + WRF    - parou em: ${year1}/${month1}/${day1} -- ${year2}/${month2}/${day2}">>$dir_local/d-log/wrf.log
    cp rsl.error.0000 $dir_local/d-log  #somente um log, para ver o erro da rodada
    exit

fi
success="NO"

echo "    + WRF    - executa o                                                  = WRF.EXE"
$dir_mpich/mpirun -np $numproc ./wrf.exe

#verificando se o WRF rodou com sucesso, caso contrario, acaba aqui
success=$( grep SUCCESS $dir_wrf/test/em_real/rsl.error.0000 | awk '{print $4}' )
echo "    + WRF    - wrf.exe:  $success">>$dir_local/d-log/wrf.log

if [[ $success != "SUCCESS" ]]; then

    echo "    + WRF    - WRF nao rodou completamente">$dir_local/d-log/wrf.log
    echo "    + WRF    - ver wrf.log e rsl.error.0000 na pasta d-log"
    echo "    + WRF    - parou em: ${year1}/${month1}/${day1} -- ${year2}/${month2}/${day2}">>$dir_local/d-log/wrf.log
    cp rsl.error.0000 $dir_local/d-log  #somente um log, para ver o erro da rodada
    exit

fi


echo "    + WRF    - saindo do                                                  = WRF"
cd $dir_local 
