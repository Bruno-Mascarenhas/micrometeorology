#!/bin/bash
####ksh

echo " "
echo '    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
echo "    + MMIF - EXECUTANDO MMIF"
echo '    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
echo " "


#local de trabalho
#-----------------
dir_local=$( grep dir_local /home/models/WRF/wrf-op/.FUNCTIONS_WRF_environment | awk '{print $2}' )
dir_python=$( grep dir_python /home/models/WRF/wrf-op/.FUNCTIONS_WRF_environment | awk '{print $2}' )
export PATH=$PATH:${dir_local}:.


#         0                 1            2           3         4          5             6     7     8         9        10
#vetmmif=($processa_wrf_wap $VERSAO_MMIF $WRFoutput1 $yyyymmdd $deltaHour $nome_estacao $lat1 $lon1 $timezone $TSgrade $MMIF $projeto)


#lendo as variaveis passadas pelo SYSTEM
#---------------------------------------
vetmmif=("$@") #recebendo o vetor das variaveis

#pegando dados passados por processa_wrf_system
processa_wrf_wap=${vetmmif[0]}
VERSAO_MMIF=${vetmmif[1]}
WRFoutput=${vetmmif[2]}
yyyymmdd=${vetmmif[3]}
deltaHour=${vetmmif[4]}
nome_estacao=${vetmmif[5]}
lat1=${vetmmif[6]}
lon1=${vetmmif[7]}
timezone=${vetmmif[8]}
TSgrade=${vetmmif[9]}
projeto=${vetmmif[10]}

#data inicial
#------------
year1=$(echo ${yyyymmdd}  | cut -b 1-4)
month1=$(echo ${yyyymmdd} | cut -b 5-6)
day1=$(echo ${yyyymmdd}   | cut -b 7-8)

#data final
#----------
tmpTime=`$dir_local/d-fortran/calcTime/calcTime $year1$month1$day1'000000' $deltaHour`
year2=`expr substr $tmpTime 1 4`
month2=`expr substr $tmpTime 5 2`
day2=`expr substr $tmpTime 7 2`
hour2=`expr substr $tmpTime 9 2`

echo "    + MMIF - recebendo ${#vetmmif[*]} variaveis do processa_wrf_00_system.scpt:">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
echo "    + MMIF - projeto      = $projeto">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
echo "    + MMIF - versao wap   = $processa_wrf_wap">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
echo "    + MMIF - versao MMIF  = $VERSAO_MMIF">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
echo "    + MMIF - dir_local    = $dir_local">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
echo "    + MMIF - WRFoutput    = $WRFoutput">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
echo "    + MMIF - localidade   = $nome_estacao">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
echo "    + MMIF - lat,lon      = ("$lat1"," $lon1")">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
echo "    + MMIF - timezone     = $timezone">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
echo "    + MMIF - TSgrade      = d0$TSgrade">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
echo "    + MMIF - data inicial = $day1/$month1/$year1 00:00:00">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
echo "    + MMIF - data final   = $day2/$month2/$year2 00:00:00">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
#---------------------------------------------------------------

cd $dir_local/MMIF
echo "    + MMIF - apagando link anterior e arquivos velhos (se tiver)"
#----------------------------------------------------------------------
\rm -f *zip
\rm -f wrfout*
\rm -f aermet*
\rm -f AERMET*
\rm -f aermod*
\rm -f AERMOD*
\rm -f calmet*
\rm -f terrain*
\rm -f domain*
\rm -f points*
\rm -f qaplot*
\rm -f *.png
\rm -f *.xlsx

# ---------------------------------------
# primeiro caso: aermod SIM e calpuff NAO
# ---------------------------------------


echo "    + MMIF - gerando o arquivo mmif.inp para o AERMOD"
#   ------------------------------------------------------------

cat << EOF > $dir_local/MMIF'/mmif.inp'
Start  ${year1}-${month1}-${day1}_00:00:00
Stop   ${year2}-${month2}-${day2}_00:00:00
TimeZone   $timezone
POINT  latlon $lat1 $lon1 $timezone! in GMT-3 timezone

#layers top 20 40 60 80 100 120 140 160 180 200 250 300 350 400 450 500 600 700 800 900 1000 1300 1600 1900 2200 2500 2800 3100 3400 3700 4000
layers top  20 40 100 200 350 500 750 1000 2000 3000 4000 5000

aer_mixht  AERMET      ! default
aer_min_mixht 1.0      ! default (same as AERMET)
aer_min_obuk  1.0      ! default (same as AERMET)
aer_min_speed 0.0      ! default (following Apr 2018 MMIF Guidance)
aer_use_TSKC  F        ! default (using TSKC is an ALPHA option)
aer_use_NEW   F        ! default (set to T for AERMET 21112 and later versions)

FSL_INTERVAL      6        ! output every 6 hours, not 12 (the default)

point  LL $lat1 $lon1
output aermet bat     aermet_${nome_estacao}.bat
output aermet Useful  aermet_${nome_estacao}.info
output aermet onsite  aermet_${nome_estacao}.dat
output aermet fsl     aermet_${nome_estacao}.fsl
output aermet aersfc  aermet_${nome_estacao}.aersfc.dat
output aermod useful  aermod_${nome_estacao}.info
output aermod sfc     aermod_${nome_estacao}.sfc
output aermod pfl     aermod_${nome_estacao}.pfl

input wrfout_d0${TSgrade}_${year1}-${month1}-${day1}_00_00_00
EOF

echo "    + MMIF - arquivo mmif.inp para o AERMOD gerado com sucesso"
#   ---------------------------------------------------------------------

echo "    + MMIF - criando o link de $WRFoutput"
echo "    + MMIF - link em      = $WRFoutput/wrfout_d0${TSgrade}_${year1}-${month1}-${day1}_00:00:00">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
\ln -sf $WRFoutput/wrfout_d0${TSgrade}_${year1}-${month1}-${day1}_00:00:00 wrfout_d0${TSgrade}_${year1}-${month1}-${day1}_00_00_00


echo "    + MMIF - executando o MMIF"
#para gerar exemplos do arquivo *.inp digite: ./MMIF-VERSAO.exe --sample
./${VERSAO_MMIF}.exe mmif.inp>mmif.out


echo "    + MMIF - apaga o link de $WRFoutput"
\rm -f wrfout*


#gerando as figuras para o aermod, somente se MMIFA = S (sim)
echo "    + MMIF - gerando as figuras para o AERMOD"
echo "    + MMIF - criando o arquivo namelist.py"

cat << EOF > "$dir_local/d-python/wap-aermod/namelist.py"
#arquivo contendo todas as configuracoes do codigo.
#cada item possui as opcoes possiveis listadas em comentarios proximos
#mais detalhes podem ser encontrados na documentacao, presente no repositorio.

config = {
    'mode' : 'script',
    #'mode' : 'cli',

    # 'cli' :
        # Recomendado para testes ou para geracao ocasional de dados
        # Habilita interface com usuario
        # Exige entrada manual de opcoes e algumas configuracoes
        # Escolhe arquivos, janelas de tempo e outras configuracoes manualmente
        # Aviso: Configuracoes manuais tem prioridade sobre as configuracoes do namelist
    # 'script' :
        # Recomendado para acionamento completamente automatico dos scripts
        # Desabilita interface com usuario
        # Nao exige entrada manual de janela de tempo ou de outras configuracoes

    'enable_time_window' : 'no'
    # 'no':
        # Desabilita janela de tempo
    # 'yes':
        # Habilita janela de tempo
        # A janela e definida em time_window
}

time_window = {
    'start' : '${year1}-${month1}-${day1}',
    'end'   : '${year2}-${month2}-${day2}'
    #'start' : '2000-01-01',
    #'end'   : '2030-01-01'
    # Janela de tempo. Deve ser definida no formato AAAA-MM-DD
    # Time Window. Must de defined on the format YYYY-MM-DD
}

output_paths = {
    'fig' : '/home/models/WRF/wrf-op/MMIF/',
    'database' : '/home/models/WRF/wrf-op/MMIF/',
    'maps' : '/home/models/WRF/wrf-op/MMIF/',
    'gifs' : '/home/models/WRF/wrf-op/MMIF/'
}

input_paths = {
    'database' : '/home/models/WRF/wrf-op/MMIF/',
    'aermod_path' : '/home/models/WRF/wrf-op/MMIF/'
}


# Temperatura deve ser inserida em Kelvin
# Temp_Kelvin = Temp_ºC + 273.15

filtros = {
    'temp_max' : '313.15',
    'temp_min' : '288.15',
    'ur_max'   : '100',
    'ur_min'   : '20',
    'patm_max' : '1020',
    'patm_min' : '970',
    'Ws_max'  : '10',           #O filtro de velocidade nao e aplicado no grafico de distribuicao do vento (weibull)
    'Ws_min'  : '0'   ,         #

    'windrose_max_speed' : '10' # Maxima velocidade exibida no grafico do windrose
}
EOF
echo "    + MMIF - executando $processa_wrf_wap"
#local do script python se localiza para gerar as figuras do wrf
dir_script=$dir_local/d-python/wap-aermod
eval "$($dir_python/conda shell.bash hook)"
conda activate venv-op
$dir_python/conda env list

echo "$dir_script/$processa_wrf_wap aermod_${nome_estacao}.sfc"

echo "    + MMIF   - python3 $dir_script/$processa_wrf_wap aermod_${nome_estacao}.sfc">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
python3 $dir_script/$processa_wrf_wap aermod_${nome_estacao}.sfc
conda deactivate


echo "    + MMIF - renomeia os arquivos gerados pelo MMIF de maiusculo para minusculo"

for foo in * ; do

    if ! [ -a "`echo $foo | tr /[A-Z]/ /[a-z]/`" ]; then

        mv "$foo" "`echo $foo | tr /[A-Z]/ /[a-z]/`"
        #altera o conteudo de foo, se for um diretorio, segunda parte faz dentro dele
        foo=`echo $foo | tr /[A-Z]/ /[a-z]/`

    fi

done

#echo "    + MMIF - copianado os arquivos sfc e pfl para o aermod"
#----------------------------------------------------------------
#cp *.sfc $dir_local/AERMOD/$nome_estacao
#cp *.pfl $dir_local/AERMOD/$nome_estacao

echo "    + MMIF - transformando os arquivos para o windows"
/usr/bin/unix2dos *dat
/usr/bin/unix2dos *bat
/usr/bin/unix2dos *fsl
/usr/bin/unix2dos *in1
/usr/bin/unix2dos *in2
/usr/bin/unix2dos *in3
/usr/bin/unix2dos *info
/usr/bin/unix2dos *pfl
/usr/bin/unix2dos *sfc

echo "    + MMIF - compactando dos arquivos como MMIF.AERMOD.${yyyymmdd}.${nome_estacao}.zip"
echo "    + MMIF - compactando dos arquivos como MMIF.AERMOD.${yyyymmdd}.${nome_estacao}.zip">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
#   ---------------------------------------------------------------------------------------------
#/usr/bin/zip MMIF.AERMOD.${yyyymmdd}.${nome_estacao}.zip aerm* windrose.png
/usr/bin/zip MMIF.AERMOD.${yyyymmdd}.${nome_estacao}.zip aerm*


#pegando o aquivo da lista.json de d-namelist
dir_namelist=$dir_local/d-namelist
\cp -f $dir_namelist/$projeto-lista.json lista.json

echo "    + MMIF - copiando a $dir_namelist/$projeto-lista.json como lista.json">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
\cp -f $dir_namelist/$projeto-lista.json lista.json

echo "    + MMIF - preenchedo a lista.json"
echo "    + MMIF - exclui as 10 primeiras linhas da lista.json"
sed -i '1,10d' lista.json
echo "    + MMIF - inserindo a primeira linha em branco na lista.json"
sed -i '1s/^/\n/' lista.json
echo "    + MMIF - inserindo [{ na primeira linha da lista.json"
sed -i '1s/^/[{/' lista.json 
echo "    + MMIF - apagando }] na ultima linha"
sed -i '/}]/d' lista.json
echo "    + MMIF - preenchendo a ultima linha"
echo "}," >> lista.json
echo "{" >> lista.json
echo '    "nome":"MMIF.AERMOD.'${yyyymmdd}.${nome_estacao}'.zip",' >>lista.json
echo '    "ano":"'${year1}'",' >>lista.json
echo '    "mes":"'${month1}'",' >>lista.json
echo '    "dia":"'${day1}'",' >>lista.json
echo '    "local":"'${nome_estacao}'",' >>lista.json
echo '    "lat":"'${lat1}'",' >>lista.json
echo '    "lon":"'${lon1}'"' >>lista.json
echo "}]" >> lista.json

echo "    + MMIF - enviando *.${nome_estacao}.zip e lista.json para $WRFoutput"
#---------------------------------------------------------------------------------
\cp -f *.${nome_estacao}.zip $WRFoutput
\cp -f lista.json $WRFoutput

#enviando a lista.json para d-namelist
echo "    + MMIF - copiando a $dir_namelist/lista.json para $projeto-lista.json">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
\cp -f lista.json $dir_namelist/$projeto-lista.json

#apagando link dos arquivos gerados
#\rm -f *zip
#\rm -f wrfout*
#\rm -f aermet*
#\rm -f AERMET*
#\rm -f aermod*
#\rm -f AERMOD*
#\rm -f calmet*
#\rm -f terrain*
#\rm -f domain*
#\rm -f points*
#\rm -f qaplot*

cd $dir_local
