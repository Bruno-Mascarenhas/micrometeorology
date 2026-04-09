#!/bin/ksh

echo " "
echo '    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
echo "    + FTP - EXECUTANDO DELETA FILES"
echo '    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
echo " "


#                0
#vetdeletafiles=($yyyymmdd)

#local de trabalho
#-----------------
dir_local=$( grep dir_local /home/models/WRF/wrf-op/.FUNCTIONS_WRF_environment | awk '{print $2}' )
export PATH=$PATH:${dir_local}:.

#lendo as variaveis passadas pelo SYSTEM
#---------------------------------------
vetdeletafiles=("$@") #recebendo o vetor das variaveis

#pegando dados passados por processa_wrf_00.scpt
yyyymmdd=${vetdeletafiles[0]}

cd $dir_local

#para o input e output - guardando os ultimos 30 dias
data_cut=$($dir_local/d-functions/faz_data.func  ${yyyymmdd}00 -720 | cut -b 1-8)
#echo "    + DELETA - data_cut = $data_cut"
echo "    + DELETA - remocao dos arquivos em d-input anteriores a $data_cut"

ls  ${dir_local}/d-input > arc

for name in $(cat arc); do

    #echo "    + DELETA - $name"
    if [[ -n "$name" ]] && [[ $name -le $data_cut ]] ; then

        echo '    + DELETA - vou remover = '${dir_local}/d-input/$name
        \rm -rf ${dir_local}/d-input/$name

    #else

    #    echo '    + DELETA - nao vou remover = '${dir_local}/d-input/$name

    fi

done

\rm -f ${dir_local}/arc
#echo "    + DELETA - data_cut = $data_cut"
echo "    + DELETA - remocao dos arquivos em d-output anteriores a $data_cut"

ls  ${dir_local}/d-output > arc
#retirar a pasta figuras, caso contrario, serao apagadas
sed -i "s/figuras/ /g" arc

for name in $(cat arc); do
    #echo "    + DELETA - $name"

    if [[ -n "$name" ]] && [[ $name -le $data_cut ]] ; then

        echo '    + DELETA - vou remover = '${dir_local}/d-output/$name
        \rm -rf ${dir_local}/d-output/$name

    #else

    #    echo '    + DELETA - nao vou remover = '${dir_local}/d-output/$name

    fi

done

# echo "    + DELETA - data_cut = $data_cut"
rm -f ${dir_local}/arc


#para o log - guardando os ultimos 3 dias
data_cut=$($dir_local/d-functions/faz_data.func  ${yyyymmdd}00 -72 | cut -b 1-8)
#echo "    + DELETA - data_cut = $data_cut"
echo "    + DELETA - remocao dos arquivos em d-log anteriores a $data_cut"

ls  ${dir_local}/d-log > arc
#retirar as palavra -wrf-op.debug e nome dos arquivos que não serão deletados
sed -i "s/geogrid.log/ /g" arc
sed -i "s/metgrid.log/ /g" arc
sed -i "s/rsl.error.0000/ /g" arc
sed -i "s/wps.log/ /g" arc
sed -i "s/wrf.log/ /g" arc
sed -i "s/wrf-op.log/ /g" arc
sed -i "s/wrf-case.debug/ /g" arc
sed -i "s/-wrf-op.debug/ /g" arc
sed -i "s/wrf-op.debug/ /g" arc
sed -i "s/WUDAPT.LOG/ /g" arc





 

 



for name in $(cat arc); do
    #echo "    + DELETA - $name"
    echo "$name, $data_cut"

    if [[ -n "$name" ]] && [[ $name -le $data_cut ]] ; then

        echo "    + DELETA - vou remover = ${dir_local}/d-log/${name}-wrf-op.debug"
        \rm -rf ${dir_local}/d-log/${name}

    #else

    #    echo '    + DELETA - nao vou remover = '${dir_local}/d-output/$name

    fi

done

# echo "    + DELETA - data_cut = $data_cut"
rm -f ${dir_local}/arc

echo '    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
echo "    + FTP - FIM DA REMOCAO"
echo '    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
