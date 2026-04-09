#!/bin/ksh

echo " "
echo '    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
echo "    + BUSCA - EXECUTANDO BUSCA"
echo '    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
echo " "

#path das funcoes
#----------------
export PATH=$PATH:/home/models/WRF/wrf-op/d-functions

#          0         1         2
#vetbusca=($yyyymmdd $hour_end $DATAtype)
vetbusca=("$@") #recebendo o vetor das variaveis

#pegando dados passados por processa_wrf_00.scpt
yyyymmdd=${vetbusca[0]}
hour_end=${vetbusca[1]}
DATAtype=${vetbusca[2]}

#local de trabalho
#-----------------
dir_local=$( grep dir_local /home/models/WRF/wrf-op/.FUNCTIONS_WRF_environment | awk '{print $2}' )
export PATH=$PATH:${dir_local}:.

echo "    + BUSCA - inicio da busca de dados ateh $hour_end horas"
    
#criando o diretorio dentro do d-input onde guardo os arquivos da noaa
#---------------------------------------------------------------------
dir_input=${dir_local}/d-input/${yyyymmdd}/$DATAtype
mkdir -p ${dir_input}
echo "    + BUSCA - baixando os arquivos para $dir_input"
cd ${dir_input}

#extraindo ano e mes da data inicial
year1=${yyyymmdd:0:4}
month1=${yyyymmdd:4:2}
day1=${yyyymmdd:6:2}

#echo $yyyymmdd ${dir_input}
#horas do 1o. e ultimo arquivo
#-----------------------------
hour=0

while [[ $hour -le $hour_end ]]; do

    if [[ $hour -lt 10 ]]; then

        hour1=00${hour}

    else

        if [[ $hour -lt 100 ]]; then

            hour1=0${hour}

        else

            hour1=${hour}

        fi

    fi
    
    (( hour  = hour + 03 ))

    #local onde vai buscar os arquivos no noaa ou ucar (de 3 em 3 horas)
    #-------------------------------------------------------------------
    if [[ ${DATAtype} = "gfs-0.50" ]] ; then

        echo "    + BUSCA - buscando gfs.t00z.pgrb2.1p00.f${hour1} com 0.50 graus"
        echo '      ─────────────────────────────────────────────────────────────'
        #wget -c https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.20250522/00/atmos/gfs.t00z.pgrb2.1p00
        #wget -c https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.${yyyymmdd}/00/atmos/gfs.t00z.pgrb2.1p00.f${hour1}
        #wget -c https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.${yyyymmdd}/00/atmos/gfs.t00z.pgrb2.0p25.f${hour1}

        #wget -c ftp://ftpprd.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.20250522/00/atmos/gfs.t00z.pgrb2.1p00.f000
        #wget -c ftp://ftpprd.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.${yyyymmdd}/00/atmos/gfs.t00z.pgrb2.1p00.f${hour1}

        wget -c https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.${yyyymmdd}/00/atmos/gfs.t00z.pgrb2.1p00.f${hour1}   #outro link do download (alternativo)
        #    wget -c https://ftp.ncep.noaa.gov/data/nccf/com/gfs/prod/gfs.${yyyymmdd}/00/atmos/gfs.t00z.pgrb2.1p00.f${hour1}
        #    https://ftp.ncep.noaa.gov/data/nccf/com/gfs/prod/gfs.20210323/00/atmos/gfs.t00z.pgrb2.1p00.f000

    fi

    if [[ ${DATAtype} = "gfs-0.25" ]] ; then

        echo "    + BUSCA - buscando gfs.0p25.${yyyymmdd}00.f${hour1}.grib2 com 0.25 graus"
        echo '      ─────────────────────────────────────────────────────────────'
        wget -c https://data.rda.ucar.edu/ds084.1/${year1}/${yyyymmdd}/gfs.0p25.${yyyymmdd}00.f${hour1}.grib2

    fi

done

#local onde vai buscar os arquivos no noaa ou ucar (de 6 em 6 horas)
#-------------------------------------------------------------------

if [[ ${DATAtype} = "fnl" || ${DATAtype} = "gdas" ]] ; then


    data_inicial=$yyyymmdd
    hour_end=$((hour_end + 6))
    intervalo=6


    #validar entrada
    if ! [[ $data_inicial =~ ^[0-9]{8}$ ]]; then
        echo "Erro: data_inicial deve estar no formato YYYYMMDD"
        exit 1
    fi

    if ! [[ $hour_end =~ ^[0-9]+$ ]]; then
        echo "Erro: hour_end deve ser um número inteiro"
        exit 1
    fi

    if ! [[ $intervalo =~ ^[0-9]+$ ]]; then
        echo "Erro: intervalo_horas deve ser um número inteiro"
        exit 1
    fi

    #extrair componentes da data inicial
    ano=${yyyymmdd:0:4}
    mes=${yyyymmdd:4:2}
    dia=${yyyymmdd:6:2}

    #funcao para validar se eh ano bissexto
    eh_bissexto() {
        ano_check=$1
        if (( (ano_check % 4 == 0 && ano_check % 100 != 0) || (ano_check % 400 == 0) )); then
            return 0
        else
            return 1
        fi
    }

    #funcao para retornar dias do mes
    dias_do_mes() {
        #normaliza a entrada para evitar problemas com numeros com zero a esquerda
        #(ex: "01" sendo interpretado como octal quando usado em aritmetica).
        mes_check=$((10#$1))
        ano_check=$2

        case $mes_check in
            1|3|5|7|8|10|12) echo 31 ;;
            4|6|9|11) echo 30 ;;
            2)
                if eh_bissexto $ano_check; then
                    echo 29
                else
                    echo 28
                fi
                ;;
            *)
                echo 0
                ;;
        esac
    }

    #inicializar variaveis
    hora_acumulada=0
    dia_atual=$dia
    mes_atual=$mes
    ano_atual=$ano

    #gerar sequencia
    while (( $hora_acumulada < $hour_end )); do

        #calcular hora local (0-23)
        hora_local=$((hora_acumulada % 24))

        #calcular quantos dias se passaram desde o inicio
        dias_passados=$((hora_acumulada / 24))

        #calcular dia, mes e ano apos adicionar dias_passados
        dia_temp=$((dia + dias_passados))
        mes_temp=$mes
        ano_temp=$ano

        #ajustar mes e ano para o dia_temp
        while (( $dia_temp > 0 )); do

            #garantir que mes_temp esta no intervalo 1-12
            while (( $mes_temp > 12 )); do

                mes_temp=$((mes_temp - 12))
                (( ano_temp++ ))

            done

            while (( $mes_temp < 1 )); do

                mes_temp=$((mes_temp + 12))
                (( $ano_temp-- ))

            done

            dias_max=$(dias_do_mes $mes_temp $ano_temp)

            if (( $dia_temp <= $dias_max )); then

                break

            fi

            dia_temp=$((dia_temp - dias_max))
            (( $mes_temp++ ))

        done

        #formatar e exibir data/hora
        #------------
        year=$(printf  "%04d" $ano_temp)
        month=$(printf "%02d" $mes_temp)
        day=$(printf   "%02d" $dia_temp)
        hour=$(printf  "%02d" $hora_local)

        if [[ ${DATAtype} = "fnl" ]] ; then

            #                            /home/models/WRF/wrf-data/fnl/2024 /fnl_2024   01      01    _00    _00.grib2 
            echo "    + BUSCA - copiando /home/models/WRF/wrf-data/fnl/$year/fnl_${year}${month}${day}_${hour}_00.grib2"
            echo '      ─────────────────────────────────────────────────────────────'
            cp /home/models/WRF/wrf-data/fnl/$year/fnl_${year}${month}${day}_${hour}_00.grib2 .

        fi

        if [[ ${DATAtype} = "gdas" ]] ; then

            #                            /home/models/WRF/wrf-data/gdas/2024 /gdas1.fnl0p25.2024   12      30    12     .f00.grib2
            echo "    + BUSCA - copiando /home/models/WRF/wrf-data/gdas/$year/gdas1.fnl0p25.${year}${month}${day}${hour}.f00.grib2"
            echo '      ─────────────────────────────────────────────────────────────'
            cp /home/models/WRF/wrf-data/gdas/$year/gdas1.fnl0p25.${year}${month}${day}${hour}.f00.grib2 .

        fi

        #incrementar hora acumulada pelo intervalo
        (( hora_acumulada += $intervalo ))

    done

fi

echo " "
echo '    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
echo "    + BUSCA - fim da busca de dados"
echo '    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
echo " "

#voltando ao local de trabalho
#
cd ${dir_local}
