#!/bin/ksh

#         ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
#         │      lista de modificacoes:                                                                                                       │
#         │      1) 05/01/2018 - codigo inicial - Maxsuel M R Pereira - maxsuel.pereira@ufes.br                                               │
#         │      2) 10/01/2023 - insercao de varias rodadas do wrf                                                                            │
#         │      3) 14/01/2023 - insercao de diferentes opcoes de extacao de dados do mmif                                                    │
#         │      4) 17/01/2023 - insercao de diferentes opcoes de extacao de dados do calwrf                                                  │
#         │      5) 30/01/2023 - integracao de todos os sistemas wrf                                                                          │
#         │      6) 12/09/2023 - integracao para gerar series temporais                                                                       │
#         │      7) 13/09/2023 - criacao do link para integracao dados da plataforma                                                          │
#         │      8) 10/08/2024 - cria o sistema de versoes dos scripts X.Y.Z, onde:                                                           │
#         │                          - X - eh o numero da nova versao, nao compativel com a anterior                                          │
#         │                          - Y - eh o numero da nova versao, compativel com a anterior                                              │
#         │                          - Z - eh o numero da versao atual, com a correcao de bugs                                                │
#         │                          - versao inicial V1.0.1                                                                                  │
#         │      9) 19/12/2024 - V1.0.3 atualizacao dos arquivos, permitindo serem executados como binarios                                   │
#         │     10) 18/08/2025 - V2.0.0 inserido o script processa_wrf_interativemap e os vetores de leitura dos arquivos de saida            │
#         │     11) 26/12/2025 - V2.0.1 ajustando o scrip para novas funcionalidades do d-interative-map                                      │
#         │     12) 21/01/2026 - V3.0.0 vetorizando toas as tranferencias de dados e integrando ao wudapt                                     │
#         │     13) 15/03/2026 - V3.1.0 corrigindo bugs em passagem de dados                                                                  │
#         │                                                                                                                                   │
#         │                                                                                                                                   │
#         └───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
#~~~~~~~~nao apagar~~~~~~~~~~~~~~~~
#├  ┤  ┐  └  ┴  ┬  ─  ┼  ┘  ┌  │  ¬
#╠  ╣  ╗  ╚  ╩  ╦  ═  ╬  ╝  ╔  ║  ╗
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#        ┌───────────────────────┬──────────────────────────────────────────────────────────┬─────────────────────────────────────────────────┐
#        │                       │                                                          │                                                 │
#        │                       │          WRF OPERACIONAL                                 │                                                 │
#        │                       │  SEQUENCIA DE SCRIPTS (EXECUCAO)                         │         OBJETIVO                                │
#        │     ORDEM             │            NOME SCRIPT                                   │                                                 │
#        │                       │                                                          │                                                 │
#        ├───────┬───────────────┼──────────────────────────────────────────────────────────┼─────────────────────────────────────────────────┤
#        │  00.  │ SYSTEM        │ processa_wrf_system.scpt                                 │ controle dos scripts                            │
#        ├───────┼───────────────┼──────────────────────────────────────────────────────────┼─────────────────────────────────────────────────┤
#        │  01.  │ BUSCA         │ processa_wrf_busca.scpt                                  │ busca de dados do GFS                           │
#        ├───────┼───────────────┼──────────────────────────────────────────────────────────┼─────────────────────────────────────────────────┤
#        │  02.  │ WPS           │ processa_wrf_wps.scpt                                    │ processa o WPS                                  │
#        ├───────┼───────────────┼──────────────────────────────────────────────────────────┼─────────────────────────────────────────────────┤
#        │  03.  │ WRF           │ processa_wrf_wrf.scpt                                    │ processa o WRF                                  │
#        ├───────┼───────────────┼──────────────────────────────────────────────────────────┼─────────────────────────────────────────────────┤
#        │  04.  │ PYTHON        │ processa_wrf_python.scpt                                 │ formata dados do python                         │
#        │  04.1 │ PYTHON        │ d-python/wrf/processa_wrf_figuras.py                     │ faz figuras em python do WRF                    │
#        ├───────┼───────────────┼──────────────────────────────────────────────────────────┼─────────────────────────────────────────────────┤
#        │  05.  │ MMIF          │ processa_wrf_mmif.scpt                                   │ processa os dados do MMIF                       │
#        │  05.1 │ MMIF          │ d-python/wap-aermod/processa_wrf_aermod.py               │ faz a rosa dos ventos                           │
#        │  ---- │ MMIF          │ MMIF/mmif.exe                                            │ executa o MMIF (binario)                        │
#        ├───────┼───────────────┼──────────────────────────────────────────────────────────┼─────────────────────────────────────────────────┤
#        │  06.  │ CALWRF        │ processa_wrf_calwrf.scpt                                 │ formata dados para o CALWRF                     │
#        │  ---- │ CALWRF        │ CALWRF/calwrf.exe                                        │ executa o CALWRF (binario)                      │
#        ├───────┼───────────────┼──────────────────────────────────────────────────────────┼─────────────────────────────────────────────────┤
#        │  07.  │ READWRFNC     │ processa_wrf_readwrfnc.scpt                              │ formata dados para o READ_WRF_NC                │
#        │  ---- │ READWRFNC     │ READ_WRF_NC/read_wrf_nc.exe                              │ executa o read_wrf_nc (binario)                 │
#        ├───────┼───────────────┼──────────────────────────────────────────────────────────┼─────────────────────────────────────────────────┤
#        │  08.  │ LEAFLET       │ processa_wrf_interativemap.scpt                          │                                                 │
#        │  08.1 │               │ d-leaflet/processa_wrf_geoJSON.py                        │ gera as figuras para o leaflet                  │
#        │  08.2 │               │ d-leaflet/processa_wrf_tiles                             │                                                 │
#        ├───────┼───────────────┼──────────────────────────────────────────────────────────┼─────────────────────────────────────────────────┤
#        │  09.  │ FTP           │ processa_wrf_ftp.scpt                                    │ envia os dados para o portal ufes               │
#        ├───────┼───────────────┼──────────────────────────────────────────────────────────┼─────────────────────────────────────────────────┤
#        │  10.  │ DELETA        │ processa_wrf_deletafiles.scpt                            │ deleta arquivos antigos (>30 dias)              │
#        └───────┴───────────────┴──────────────────────────────────────────────────────────┴─────────────────────────────────────────────────┘


echo ' '
echo ' '
echo '    ╔══════════════════════════════════════════════════════╗  ╔═════════════════════════════════════════════════════════════════════════╗'
echo '    ║ ╔════╗       ╔════════════╗════════════╗════╗        ║  ║ ╔════╗       ╔════════════╗══════════╗ ╔════╗  ╔════╗════╗════╗  ╔════╗ ║'
echo '    ║ ║ ░░ ║       ║ ░░░░░░░░░░ ║ ░░░░░░░░░░ ║ ░░ ║        ║  ║ ║ ░░ ║       ║ ░░░░░░░░░░ ║ ░░░░░░░░ ╚═╗ ░░ ║  ║ ░░ ║ ░░ ║ ░░ ║  ║ ░░ ║ ║'
echo '    ║ ║ ░░ ║       ║ ░░╔════════╝ ░░ ╔══╗ ░░ ║ ░░ ║        ║  ║ ║ ░░ ║       ║ ░░ ╔══╗ ░░ ║ ░░ ╔══╗ ░░ ║ ░░░ ║║ ░░░ ║ ░░ ║ ░░░ ║║ ░░░ ║ ║'
echo '    ║ ║ ░░ ║       ║ ░░╚═════╗  ║ ░░ ╚══╝ ░░ ║ ░░ ║        ║  ║ ║ ░░ ║       ║ ░░ ╚══╝ ░░ ║ ░░ ╚══╝ ░░ ║ ░░ ░  ░ ░░ ║ ░░ ║ ░░ ░  ░ ░░ ║ ║'
echo '    ║ ║ ░░ ║       ║ ░░░░░░░ ║  ║ ░░░░░░░░░░ ║ ░░ ║        ║  ║ ║ ░░ ║       ║ ░░░░░░░░░░ ║ ░░░░░░░░  ═╝ ░░  ░░  ░░ ║ ░░ ║ ░░  ░░  ░░ ║ ║'
echo '    ║ ║ ░░ ║       ║ ░░╔═════╝  ║ ░░ ╔══╗ ░░ ║ ░░ ║        ║  ║ ║ ░░ ║       ║ ░░ ╔══╗ ░░ ║ ░░ ╔══╗ ░░ ║ ░░ ╔══╗ ░░ ║ ░░ ║ ░░ ╔══╗ ░░ ║ ║'
echo '    ║ ║ ░░ ╚═══════╗ ░░╚════════╗ ░░ ║  ║ ░░ ║ ░░ ╚══════╗ ║  ║ ║ ░░ ╚═══════╗ ░░ ║  ║ ░░ ║ ░░ ╚══╝ ░░ ║ ░░ ║  ║ ░░ ║ ░░ ║ ░░ ║  ║ ░░ ║ ║'
echo '    ║ ║ ░░░░░░░░░░ ║ ░░░░░░░░░░ ║ ░░ ║  ║ ░░ ║ ░░░░░░░░░ ║ ║  ║ ║ ░░░░░░░░░░ ║ ░░ ║  ║ ░░ ║ ░░░░░░░░ ╔═╝ ░░ ║  ║ ░░ ║ ░░ ║ ░░ ║  ║ ░░ ║ ║'
echo '    ║ ╚════════════╝════════════╝════╝  ╚════╝═══════════╝ ║  ║ ╚════════════╝════╝  ╚════╝══════════╝ ╚════╝  ╚════╝════╝════╝  ╚════╝ ║'
echo '    ║                                                      ║  ║                                                                         ║'
echo '    ║                https://leal.ufes.br                  ║  ║                          http://labmim.if.ufba.br                       ║'
echo '    ║                                                      ║  ║                                                                         ║'
echo '    ╚══════════════════════════════════════════════════════╝  ╚═════════════════════════════════════════════════════════════════════════╝'
echo ' '
echo ' '
echo '    ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
echo '    │                                                                                                                                   │'
echo '    │                                              PROCESSA WRF OPERACIONAL                                                             │'
echo '    │                                                                                                                                   │'
echo '    ├───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤'
echo '    │ SINTAXE DO COMANDO DE EXECUCAO DO SCRIPT (EX.):       O QUE FAZ (veja namelist.processa as configuracoes):                        │'
echo '    ├───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤'
echo '    │                                                                                                                                   │'
echo '    │     ./processa_wrf_00_system-V* (*.scpt ou *.bin)  ──────➤ roda a data de hoje baixando os dados do GFS (0.50")                   │'
echo '    │                                                                                                                                   │'
echo '    │                                                     ┌──➤ roda WRF com data escolhida e dados do GFS (0.50" ou 0.25"), FNL ou GDAS │'
echo '    │ ┌──➤ ./processa_wrf_00_system-V*     20240811 S  ───┼──➤ arquivos GFS ou FNLl com 0.50" =  ~44 Mb (55 km x 55 km)                 │'
echo '    │ │                                                   └──➤ arquivo GFS ou GDAS com 0.25"  = ~520 Mb (27 km x 27 km)                 │'
echo '    │ │                                                                                                                                 │'
echo '    │ └──➤ ./processa_wrf_00_system-V*     20240811 N  ──────➤ roda a data de hoje sem baixar os dados novamente                        │'
echo '    │                                                                                                                                   │'
echo '    └───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
echo ' '
echo ' '


#SYSTEM - LEITURA INICIAL
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


    #local de trabalho
    dir_local=$( grep dir_local /home/models/WRF/wrf-op/.FUNCTIONS_WRF_environment | awk '{print $2}' )
    export PATH=$PATH:${dir_local}:.
    cd $dir_local


    #path das funcoes
    #----------------
    export PATH=$PATH:d-functions
    PATH_FUNC=/home/models/WRF/wrf-op/d-functions


    #verificando de o arquivo namelist.processa existe
    file_in=namelist.processa
    echo "    + SYSTEM - Abrindo o arquivo $file_in em $dir_local"
    if [[ ! -f $file_in ]]; then
        echo "    + SYSTEM - ERRO! Nao existe o arquivo $file_in em $dir_local"
        echo "    + SYSTEM - Insira o arquivo e rode novamente o processa_wrf_00_system"
        exit
    fi

    file_out=namelist.tmp

    [ -e $file_out ] && \rm -f  $file_out

    nc=0
    while IFS= read -r line; do

            (( nc = nc + 1 ))
            head -${nc} ${file_in} | tail -1  > line
            first=$( cat line | cut -c1-1 )

            if [[ ${first} != ' ' && ${first} != '' ]]
                then
                cat line >> ${file_out}
            fi

    done < ${file_in}

    \rm -f line

    namelist=$file_out

    #para o log
    echo "    + SYSTEM - ════════════════════════════════════════" >$dir_local/d-log/wrf-op.debug
    echo "    + SYSTEM - PROCESSA-COMANDOS DE EXECUCAO DO SYSTEMA">>$dir_local/d-log/wrf-op.debug
    echo "    + SYSTEM - ════════════════════════════════════════">>$dir_local/d-log/wrf-op.debug


    if  [[ $# -eq 0 ]]; then       #  Recebi 2 parametros?
        echo "    + SYSTEM - EXCUCUTANTO ./processa_wrf_00_system"
    elif [[ $# -eq 1 ]]; then
        echo "    + SYSTEM - EXCUCUTANTO ./processa_wrf_00_system $1"
    elif [[ $# -eq 2 ]]; then
        echo "    + SYSTEM - EXCUCUTANTO ./processa_wrf_00_system $1 $2"
    else
        echo "Exemplos:"
        echo "         ./processa_wrf_00_system-Vx.x.x.scpt"
        echo "         ./processa_wrf_00_system-Vx.x.x.scpt 20240811 S"
        echo "         ./processa_wrf_00_system-Vx.x.x.scpt 20240811 N"
        echo ' '
        echo ' '
        exit 1
    fi


#SYSTEM - processa_wrf_system
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


    #pegando dados de entrada (argumentos 1 e 2) (se operacional "" - branco = data de hoje)
    #---------------------------------------------------------------------------------------
    busca_dados=$2


    #preenchendo a data
    #------------------
    if [[ ${busca_dados} = '' ]]; then
        #pega a data de hoje
        #-------------------
        year=$(date +%Y)
        month=$(date +%m)
        day=$(date +%d)
        yyyymmdd=${year}${month}${day}
    elif [[ ${busca_dados} = "S" ]]; then
        #pega o argumento da data digitada
        #---------------------------------
        yyyymmdd=$1
    else
        #roda o dia de hoje sem baixar dados
        #-----------------------------------
        run_manual="manual"
        yyyymmdd=$1
    fi


    #verificando de o arquivo namelist.processa existe
    if [[ ! -f $namelist ]]; then
        echo "    + SYSTEM - ERRO! Nao existe o arquivo $namelist em $dir_local"
        echo "    + SYSTEM - Insira o arquivo e rode novamente o processa_wrf"
        exit
    fi


    VERSAO_NAMELIST=$(            grep VERSAO_NAMELIST            $namelist | awk '{print $3}' )
    processo_binario=$(           grep processo_binario           $namelist | awk '{print $3}' )
    echo " "
    echo '    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
    echo "    + SYSTEM - SCRIPTS QUE SERAO EXECUTADOS PELO NAMELIST VERSAO $VERSAO_NAMELIST"
    if [[ $processo_binario = 'S' ]]; then

        echo "    + SYSTEM - PROCESSO EXECUTAVEL EH BINARIO"
        ext=bin

    else

        echo "    + SYSTEM - PROCESSO EXECUTAVEL EH EDITAVEL"
        ext=scpt

    fi
    echo '    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
    echo " "


    processa_wrf_system=$(        grep processa_wrf_system        $namelist | awk '{print $3}' )
    processa_wrf_busca=$(         grep processa_wrf_busca         $namelist | awk '{print $3}' )
    processa_wrf_wps=$(           grep processa_wrf_wps           $namelist | awk '{print $3}' )
    processa_wrf_wrf=$(           grep processa_wrf_wrf           $namelist | awk '{print $3}' )
    processa_wrf_python=$(        grep processa_wrf_python        $namelist | awk '{print $3}' )
    processa_wrf_figuras=$(       grep processa_wrf_figuras       $namelist | awk '{print $3}' )
    processa_wrf_mmif=$(          grep processa_wrf_mmif          $namelist | awk '{print $3}' )
    processa_wrf_wap=$(           grep processa_wrf_wap           $namelist | awk '{print $3}' )
    processa_wrf_readwrfnc=$(     grep processa_wrf_readwrfnc     $namelist | awk '{print $3}' )
    processa_wrf_interativemap=$( grep processa_wrf_interativemap $namelist | awk '{print $3}' )
    processa_wrf_geoJSON=$(       grep processa_wrf_geoJSON       $namelist | awk '{print $3}' )
    processa_wrf_tiles=$(         grep processa_wrf_tiles         $namelist | awk '{print $3}' )
    processa_wrf_ftp=$(           grep processa_wrf_ftp           $namelist | awk '{print $3}' )
    processa_wrf_deletafiles=$(   grep processa_wrf_deletafiles   $namelist | awk '{print $3}' )
    VERSAO_WPS=$(                 grep VERSAO_WPS                 $namelist | awk '{print $3}' )
    VERSAO_WRF=$(                 grep VERSAO_WRF                 $namelist | awk '{print $3}' )
    VERSAO_MMIF=$(                grep VERSAO_MMIF                $namelist | awk '{print $3}' )
    

    #sao binarios somente os *.scpt
    processa_wrf_system=$processa_wrf_system.$ext
    processa_wrf_busca=$processa_wrf_busca.$ext
    processa_wrf_wps=$processa_wrf_wps.$ext
    processa_wrf_wrf=$processa_wrf_wrf.$ext
    processa_wrf_python=$processa_wrf_python.$ext
    processa_wrf_mmif=$processa_wrf_mmif.$ext
    processa_wrf_readwrfnc=$processa_wrf_readwrfnc.$ext
    processa_wrf_interativemap=$processa_wrf_interativemap.$ext
    processa_wrf_tiles=$processa_wrf_tiles.$ext
    processa_wrf_ftp=$processa_wrf_ftp.$ext
    processa_wrf_deletafiles=$processa_wrf_deletafiles.$ext


    VERSAO_WPS=WPS-$VERSAO_WPS
    VERSAO_WRF=WRF-$VERSAO_WRF
    VERSAO_MMIF=mmif-$VERSAO_MMIF


    EXECUTAR_processa_wrf_system=$(        grep processa_wrf_system        $namelist | awk '{print $4}' )
    EXECUTAR_processa_wrf_busca=$(         grep processa_wrf_busca         $namelist | awk '{print $4}' )
    EXECUTAR_processa_wrf_wps=$(           grep processa_wrf_wps           $namelist | awk '{print $4}' )
    EXECUTAR_processa_wrf_wrf=$(           grep processa_wrf_wrf           $namelist | awk '{print $4}' )
    EXECUTAR_processa_wrf_python=$(        grep processa_wrf_python        $namelist | awk '{print $4}' )
    EXECUTAR_processa_wrf_figuras=$(       grep processa_wrf_figuras       $namelist | awk '{print $4}' )
    EXECUTAR_processa_wrf_mmif=$(          grep processa_wrf_mmif          $namelist | awk '{print $4}' )
    EXECUTAR_processa_wrf_wap=$(           grep processa_wrf_wap           $namelist | awk '{print $4}' )
    EXECUTAR_processa_wrf_readwrfnc=$(     grep processa_wrf_readwrfnc     $namelist | awk '{print $4}' )
    EXECUTAR_processa_wrf_interativemap=$( grep processa_wrf_interativemap $namelist | awk '{print $4}' )
    EXECUTAR_processa_wrf_ftp=$(           grep processa_wrf_ftp           $namelist | awk '{print $4}' )
    EXECUTAR_processa_wrf_deletafiles=$(   grep processa_wrf_deletafiles   $namelist | awk '{print $4}' )


    echo "    + SYSTEM - $processa_wrf_system                 - executar (S/N)?  $EXECUTAR_processa_wrf_system"
    if [[ $EXECUTAR_processa_wrf_system != 'S' ]]; then; echo "    + SYSTEM - FIM DO PROCESSO"; exit; fi;
    echo "    + SYSTEM - $processa_wrf_busca                  - executar (S/N)?  $EXECUTAR_processa_wrf_busca"
    echo "    + SYSTEM - $processa_wrf_wps                    - executar (S/N)?  $EXECUTAR_processa_wrf_wps"
    echo "    + SYSTEM - $processa_wrf_wrf                    - executar (S/N)?  $EXECUTAR_processa_wrf_wrf"
    echo "    + SYSTEM - $processa_wrf_python                 - executar (S/N)?  $EXECUTAR_processa_wrf_python"
    echo "    +        - $processa_wrf_figuras"
    echo "    + SYSTEM - $processa_wrf_mmif                   - executar (S/S)?  $EXECUTAR_processa_wrf_mmif"
    echo "    +        - $processa_wrf_wap"
    echo "    + SYSTEM - $processa_wrf_readwrfnc              - executar (S/N)?  $EXECUTAR_processa_wrf_readwrfnc"
    echo "    + SYSTEM - $processa_wrf_interativemap         - executar (S/N)?  $EXECUTAR_processa_wrf_interativemap"
    echo "    + SYSTEM - $processa_wrf_ftp                    - executar (S/N)?  $EXECUTAR_processa_wrf_ftp"
    echo "    + SYSTEM - $processa_wrf_deletafiles            - executar (S/N)?  $EXECUTAR_processa_wrf_deletafiles"
    echo "    + SYSTEM - $VERSAO_WPS"
    echo "    + SYSTEM - $VERSAO_WRF"
    echo "    + SYSTEM - $VERSAO_MMIF"


    echo " "
    echo '    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
    echo "    + SYSTEM - LEITURA DE DADOS"
    echo '    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
    echo " "


    WPSoutput=$(        grep WPSoutput        $namelist | awk '{print $3}' )
    WRFoutput=$(        grep WRFoutput        $namelist | awk '{print $3}' )
    DATAhistory=$(      grep DATAhistory      $namelist | awk '{print $3}' )
    DATAtype=$(         grep DATAtype         $namelist | awk '{print $3}' )
    DATAgeog=$(         grep DATAgeog         $namelist | awk '{print $3}' )
    numproc=$(          grep numproc          $namelist | awk '{print $3}' )
    time_step=$(        grep time_step        $namelist | awk '{print $3}' )
    geog_data_res=$(    grep geog_data_res    $namelist | awk '{print $3}' )
    interval_seconds=$( grep interval_seconds $namelist | awk '{print $3}' )
    wudapt_op=$(        grep wudapt_op        $namelist | awk '{print $3}' )


    WPSoutput=$WPSoutput/$yyyymmdd
    WRFoutput=$WRFoutput/$yyyymmdd
    DATAhistory=$DATAhistory/$yyyymmdd
    dir_wps=$dir_local/$VERSAO_WPS
    dir_wrf=$dir_local/$VERSAO_WRF


    echo "    + SYSTEM - (dir_local)        - local de trabalho                      = $dir_local"
    echo "    + SYSTEM - (DATAhistory)      - local do modelo global                 = $DATAhistory - ($DATAtype)"
    echo "    + SYSTEM - (DATAgeog)         - local DO landuse (WPS)                 = $DATAgeog"
    echo "    + SYSTEM - (WPSoutput)        - local de output do WPS                 = $WPSoutput"
    echo "    + SYSTEM - (WRFoutput)        - local de output do WRF                 = $WRFoutput"
    echo "    + SYSTEM - (DATAtype)         - dados de input do WRF                  = $DATAtype"
    echo "    + SYSTEM - (dir_wps)          - local do WPS                           = $dir_wps"
    echo "    + SYSTEM - (dir_wrf)          - local do WRF                           = $dir_wrf"
    echo "    + SYSTEM - (numproc)          - numero de processadores utilizados     = $numproc"
    echo "    + SYSTEM - (time_step)        - passo no tempo                         = $time_step s"
    echo "    + SYSTEM - (geog_data_res)    - opcao dos dados geograficos (WPS)      = $geog_data_res"
    echo "    + SYSTEM - (interval_seconds) - intervalo de leitura                   = $interval_seconds s - ($DATAtype)"
    echo "    + SYSTEM - (wudapt)           - WUDAPT                                 = $wudapt_op"


    #para o log
    echo "    + ───────────────────────────────────────────────────" >$dir_local/d-log/${yyyymmdd}-wrf-op.debug
    echo "    + SYSTEM - COMANDOS DE EXECUCAO DO SYSTEMA PARA DEBUG">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
    echo "    + ───────────────────────────────────────────────────">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug


    #procurando o tempo maximo para baixar os dados
    hour_end=0  #tempo maximo
    nexe_simulacao=0   #numero de vezes que a simulacao do wrf/wps serao executados

    #simulacao inicial e final
    wrf_ini=$(          grep wrf_ini          $namelist | awk '{print $3}' )
    wrf_fin=$(          grep wrf_fin          $namelist | awk '{print $3}' )
    echo "    + SYSTEM - (wrf_ini)          - wrf inicial                            = $wrf_ini"
    echo "    + SYSTEM - (wrf_fin)          - wrf final                              = $wrf_fin"


    for i in {$wrf_ini..$wrf_fin}; do

        if [[ $i -le 9 ]]; then

            ii="0$i"

        else

            ii=$i

        fi

        #verificar se o WRF estah ativado (opcao S para exe_simulacao)
        #1            2          3              4            5              6
        #simulacao    projeto    num_max_dom    deltahora    lat_central    lon_central
        exe_simulacao=$(grep wrf$ii $namelist | awk '{print  $3}' )
        projeto=$(      grep wrf$ii $namelist | awk '{print  $4}' )
        num_max_dom=$(  grep wrf$ii $namelist | awk '{print  $5}' )
        deltahora=$(    grep wrf$ii $namelist | awk '{print  $6}' )
        lat_central=$(  grep wrf$ii $namelist | awk '{print  $7}' )
        lon_central=$(  grep wrf$ii $namelist | awk '{print  $8}' )

        #verificar se o WRF estah ativado (opcao S para EXECUTAR_processa_wrf_wrf)
        if [[ $exe_simulacao = 'S' ]]; then

            nexe_simulacao=$(($nexe_simulacao + 1))   #incremento
            if [[ $hour_end -lt $deltahora ]]; then; hour_end=$deltahora; fi; #definie a quantidade de arquivos a serem baixados em d-input de acordo com as simulacoes a serem executadas

        fi

    done


    echo "    + SYSTEM - o WPS e o WRF sera(o) executado(s)                          = $nexe_simulacao vez(es)"
    echo "    + SYSTEM - quantidade de arquivos para download (WPS/WRF)              = $((hour_end/3)) arquivos $DATAtype para $hour_end horas de simulacao"


#SYSTEM - processa busca dados para o wrf
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


#buscando os dados do modelo global gfs se operacional
    if [[ $run_manual != "manual" ]] ; then

        if [[ $EXECUTAR_processa_wrf_busca = 'S' ]]; then

            echo "    + SYSTEM - buscando os dados do modelo global para as simulacoes       = $hour_end h"
            #executando a busca de dados
            #---------------------------
            #definindo o vetor de variaveis para passagem para o processa_wrf_busca
            #         0         1         2
            vetbusca=($yyyymmdd $hour_end $DATAtype)

            echo "    + SYSTEM - ./${processa_wrf_busca} ${vetbusca[@]}">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
            echo "    + SYSTEM - ----------------------------">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
            echo "    + SYSTEM - passando ${#vetbusca[*]} variaveis para o ${processa_wrf_busca}"
            ${dir_local}/$processa_wrf_busca ${vetbusca[@]}

        else

            echo "    + SYSTEM - buscar dados para o WPS/WRF?                                = N"

        fi

        #run_manual='manual'    #se tiver mais de uma rodada do wrf nao busca mais os dados

    else

        echo "    + SYSTEM - buscar dados para o WPS/WRF?                                = N"

    fi



#SYSTEM - processa wps, wrf, graficos etc
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


    #lendo os dados das opcoes para rodar o WRF ou extracao de dados do WRF (serie temporal)
    for i in {$wrf_ini..$wrf_fin}; do

        if [[ $i -le 9 ]]; then

            ii="0$i"

        else

            ii=$i

        fi

        #buscando a(s) simulacao do wrf que serao executadas
        exe_simulacao=$(         grep wrf$ii $namelist | awk '{print  $3}' )
        echo "    + SYSTEM - (exe_simulacao)    - o wrf$ii serah executado                ? $exe_simulacao"
        echo "    + SYSTEM - (exe_simulacao)    - o wrf$ii serah executado                ? $exe_simulacao">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug


        if [[ $exe_simulacao = 'S' ]]; then

            echo " "
            echo '    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
            echo "    + SYSTEM - EXECUTANDO O SYSTEM PARA wrf$ii do $namelist"
            echo '    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
            echo " "
            echo " "

            projeto=$(        grep wrf$ii $namelist | awk '{print  $4}' )
            num_max_dom=$(    grep wrf$ii $namelist | awk '{print  $5}' )
            deltahora=$(      grep wrf$ii $namelist | awk '{print  $6}' )
            lat_central=$(    grep wrf$ii $namelist | awk '{print  $7}' )
            lon_central=$(    grep wrf$ii $namelist | awk '{print  $8}' )
            localidade_ini=$( grep wrf$ii $namelist | awk '{print  $9}' )
            localidade_fin=$( grep wrf$ii $namelist | awk '{print $10}' )

            #local de output do WRF
            WRFoutput1=$WRFoutput/wrf${ii}
            mkdir -p $WRFoutput1

            echo "    + SYSTEM - (wrfii)            - lendo os dados da SIMULACAO            = wrf$ii"
            echo "    + SYSTEM - (projeto)          - a quem se destina os dados             = $projeto"
            echo "    + SYSTEM - (num_max_dom)      - numero de maximo de dominios do WRF    = $num_max_dom"
            echo "    + SYSTEM - (lat_central       - latitude central                       = $lat_central"
            echo "    + SYSTEM - (lon_central)      - longitude central                      = $lon_central"
            echo "    + SYSTEM - (deltahora)        - tempo total de simulacao do WRF (h)    = $deltahora"
            echo "    + SYSTEM - (localidade_ini)   - local inicial extracao de dados do WRF = $localidade_ini"
            echo "    + SYSTEM - (localidade_fin)   - local final   extracao de dados do WRF = $localidade_fin"
            echo "    + SYSTEM - (WRFoutput)        - local de output do WRF                 = $WRFoutput1"

#             for (( j=5; j>0; j--)); do; sleep 1 &; printf "    + SYSTEM - aguarde por $j segundos... ** OU presione Ctrl+c para cancelar\r"; wait; done

            #data final
            #----------
            #intervalo de tempo total da rodada em namelist.processa em deltahora, onde:
            #S      = (+/-) indicando adicao ou subtracao de tempo
            #para S = - (menos) --> rodada do dia atual-deltahora  ateh dia atual
            #para S = + (mais)  --> rodada do dia atual ateh dia atual+ deltahora
            #hh     = hora
            #mm     = minuto
            #ss     = segundos
            deltaHour=+0${deltahora}0000

            if [[ $deltahora -gt 99 ]]; then

                deltaHour=+${deltahora}0000

            fi

            year1=$(echo ${yyyymmdd}  | cut -b 1-4)
            month1=$(echo ${yyyymmdd} | cut -b 5-6)
            day1=$(echo ${yyyymmdd}   | cut -b 7-8)
            hour1=00

            tmpTime=`$dir_local/d-fortran/calcTime/calcTime $year1$month1$day1'000000' $deltaHour`
            year2=`expr substr $tmpTime 1 4`
            month2=`expr substr $tmpTime 5 2`
            day2=`expr substr $tmpTime 7 2`
            hour2=`expr substr $tmpTime 9 2`

            #para o log
            echo "                               ">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
            echo "                               ">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
            echo "                               ">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
            echo "    + SYSTEM - ================">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
            echo "    + SYSTEM - data: $day1/$month1/$year1">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
            echo "    + SYSTEM - executando SIMULACAO wrf$ii do $namelist">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
            echo "    + SYSTEM - === COMANDOS ===">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
            echo "    +                          ">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug

            echo "    + SYSTEM - para executar BUSCA:">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
            echo "    + SYSTEM - ./${processa_wrf_busca} $yyyymmdd $hour_end $DATAtype">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
            echo "    + SYSTEM - ----------------------------">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug

            if [[ ${run_manual} == "manual" ]] ; then

                echo "    + SYSTEM - WRF RODANDO MANUAL por                                      = $hour_end h"
                echo "    + SYSTEM - data inicial da simulacao                                   = $year1/$month1/$day1 as 00 h"
                echo "    + SYSTEM - data final   da simulacao                                   = $year2/$month2/$day2 as $hour2 h"

            else

                echo "    + SYSTEM - WRF RODANDO OPERACIONAL por                                 = $hour_end h"
                echo "    + SYSTEM - data inicial da simulacao                                   = $year1/$month1/$day1 as 00 h"
                echo "    + SYSTEM - data final   da simulacao                                   = $year2/$month2/$day2 as $hour2 h"

            fi


#SYSTEM - processa wps
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


            echo "    + SYSTEM - criando o diretorio de saida do WPS  em                     = $WPSoutput/wrf${ii}"
            WPSoutput1=$WPSoutput/wrf${ii}
            mkdir -p $WPSoutput1

            if [[ $EXECUTAR_processa_wrf_wps = 'S' ]]; then

                echo " "
                echo '    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
                echo "    + SYSTEM - EXCUTANDO O $VERSAO_WPS"
                echo '    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
                echo " "

                echo "    + SYSTEM - para executar o WPS:">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug

                #definindo o vetor de variaveis para passagem para o processa_wrf_wps
                #       0      1       2     3      4      5       6     7      8            9            10           11       12         13        14            15        16             17       18                 19       20
                vetwps=($year1 $month1 $day1 $hour1 $year2 $month2 $day2 $hour2 $num_max_dom $lat_central $lon_central $numproc $WPSoutput1 $DATAgeog $DATAhistory $DATAtype $geog_data_res $dir_wps $interval_seconds  $projeto $wudapt_op)

                echo "    + SYSTEM - ./${processa_wrf_wps} ${vetwps[@]}">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
                echo "    + SYSTEM - ----------------------------">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
                echo "    + SYSTEM - passando ${#vetwps[*]} variaveis para o ${processa_wrf_wps}"
                #chama o processa_wrf_wps
                ${dir_local}/$processa_wrf_wps ${vetwps[@]}

            else

                echo "    + SYSTEM - WPS                                                         = nao executado"

            fi


#SYSTEM - processa wrf
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


            if [[ $EXECUTAR_processa_wrf_wrf = 'S' ]]; then

                echo " "
                echo '    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
                echo "    + SYSTEM - EXCUTANDO O WRF"
                echo '    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
                echo " "

                echo "    + SYSTEM - para executar o WRF:">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug

                #definindo o vetor de variaveis para passagem para o processa_wrf_wrf
                #       0      1       2     3      4      5       6     7      8            9        10         11          12       13       14
                vetwrf=($year1 $month1 $day1 $hour1 $year2 $month2 $day2 $hour2 $num_max_dom $numproc $time_step $WPSoutput1 $dir_wrf $projeto $wudapt_op)

                echo "    + SYSTEM - ./$processa_wrf_wrf ${vetwrf[@]}">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
                echo "    + SYSTEM - ----------------------------">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
                echo "    + SYSTEM - passando ${#vetwrf[*]} variaveis para o $processa_wrf_wrf"
                #chama o processa_wrf_wrf
                ${dir_local}/${processa_wrf_wrf} ${vetwrf[@]}

                echo "    + SYSTEM - movendo wrfout* para /d-output/$yyyymmdd/wrf${ii}"
                cd $dir_local
                #se existe o arquivo wrfout_d0* mover para WRFoutput1
                [ ! -f $dir_wrf/test/em_real/wrfout_d0* ] || mv -f $dir_wrf/test/em_real/wrfout_d0* $WRFoutput1

                if [[ $projeto = "LEAL" ]];then

                    echo "    + SYSTEM - criando o link simbolico para os dados da plataforma LEAL:"
                    echo "    + SYSTEM - $$WRFoutput1"

                    cd $dir_local/d-output/$yyyymmdd
                    ln -sf $WRFoutput1/wrfout_d0* .
                    cd $dir_local

                    echo "    + SYSTEM - criando o link simbolico para integração WRF-monitoramento no site"
                    dir_site=/home/models/site-lealufes-op/Database
                    cd $dir_site
                    echo "    + SYSTEM - $dir_site"
                    echo "    + SYSTEM - $WRFoutput1/wrfout_d0$num_max_dom* ."
                    ln -sf $WRFoutput1/wrfout_d0$num_max_dom* .
                    cd $dir_local

                fi

            else

                echo "    + SYSTEM - WRF                                                         = nao executado"

            fi


#SYSTEM - processa python, mmif, read_wrf_nc e interative map
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


            echo " "
            echo '    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
            echo "    + SYSTEM - EXECUTANDO 1-PYTHON, 2-MMIF, 3-READWRFNC E INTERATIVE MAP"
            echo '    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
            echo " "

            loc_ini=$(echo $localidade_ini  | cut -b 12-14)
            loc_fin=$(echo $localidade_fin  | cut -b 12-14)

            for j in {$loc_ini..$loc_fin}; do

                #preenchendo 01,02,...,10,11,...
                if [[ $j -le 9 ]]; then

                    jj="0$j"

                else

                    jj=$j

                fi

                #pega as informacoes dos pontos de interesse para a geracao dos arquivos para o AERMOD
                nome_estacao=$( grep localidade$jj $namelist | awk '{print $3}' )
                utm_zona=$(     grep localidade$jj $namelist | awk '{print $4}' )
                lat1=$(         grep localidade$jj $namelist | awk '{print $5}' )
                lon1=$(         grep localidade$jj $namelist | awk '{print $6}' )
                timezone=$(     grep localidade$jj $namelist | awk '{print $7}' )
                time_serie=$(   grep localidade$jj $namelist | awk '{print $8}' )
                TSgrade=$(      grep localidade$jj $namelist | awk '{print $9}' )
                pyFig=$(        grep localidade$jj $namelist | awk '{print $10}')
                DOMini=$(       grep localidade$jj $namelist | awk '{print $11}')
                DOMfim=$(       grep localidade$jj $namelist | awk '{print $12}')
                IntMap=$(       grep localidade$jj $namelist | awk '{print $13}')
                MMIF=$(         grep localidade$jj $namelist | awk '{print $14}')
                h_estacao=$(    grep localidade$jj $namelist | awk '{print $15}')
                exe_ftp=$(      grep localidade$jj $namelist | awk '{print $16}')


#SYSTEM - leitura das variaveis de saida do wrf para graficos e extracao de dados
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


                #variaveis de saida do WRF de output para gerar figuras (series temporais e graficos do wrf)
                if [[ $time_serie = "S" || $IntMap = "S" || $pyFig = "S" ]]; then

                    nvar=$( grep nvar $namelist | awk '{print $3}' )
                    echo "    + SYSTEM - (nvar)                 - no. de variaveis do wrfout         = $nvar"

                    k1=0
                    k2=0
                    k3=0
                    k4=0
                    for j in {1..$nvar}; do

                        #preenchendo 01,02,...,10,11,...
                        if [[ $j -le 9 ]]; then
                            jj="00$j"
                        elif [[ $j -le 99 ]]; then
                            jj=0$j
                        else
                            jj=$j
                        fi

                        #pega as informacoes dos pontos de interesse para a geracao dos arquivos para o AERMOD
                        vet=($( grep var$jj $namelist | awk '{print $3}' ))
                        if [[ $vet != 0000 ]]; then

                            num=($( grep var$jj $namelist | awk '{print $3}' ))
                            var=($( grep var$jj $namelist | awk '{print $4}' ))

                            TIME_SERIES_SOUNDING1=`expr substr $num 1 1`
                            TIME_SERIES_SURFACE1=`expr substr $num 2 1`
                            GRAFICOS_WRF1=`expr substr $num 3 1`
                            INTERATIVEMAP1=`expr substr $num 4 1`

                            #echo "    + SYSTEM - (TIME_SERIES_SOUNDING1) = $TIME_SERIES_SOUNDING1"
                            #echo "    + SYSTEM - (TIME_SERIES_SURFACE1)  = $TIME_SERIES_SURFACE1"
                            #echo "    + SYSTEM - (GRAFICOS_WRF1)         = $GRAFICOS_WRF1"
                            #echo "    + SYSTEM - (INTERATIVEMAP1)  = $INTERATIVEMAP1"

                            if [[ $TIME_SERIES_SOUNDING1 -eq 1 ]]; then; TIME_SERIES_SOUNDING[$k1]=$var; k1=$((k1 + 1)); fi
                            if [[ $TIME_SERIES_SURFACE1  -eq 1 ]]; then; TIME_SERIES_SURFACE[$k2]=$var;  k2=$((k2 + 1)); fi
                            if [[ $GRAFICOS_WRF1         -eq 1 ]]; then; GRAFICOS_WRF[k3]=$var;          k3=$((k3 + 1)); fi
                            if [[ $INTERATIVEMAP1  -eq 1 ]]; then; INTERATIVEMAP[k4]=$var;   k4=$((k4 + 1)); fi

                        fi
                    done

                    echo "    + SYSTEM - (TIME_SERIES_SOUNDING) - variaveis escolhidas               = ${TIME_SERIES_SOUNDING[@]}"
                    echo "    + SYSTEM - (TIME_SERIES_SURFACE)  - variaveis escolhidas               = ${TIME_SERIES_SURFACE[@]}"
                    echo "    + SYSTEM - (GRAFICOS_WRF)         - variaveis escolhidas               = ${GRAFICOS_WRF[@]}"
                    echo "    + SYSTEM - (INTERATIVEMAP)        - variaveis escolhidas               = ${INTERATIVEMAP[@]}"

                fi


#SYSTEM - processa_wrf_mmif
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


                if [[ $EXECUTAR_processa_wrf_mmif = 'S' ]]; then

                    if [[ $MMIF = "S" ]];then

                        echo " "
                        echo '    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
                        echo "    + SYSTEM - GERANDO ARQUIVOS DO $VERSAO_MMIF POR $(($deltahora+$timezone)) HORAS DA LOCALIDADE${ii}: $nome_estacao"
                        echo '    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
                        echo " "

                        #executando o MMIF
                        #-----------------
                        #definindo o vetor de variaveis para passagem para o processa_wrf_mmif
                        #        0                 1            2           3         4          5             6     7     8         9        10
                        vetmmif=($processa_wrf_wap $VERSAO_MMIF $WRFoutput1 $yyyymmdd $deltaHour $nome_estacao $lat1 $lon1 $timezone $TSgrade $MMIF $projeto)

                        echo "    + SYSTEM - para executar o MMIF:">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
                        echo "    + SYSTEM - ./$processa_wrf_mmif ${vetmmif[@]}">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
                        echo "    + SYSTEM - ----------------------------">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
                        echo "    + SYSTEM - passando ${#vetmmif[*]} variaveis para o $processa_wrf_mmif"
                        $dir_local/$processa_wrf_mmif ${vetmmif[@]}

                    fi

                else

                    echo "    + SYSTEM - MMIF                                                        = nao executado"

                fi


#SYSTEM - processa_wrf_readwrfnc
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


                if [[ $EXECUTAR_processa_wrf_readwrfnc == "S" ]]; then

                    if [[ $time_serie == "S" ]]; then


                        echo " "
                        echo '    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
                        echo "    + SYSTEM - PROCESSANDO EXTRACAO DE DADOS PARA A SERIE TEMPORAL"
                        echo '    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
                        echo " "

                        #executando o readwrfnc
                        #----------------------
                        #definindo o vetor de variaveis para passagem para o processa_wrf_readwrfnc
                        #             0           1         2             3     4     5         6      7               8     9                           10                         11                         12
                        vetreadwrfnc=($WRFoutput1 $yyyymmdd $nome_estacao $lat1 $lon1 $timezone $TSgrade $h_estacao $projeto ${#TIME_SERIES_SOUNDING[*]} ${#TIME_SERIES_SURFACE[*]} ${TIME_SERIES_SOUNDING[@]} ${TIME_SERIES_SURFACE[@]})
                        echo "    + SYSTEM - para executar o READ_WRF_NC:">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
                        echo "    + SYSTEM - ./$processa_wrf_readwrfnc ${vetreadwrfnc[@]}">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
                        echo "    + SYSTEM - ----------------------------">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
                        echo "    + SYSTEM - passando ${#vetreadwrfnc[*]} variaveis para o $processa_wrf_readwrfnc"
                        $dir_local/$processa_wrf_readwrfnc ${vetreadwrfnc[@]}

                    fi

                else

                    echo "    + SYSTEM - READWRFNC                                                   = nao executado"

                fi


#SYSTEM - processa_wrf_interativemap
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


                if [[ $EXECUTAR_processa_wrf_interativemap == "S" ]]; then

                    if [[ $IntMap == "S" ]]; then

                        echo " "
                        echo '    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
                        echo "    + SYSTEM - GERANDO O CONJUNTO DE FIGURAS PARA O INTERATIVE MAP"
                        echo '    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
                        echo " "

                        #executando o IntMap
                        #--------------------
                        k1=$((k1 + 1))
                        HORAS_simuladas=$((deltahora-2))

                        #definindo o vetor de variaveis para passagem de dados
                        #           0                     1           2         3             4
                        vetintmap=($processa_wrf_geoJSON  $WRFoutput1 $yyyymmdd $num_max_dom  ${INTERATIVEMAP[@]})


                        echo "    + SYSTEM - para executar o INTERATIVE MAP:">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
                        echo "    + SYSTEM - ./$processa_wrf_interativemap ${vetintmap[@]}">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
                        echo "    + SYSTEM - ----------------------------">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
                        echo "    + SYSTEM - passando ${#vetintmap[*]} variaveis para o $processa_wrf_interativemap"
                        $dir_local/$processa_wrf_interativemap ${vetintmap[@]}

                    fi

                else

                    echo "    + SYSTEM - IntMap                                                      = nao executado"

                fi


#SYSTEM - processa_wrf_python
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


                if [[ $EXECUTAR_processa_wrf_python = 'S' ]]; then #nessa condicao bloqueia a geracao de figuras para todos

                    if [[ $pyFig = 'S' ]]; then                    #nessa condicao somente bloqueia a geracao de figuras para definidos em localidadexx no namelist.processa

                        echo " "
                        echo '    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
                        echo "    + SYSTEM - GERANDO AS FIGURAS COM PYTHON DO wrf${ii}: $nome_estacao - veja o namelist.processa"
                        echo '    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
                        echo " "

                        echo "    + SYSTEM - gerando as figuras em $WRFoutput1"
                        echo "    + SYSTEM - para executar o PYTHON:">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug

                        #definindo o vetor de variaveis para passagem para o processa_wrf_python
                        #          0                     1         2           3       4       5
                        vetpython=($processa_wrf_figuras $yyyymmdd $WRFoutput1 $DOMini $DOMfim ${GRAFICOS_WRF[@]}) #fazendo as figuras de grade em grade pois o python consome muita memoria
                        echo "    + SYSTEM - ./$processa_wrf_python  ${vetpython[@]}">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
                        echo "    + SYSTEM - ----------------------------">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
                        echo "    + SYSTEM - passando ${#vetpython[*]} variaveis para o $processa_wrf_python"
                        #chama o processa_wrf_python
                        ${dir_local}/${processa_wrf_python}  ${vetpython[@]}

                    fi

                else

                    echo "    + SYSTEM - PYTHON                                                      = nao executado"

                fi


#SYSTEM - processa_wrf_ftp
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


                #executando o ftp dos arquivos
                #-----------------------------
                if [[ $EXECUTAR_processa_wrf_ftp = 'S' ]]; then

                    if [[ $exe_ftp = "S" ]]; then

                        echo "    + SYSTEM - enviando as figuras para a area ftp"

                        #definindo o vetor de variaveis para passagem para o processa_wrf_ftpt
                        #       0           1
                        vetftp=($WRFoutput1 $projeto)
                        echo "    + SYSTEM - para executar o FTP:">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
                        echo "    + SYSTEM - ./$processa_wrf_ftp ${vetftp[@]}">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
                        echo "    + SYSTEM - ----------------------------">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
                        echo "    + SYSTEM - passando ${#vetftp[*]} variaveis para o $processa_wrf_ftp"
                        $dir_local/$processa_wrf_ftp ${vetftp[@]}

                    fi

                else

                    echo "    + SYSTEM - FTP para o SITE                                             = nao executado"

                fi

            done


            cd $dir_local

            echo " "
            echo "    + SYSTEM - FIM DA SIMULACAO wrf${ii} - NOVA COMECA EM 5 SEGUNDOS"
#             for (( i=5; i>0; i--)); do; sleep 1 &; printf "    + SYSTEM - aguarde por $i segundos... ** OU presione Ctrl+C para cancelar\r"; wait; done
            echo " "

        fi
        
    done


#SYSTEM - processa_wrf_ftp
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

    #executando o deleta files
    #-------------------------
    #definindo o vetor de variaveis para passagem para o processa_wrf_deletafiles
    #               0

    if [[ $EXECUTAR_processa_wrf_deletafiles = 'S' ]]; then

        vetdeletafiles=($yyyymmdd)
        echo "    + SYSTEM - para executar o DELETAFILES:">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
        echo "    + SYSTEM - ./$processa_wrf_deletafiles ${vetdeletafiles[@]}">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
        echo "    + SYSTEM - ----------------------------">>$dir_local/d-log/${yyyymmdd}-wrf-op.debug
        echo "    + SYSTEM - passando ${#vetdeletafiles[*]} variaveis para o $processa_wrf_deletafiles"

        echo " "
        echo "    + SYSTEM - remocao dos arquivos anteriores a 30 dias em d-output       = S"

        $dir_local/$processa_wrf_deletafiles ${vetdeletafiles[@]}
    
    fi

    cd $dir_local

    \rm -f $namelist

    echo " "
    echo '    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
    echo "    + SYSTEM - FIM DO SCRIPT                                                                                    "
    echo '    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
    echo " "
    exit
