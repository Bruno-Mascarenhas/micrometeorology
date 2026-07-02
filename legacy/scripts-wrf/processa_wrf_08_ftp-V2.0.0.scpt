#!/bin/ksh

echo " "
echo '    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
echo "    + FTP - EXECUTANDO O FTP"
echo '    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
echo " "

#        0          1
#vetftp=($WRFoutput $projeto)

#local de trabalho
#-----------------
dir_local=$( grep dir_local /home/models/WRF/wrf-op/.FUNCTIONS_WRF_environment | awk '{print $2}' )
export PATH=$PATH:${dir_local}:.

#lendo as variaveis passadas pelo SYSTEM
#---------------------------------------
vetftp=("$@") #recebendo o vetor das variaveis

#pegando dados passados por processa_wrf_00.scpt
WRFoutput=${vetftp[0]}
projeto=${vetftp[1]}



if [[ $projeto = "PETROBRAS" ]];then

echo '    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
echo "    + FTP - $projeto"
echo '    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
echo " "


#/public_html/files/cenpes/aermod
echo "    + FTP - sending data from $WRFoutput to /public_html/files/cenpes/aermod. Wait..."
cd $WRFoutput

lftp -u leal,${FTP_PASSWORD} webhost.ufes.br << EOF
set ssl:verify-certificate no
cd ~/public_html/files/cenpes/aermod
mput *.zip && echo '    + FTP - upload files data success!'
mput lista.json && echo '    + FTP - upload lista.json success!'
bye
EOF

fi



if [[ $projeto = "IFES" ]];then

echo '    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
echo "    + FTP - $projeto"
echo '    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
acho " "

#/public_html/files/ifes/aermod
echo "    + FTP - sending data from $WRFoutput to /public_html/files/ifes/aermod. Wait..."
cd $WRFoutput


lftp -u leal,${FTP_PASSWORD} webhost.ufes.br << EOF
set ssl:verify-certificate no
cd ~/public_html/files/ifes/aermod
mput *.zip && echo '    + FTP - upload files success!'
mput lista.json && echo '    + FTP - upload lista.json success!'
bye
EOF

fi



if [[ $projeto = "LEAL" ]];then

echo '    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
echo "    + FTP - $projeto"
echo '    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
echo " "


#/public_html/videos
echo "    + FTP - sending data from $WRFoutput to /public_html/videos. Wait..."
cd $WRFoutput


lftp -u leal,${FTP_PASSWORD} webhost.ufes.br << EOF
set ssl:verify-certificate no
cd ~/public_html/videos
mput *.webm && echo '    + FTP - upload files success!'
bye
EOF



#/public_html/interative_maps
path_maps=$dir_local/d-interative-map/wrf-map-utils
echo "    + FTP - sending data from $path_maps to /public_html. Wait..."
cd $path_maps


lftp -u leal,${FTP_PASSWORD} leal.ufes.br << EOF
set ssl:verify-certificate no
cd ~/public_html
mput mapas_interativos_es.html && echo '    + FTP - upload files success!'
bye
EOF



#/public_html/interative_maps/JSON
path_json=$path_maps/JSON
echo "    + FTP - sending data from $path_json to /public_html/JSON"
cd $path_json


lftp -u leal,${FTP_PASSWORD} leal.ufes.br << EOF
set ssl:verify-certificate no
cd ~/public_html/JSON
mput * && echo '    + FTP - upload files success!'
bye
EOF


#/public_html/interative_maps/css
path_css=$path_maps/css
echo "    + FTP - sending data from $path_css to /public_html/css"
cd $path_css


lftp -u leal,${FTP_PASSWORD} leal.ufes.br << EOF
set ssl:verify-certificate no
cd ~/public_html/css
mput * && echo '    + FTP - upload files success!'
bye
EOF



#/public_html/interative_maps/geoJSON
path_geojson=$path_maps/geoJSON
echo "    + FTP - sending data from $path_geojson to /public_html/geoJSON"
cd $path_geojson


lftp -u leal,${FTP_PASSWORD} leal.ufes.br << EOF
set ssl:verify-certificate no
cd ~/public_html/geoJSON
mput * && echo '    + FTP - upload files success!'
bye
EOF



#/public_html/interative_maps/images
path_images=$path_maps/images
echo "    + FTP - sending data from $path_images to /public_html/images"
cd $path_images


lftp -u leal,${FTP_PASSWORD} leal.ufes.br << EOF
set ssl:verify-certificate no
cd ~/public_html/images
mput * && echo '    + FTP - upload files success!'
bye
EOF



#/public_html/interative_maps/js
path_js=$path_maps/js
echo "    + FTP - sending data from $path_js to /public_html/js"
cd $path_js

lftp -u leal,${FTP_PASSWORD} leal.ufes.br << EOF
set ssl:verify-certificate no
cd ~/public_html/js
mput * && echo '    + FTP - upload files success!'
bye
EOF

fi

#retorn to previous directory
cd ..
