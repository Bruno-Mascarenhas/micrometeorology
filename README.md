# Micrometeorology LabMiM-UFBA

Os processos de interação superfície-atmosfera têm um importante papel no sistema climático terrestre, por meio das trocas de momento, calor, umidade e de outros constituintes, na atmosfera como um todo e, em particular, na Camada Limite Atmosférica (CLA). A micrometeorologia estuda os fenômenos meteorológicos que ocorrem na CLA, com escalas de tempo e espaço inferiores a 60 min e a 2 km, respectivamente.

Do ponto vista prático, as atividades humanas dependem e modificam as condições microclimáticas pelo uso intensivo do solo, desmatamento, queima de combustíveis fósseis, processos de urbanização, entre outros. A compreensão dos processos físicos que ocorrem na CLA, principalmente a turbulência atmosférica, contribue para o entendimento das condições ambientais de uma região.

Os Laboratórios Associados LabMiM e LMAC, dos Institutos de Física (IF) e de Matemática (IM) da Universidade Federal da Bahia (UFBA), tem como meta investigar os processos de interação superfície-oceano-atmosfera observados na Região Metropolitana de Salvador (RMS), por meio da previsão numérica de campos meteorológicos de superfície e do monitoramento de variáveis ambientais. 

Atualmente, o modelo atmosférico de mesoescala Weather Research and Forecasting (WRF) tem sido usado para realizar a previsão numérica do tempo, com alta resolução espaço-temporal, sobre a costa leste da região nordeste e a RMS. O modelo de simulação direta dos grade turbilhões (LES) é também utilizado pelo grupo de pesquisa para estimar as propriedades estatísticas da turbulência na CLA.

## Getting Started

You will need install those packages to run all scripts.

### Prerequisites

All scripts were made with Python 3.7

* [Python](https://www.python.org/downloads/)

Basemap
```
sudo pip3 install --user https://github.com/matplotlib/basemap/archive/master.zip
```

NetCDF4, Numpy, Pandas, Seaborn and Scikit-Learn
```
pip install matplotlib numpy pandas seaborn netcdf4 sklearn
```

## Built With

* [PyCharm](https://www.jetbrains.com/pycharm/) - IDE used to code faster and view data easier

## Authors

* **Bruno S Mascarenhas** - *Researcher* - [Bruno Mascarenhas](https://github.com/Bruno-Mascarenhas)
* **Edson P Marques Filho** - *Researcher, Advisor* - [Edson P. M. Filho](https://github.com/epmfilho)

See also the list of [contributors](https://github.com/bruno-mascarenhas/meteorology-ufba/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* UFBA
* CNPq
* FABESB
