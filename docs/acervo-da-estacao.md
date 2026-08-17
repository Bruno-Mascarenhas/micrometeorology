# O acervo da estação — que arquivo, que sensor, que período

Este documento existe para que ninguém precise refazer a arqueologia do acervo.
Ele responde uma pergunta: **para a variável X no período Y, qual arquivo `.dat`
do LBM é a fonte e qual coluna daquele arquivo carrega a medida?**

Tudo aqui foi gerado a partir de três fontes do próprio repositório e verificado
contra o dado, não transcrito à mão:

- o manifesto explícito em `sensors/archive.py` (`LENTA_MANIFEST`, `RAIN_MANIFEST`);
- o bloco `sensor_switches` de `configs/micromet/calibrations.yaml`, que é a
  autoridade sobre qual coluna bruta carrega cada variável e quando;
- a medição de `output/archive/station_5min_qc.parquet`, para a cobertura real.

Os intervalos são de carimbo **local ingênuo**, como o registrador escreveu.

---

## Por que existe um manifesto, e não um glob

Ler `data/dados-labmim/*.dat` produz um registro errado de quatro maneiras, todas
medidas numa auditoria de cada tabela do acervo:

1. **`*.dat` descarta os arquivos de rotação.** Três tabelas `.backup` são a
   ÚNICA fonte de um inverno austral cada — JJA 2020, JJA 2022, e junho a meados
   de julho de 2024.
2. **O diretório guarda mais de uma estação.** `BTS_*` é outro sítio (CR1000X
   série 9429), as tabelas `celsolar` e `calibracao` são campanhas de instrumento
   em paralelo, e as famílias `solar` e `radiacao` amostram a cada minuto.
3. **Os nomes mentem.** `dados-labmim/LBM_lenta.dat` é a tabela de CHUVA — o campo
   8 do cabeçalho TOA5 diz `LBM_rain` — e é a fonte única de fevereiro de 2019.
4. **Três defeitos de relógio não cabem em configuração.** Precisam dos bytes
   corrigidos antes do merge, o que `stage_archive` faz num diretório de rascunho:
   nada aqui jamais escreve em `data/`.

---

## A cronologia por variável

Cada célula é a coluna bruta que carrega aquela variável naquele período. Um
travessão significa que a variável **não existe** no acervo ali — não que esteja
ausente por falha, mas que nenhum instrumento a media.

#### superficie

| de | T | ur | Td | pressure | WS | WD | precip |
|---|---|---|---|---|---|---|---|
| 2016-09-29 a 2019-03-15 | `Temp1_Avg` | `RH1_Avg` | — | — | `WS_ms_S_WVT` | `WindDir_D1_WVT` | `PL01_mm_Tot` |
| 2019-03-15 a 2019-03-18 | `Temp1_Avg` | `RH1_Avg` | — | `AirPressure` | `WS_ms_S_WVT` | `WindDir_D1_WVT` | `PL01_mm_Tot` |
| 2019-03-18 a 2019-03-18 | `Temp1_Avg` | `RH1_Avg` | — | — | `WS_ms_S_WVT` | `WindDir_D1_WVT` | `PL01_mm_Tot` |
| 2019-03-18 a 2019-05-31 | `Temp1_Avg` | `RH1_Avg` | — | `Pmb_WXT_Avg` | `WS_ms_S_WVT` | `WindDir_D1_WVT` | `PL01_mm_Tot` |
| 2019-05-31 a 2019-05-31 | `Temp1_Avg` | `RH1_Avg` | — | `Pmb_WXT_Avg` | — | — | `PL01_mm_Tot` |
| 2019-05-31 a 2019-06-10 | `Temp1_Avg` | `RH1_Avg` | — | `Pmb_WXT_Avg` | `WS_WXT_Avg` | `WD_WXT_Avg` | `PL01_mm_Tot` |
| 2019-06-10 a 2023-02-20 | `Temp1_Avg` | `RH1_Avg` | — | `Pmb_WXT` | `WS_WXT_Avg` | `WD_WXT_Avg` | `PL01_mm_Tot` |
| 2023-02-20 a 2023-03-10 | `Temp1_Avg` | `RH1_Avg` | — | `Pmb_WXT` | — | — | `PL01_mm_Tot` |
| 2023-03-10 a 2024-07-19 | `Temp1_Avg` | `RH1_Avg` | — | — | — | — | `PL01_mm_Tot` |
| 2024-07-19 a 2025-03-12 | `Temp1_Avg` | `RH1_Avg` | — | — | `WS_ms` | `WindDir` | `PL01_mm_Tot` |
| 2025-03-12 a 2025-03-19 | — | — | — | — | `WS_ms` | `WindDir` | `PL01_mm_Tot` |
| 2025-03-19 a 2025-03-28 | `AirT_C_Avg` | `RH` | `DP_C_Avg` | `BP_mbar_Avg` | `WS_ms` | `WindDir` | `PL01_mm_Tot` |
| 2025-03-28 em diante | `AirT1_C_Avg` | `RH1` | `DP1_C_Avg` | `BP1_mbar_Avg` | `WS_ms` | `WindDir` | `PL01_mm_Tot` |

#### radiacao

| de | Sw_dw | Sw_dif | Sw_up | Lw_dw | Lw_up | Sw_par | Sw_uv |
|---|---|---|---|---|---|---|---|
| 2016-09-29 a 2018-08-20 | `PSP1_Wm2_Avg` | — | — | `PIR1_Wm2_Avg` | — | `PAR_Wm2_Avg` | — |
| 2018-08-20 a 2018-08-21 | — | — | — | `PIR1_Wm2_Avg` | — | `PAR_Wm2_Avg` | — |
| 2018-08-21 a 2018-10-16 | — | — | — | — | — | `PAR_Wm2_Avg` | — |
| 2018-10-16 a 2018-11-13 | `CM3Up_Wm2_Avg` | — | `CM3Dn_Wm2_Avg` | `CG3Up_Wm2_cor_Avg` | `CG3Dn_Wm2_cor_Avg` | `PAR_Wm2_Avg` | — |
| 2018-11-13 a 2019-02-26 | `CM3Up_Wm2_Avg` | `PSP1_Wm2_Avg` | `CM3Dn_Wm2_Avg` | `CG3Up_Wm2_cor_Avg` | `CG3Dn_Wm2_cor_Avg` | `PAR_Wm2_Avg` | — |
| 2019-02-26 a 2019-03-15 | `CM3Up_Wm2_Avg` | — | `CM3Dn_Wm2_Avg` | — | — | `PAR_Wm2_Avg` | — |
| 2019-03-15 a 2019-03-18 | `CM3Up_Wm2_Avg` | — | `CM3Dn_Wm2_Avg` | `CG3Up_Wm2_Corr_Avg` | `CG3Dn_Wm2_Corr_Avg` | `PAR_Wm2_Avg` | — |
| 2019-03-18 a 2019-03-19 | `CM3Up_Wm2_Avg` | `PSP_Wm2_Avg` | `CM3Dn_Wm2_Avg` | `CG3Up_Wm2_Corr_Avg` | `CG3Dn_Wm2_Corr_Avg` | `PAR_Wm2_Avg` | — |
| 2019-03-19 a 2019-03-19 | `CM3Up_Wm2_Avg` | `PSP_Wm2_Avg` | `CM3Dn_Wm2_Avg` | — | — | `PAR_Wm2_Avg` | — |
| 2019-03-19 a 2019-08-31 | `CM3Up_Wm2_Avg` | `PSP_Wm2_Avg` | `CM3Dn_Wm2_Avg` | `CG3Up_Wm2Cr_Avg` | `CG3Dn_Wm2Cr_Avg` | `PAR_Wm2_Avg` | — |
| 2019-08-31 a 2019-10-01 | `CM3Up_Wm2_Avg` | — | `CM3Dn_Wm2_Avg` | `CG3Up_Wm2Cr_Avg` | `CG3Dn_Wm2Cr_Avg` | `PAR_Wm2_Avg` | — |
| 2019-10-01 a 2020-03-06 | `CM3Up_Wm2_Avg` | `CMP21_Wm2_Avg` | `CM3Dn_Wm2_Avg` | `CG3Up_Wm2Cr_Avg` | `CG3Dn_Wm2Cr_Avg` | `PAR_Wm2_Avg` | — |
| 2020-03-06 a 2020-06-01 | `CM3Up_Wm2_Avg` | — | `CM3Dn_Wm2_Avg` | `CG3Up_Wm2Cr_Avg` | `CG3Dn_Wm2Cr_Avg` | `PAR_Wm2_Avg` | — |
| 2020-06-01 a 2025-03-12 | `CM3Up_Wm2_Avg` | `CMP21_Wm2_Avg` | `CM3Dn_Wm2_Avg` | `CG3Up_Wm2Cr_Avg` | `CG3Dn_Wm2Cr_Avg` | `PAR_Wm2_Avg` | — |
| 2025-03-12 a 2025-03-19 | `CM3Up_Wm2_Avg` | — | `CM3Dn_Wm2_Avg` | `CG3Up_Wm2Cr_Avg` | `CG3Dn_Wm2Cr_Avg` | `PAR_Wm2_Avg` | — |
| 2025-03-19 a 2025-05-14 | `CM3Up_Wm2_Avg` | — | `CM3Dn_Wm2_Avg` | `CG3Up_Wm2Cr_Avg` | `CG3Dn_Wm2Cr_Avg` | `PAR_Wm2_Avg` | `CUV5_Wm2_Avg` |
| 2025-05-14 em diante | `CM3Up_Wm2_Avg` | `PSP_Wm2_Avg` | `CM3Dn_Wm2_Avg` | `CG3Up_Wm2Cr_Avg` | `CG3Dn_Wm2Cr_Avg` | `PAR_Wm2_Avg` | `CUV5_Wm2_Avg` |

#### saldo

| de | Net_CNR1 | Net_NRLite | Tbody | qc_flag |
|---|---|---|---|---|
| 2016-09-29 a 2018-08-20 | — | `NRLite_Wm2_Corr_Avg` | `T_C1_Avg` | — |
| 2018-08-20 a 2018-10-16 | — | — | `T_C1_Avg` | — |
| 2018-10-16 a 2019-03-15 | — | — | `CNR1TK_Avg` | — |
| 2019-03-15 a 2019-03-19 | — | `NRLite_Wm2_Avg` | `CNR1TK_Avg` | — |
| 2019-03-19 a 2022-04-11 | `Net_Wm2_Avg` | `NRLite_Wm2_Avg` | `CNR1TK_Avg` | — |
| 2022-04-11 a 2025-03-19 | `Net_Wm2_Avg` | — | `CNR1TK_Avg` | — |
| 2025-03-19 a 2025-03-28 | `Net_Wm2_Avg` | — | `CNR1TK_Avg` | `MetSENS_Status` |
| 2025-03-28 em diante | `Net_Wm2_Avg` | — | `CNR1TK_Avg` | `MetSENS1_Status` |
### As três lacunas que decidem um recorte

Ler a matriz acima de cima para baixo mostra que a estação não mediu tudo o tempo
todo. Três buracos são grandes o bastante para invalidar um recorte inteiro se
passarem desapercebidos:

- **Pressão, 740 dias sem dado** — o `Pmb_WXT` da Vaisala termina em
  2023-03-10 e nenhum barômetro entra até o Gill em 2025-03-19. A cobertura da
  variável no acervo é de **71,7%**, a pior de todas.
- **Vento, 515 dias** — a sônica WXT morre em 2023-02-20 e o `WS_ms` só aparece
  em 2024-07-19. Vale igual para velocidade e direção, que compartilham o
  instrumento.
- **Ponto de orvalho, 66,5% de cobertura** — não existe instrumento de orvalho
  antes do Gill MetSENS, em 2025. Toda série de `Td` anterior a isso é ausência
  de sensor, não falha de aquisição.

| variável | amostras | cobertura | maior lacuna | a lacuna começa em |
|---|---|---|---|---|
| `Lw_dw` | 975,129 | 93.9% | 57 dias | 2018-08-20 23:55 |
| `Lw_up` | 785,257 | 95.3% | 17 dias | 2019-02-26 09:30 |
| `Net_CNR1` | 783,548 | 95.1% | 17 dias | 2019-02-26 09:30 |
| `Net_NRLite` | 481,458 | 82.8% | 206 dias | 2018-08-20 23:55 |
| `Sw_dif` | 720,751 | 88.3% | 87 dias | 2020-03-05 23:55 |
| `Sw_dw` | 986,659 | 95.0% | 57 dias | 2018-08-20 23:55 |
| `Sw_par` | 933,288 | 89.8% | 113 dias | 2017-05-18 10:00 |
| `Sw_up` | 798,416 | 96.9% | 17 dias | 2019-02-26 09:30 |
| `Sw_uv` | 142,724 | 96.3% | 10 dias | 2026-04-14 05:45 |
| `T` | 906,939 | 89.8% | 112 dias | 2024-11-27 05:45 |
| `Tbody` | 988,815 | 95.2% | 56 dias | 2018-08-21 08:25 |
| `Td` | 79,292 | 66.5% | 67 dias | 2025-12-21 18:50 |
| `WD` | 861,533 | 82.9% | 515 dias | 2023-02-20 13:30 |
| `WS` | 837,110 | 80.6% | 515 dias | 2023-02-20 13:30 |
| `precip` | 996,977 | 96.0% | 78 dias | 2024-02-12 23:55 |
| `pressure` | 558,248 | 71.7% | 740 dias | 2023-03-10 06:10 |
| `qc_flag` | 142,759 | 96.3% | 10 dias | 2026-04-14 05:45 |
| `ur` | 907,061 | 93.5% | 112 dias | 2024-11-27 05:50 |
A cobertura é medida contra a grade de 5 minutos entre a primeira e a última
amostra de cada variável, no frame com controle de qualidade aplicado — ou seja,
já descontando o que o QC removeu. `docs/controle-de-qualidade.md` descreve o que
cada etapa remove e por quê.

---

## Armadilhas que já custaram tempo

Estas são as razões pelas quais a matriz acima não pode ser reconstruída por
inspeção de nomes de coluna.

**Um nome novo não significa um sensor novo.** `PSP1_Wm2_Avg` passa a se chamar
`PSP_Wm2_Avg` na troca de programa v11, em 2019-03-15, e é o mesmo piranômetro: o
multiplicador mV→W/m² programado é 119,474 = 1000/8,37 sob as duas grafias, em
todos os anos. Da mesma forma `CG3Up_Wm2_cor_Avg` → `CG3Up_Wm2_Corr_Avg` →
`CG3Up_Wm2Cr_Avg` são três grafias do mesmo canal corrigido.

**Um nome quase igual pode ser outro sensor.** `RH1_Avg` é o HMP; `RH1` é a
unidade 1 do Gill MetSENS. São instrumentos diferentes, separados por seis anos, e
a diferença de nome é um sufixo.

**O token de agregação muda no meio.** Em 2025-03-19 a umidade passa de `Avg` para
`Smp` — de média do intervalo para amostra instantânea. O nome da variável
unificada não muda, mas a grandeza sim.

**Colunas em kelvin que parecem ser outra coisa.** `T_C1_Avg` e `T_D1_Avg` na era
v4/v9 são os termistores de CASE e DOME do Eppley PIR, em kelvin (294,7 a 309,0 K,
correlação 0,9955 entre si) — não são temperatura do ar nem ponto de orvalho.

**Duas unidades em paralelo por um mês e meio.** Entre 2025-03-28 e 2025-05-14 as
duas unidades do Gill MetSENS registram simultaneamente (`AirT1_C_Avg` e
`AirT2_C_Avg`). A unidade 1 foi escolhida por continuidade; a 2 existe no acervo e
não entra em nenhuma variável unificada.

**Sentinelas que não são NaN.** A umidade escreve `0.0` de verdade entre
2018-08-27 e 2018-10-16, e `-100` entre 2024-11-27 e 2025-03-12; a temperatura
escreve `-100` na mesma segunda janela e `1000` a partir de 2025-12-13. Nenhum
deles é um valor físico e nenhum é ausente — é por isso que `mask_sentinels`
existe e roda antes de qualquer outra coisa.

---

## Os arquivos, medidos

Duas tabelas do registrador, na mesma grade de 5 minutos, unidas por JOIN e não
por concatenação. Os intervalos abaixo foram lidos de cada arquivo, não das notas.

#### tabela lenta

| arquivo | de | até | linhas | reparo | por que está no manifesto |
|---|---|---|---|---|---|
| `dados-labmim/LBM_lenta_2016.dat` | 2016-09-29 13:40 | 2016-12-31 23:55 | 25,102 | — | start of record, 2016-09-29 |
| `dados-labmim/LBM_lenta_2017.dat` | 2017-01-01 00:00 | 2017-12-31 23:55 | 103,106 | — | all of 2017, complete JJA |
| `dados-labmim/LBM_lenta_2018_1.dat` | 2018-01-01 00:05 | 2018-10-16 13:40 | 78,865 | — | 2018-01..2018-10-16, JJA 2018 |
| `dados-labmim/LBM_lenta_2018-2019.dat` | 2018-10-16 13:50 | 2019-02-26 09:30 | 38,152 | — | CNR1 commissioning era |
| `dados-labmim/LBM_lenta_2019.dat.backup` | 2019-03-15 11:20 | 2019-03-15 15:55 | 56 | — | sole source of 2019-03-15 afternoon |
| `dados-labmim/LBM_lenta_2019.dat.1.backup` | 2019-03-15 17:05 | 2019-03-18 09:05 | 769 | — | sole source of 2019-03-15..18 |
| `dados-labmim/LBM_lenta_2019.dat.2.backup` | 2019-03-18 12:55 | 2019-03-19 08:25 | 232 | — | sole source of 2019-03-18..19, WXT arrives |
| `dados-labmim/LBM_lenta_2019.dat.3.backup` | 2019-03-19 10:05 | 2019-03-20 15:00 | 348 | — | sole source of 2019-03-19..05-31 |
| `dados-labmim/LBM_lenta_2019_0531.dat` | 2019-03-20 15:55 | 2019-05-31 08:30 | 20,622 | — | 2019-05-31 onward |
| `dados-labmim/LBM_lenta_2019_0631.dat` | 2019-05-31 09:10 | 2019-06-10 15:30 | 2,951 | — | 2019-06 onward |
| `dados-labmim/LBM_lenta_2019_1011.dat` | 2019-06-10 15:35 | 2019-10-11 09:45 | 35,292 | — | 2019-10 onward, CMP21 diffuse begins |
| `dados-labmim/LBM_lenta_2019.dat` | 2019-10-01 00:00 | 2020-01-07 00:00 | 28,211 | `drop-late-tail` | 110-row tail is mis-stamped; the clock-fixed 2020_03 table carries it correctly |
| `dados-labmim/LBM_lenta_2020_03.dat` | 2020-01-01 00:00 | 2020-03-06 10:55 | 18,848 | `clock+1h` | headerless CSV, and 16855 rows are one hour early |
| `dados-labmim/LBM_lenta_2020.dat.backup` | 2020-03-06 11:00 | 2020-09-23 10:40 | 57,814 | — | SOLE SOURCE OF JJA 2020 |
| `dados-labmim/LBM_lenta_2020.dat` | 2020-09-23 11:05 | 2021-07-26 14:25 | 88,109 | — | rest of 2020 |
| `dados-labmim/LBM_lenta_2021.dat` | 2021-07-26 13:40 | 2022-04-11 10:30 | 74,472 | — | all of 2021 |
| `dados-labmim/LBM_lenta_2022.dat.backup` | 2022-04-11 10:40 | 2022-09-16 14:00 | 44,881 | — | SOLE SOURCE OF JJA 2022 |
| `dados-labmim/LBM_lenta_2022.dat` | 2022-09-16 14:05 | 2023-08-18 14:25 | 96,476 | — | rest of 2022 (superset of data/LBM_lenta_2022.dat) |
| `dados-labmim/CR5000_LBM_lenta_18-21082023.dat` | 2023-08-18 14:35 | 2023-08-21 09:25 | 803 | — | 2023-08 spare-logger block |
| `dados-labmim/LBM_lenta_2023.dat` | 2023-08-21 09:30 | 2024-03-14 17:00 | 59,389 | — | 2023 |
| `dados-labmim/LBM_lenta_2023_14032024.dat` | 2024-03-14 17:05 | 2024-03-18 09:00 | 1,056 | — | 2024-03 handover |
| `dados-labmim/LBM_lenta_2024.dat.backup` | 2024-03-18 09:05 | 2024-07-19 15:00 | 35,495 | — | SOLE SOURCE OF JUNE AND 1-19 JULY 2024 |
| `dados-labmim/LBM_lenta_2024.dat` | 2024-07-19 15:15 | 2025-03-12 13:20 | 67,931 | — | rest of 2024 |
| `dados-labmim/LBM_lenta_2025.dat.backup` | 2025-03-12 13:25 | 2025-03-19 11:15 | 1,991 | — | 2025-03 Gill MetSENS commissioning |
| `dados-labmim/LBM_lenta_2025.dat.1.backup` | 2025-03-19 11:20 | 2025-03-19 12:55 | 20 | — | 2025-03 commissioning |
| `dados-labmim/LBM_lenta_2025.dat.2.backup` | 2025-03-19 13:10 | 2025-03-19 13:25 | 4 | — | 2025-03 commissioning |
| `dados-labmim/LBM_lenta_2025.dat.3.backup` | 2025-03-19 14:00 | 2025-03-28 10:30 | 2,550 | — | 2025-03 commissioning |
| `dados-labmim/LBM_lenta_2025.dat.4.backup` | 2025-03-28 10:35 | 2025-05-14 15:20 | 13,579 | — | 2025-03-28..05-14, dual GMX units |
| `LBM_lenta_2025.dat` | 2025-05-14 15:25 | 2026-08-15 23:30 | 126,606 | — | v22 era to 2026-08-12; PSP takes over diffuse |

#### tabela rain

| arquivo | de | até | linhas | reparo | por que está no manifesto |
|---|---|---|---|---|---|
| `dados-labmim/LBM_rain_2016.dat` | 2016-09-29 13:40 | 2016-12-31 23:55 | 25,305 | — | start of rain record |
| `dados-labmim/LBM_rain_2017.dat` | 2017-01-01 00:05 | 2017-12-31 23:55 | 103,178 | — | 2017 |
| `dados-labmim/LBM_rain_2018_2019.dat` | 2018-01-01 00:05 | 2019-01-31 11:00 | 109,565 | — | 2018 into 2019 |
| `dados-labmim/LBM_lenta.dat` | 2019-01-31 11:30 | 2019-02-26 09:30 | 7,463 | — | MISNAMED: TOA5 field 8 is LBM_rain. Unique source of 2019-01-31..02-26 |
| `dados-labmim/LBM_rain_2019.dat` | 2019-03-15 11:20 | 2020-01-07 00:00 | 85,497 | `drop-late-tail` | same 110-row mis-stamped tail |
| `dados-labmim/LBM_rain_2020.dat` | 2020-01-01 00:00 | 2021-07-26 14:25 | 164,759 | — | 2020 (clock slip is in the lenta table, not here) |
| `dados-labmim/LBM_rain_2021.dat` | 2021-07-26 13:40 | 2022-04-11 10:30 | 74,470 | — | 2021 |
| `dados-labmim/LBM_rain_2022.dat` | 2022-04-11 10:40 | 2023-08-18 14:30 | 141,355 | — | 2022 (superset of data/LBM_rain_2022.dat) |
| `dados-labmim/CR5000_LBM_rain_18-21082023.dat` | 2023-08-18 14:35 | 2023-08-21 09:30 | 804 | `keep-2023-block` | only the 804-row 2023-08 block; 892 scattered pre-2016 rows are a spare logger |
| `dados-labmim/LBM_rain_2023.dat` | 2023-08-21 09:35 | 2024-03-14 17:00 | 59,386 | — | 2023 |
| `dados-labmim/LBM_rain2023_14032024.dat` | 2024-03-14 17:05 | 2024-03-18 09:00 | 1,056 | — | 2024-03 handover |
| `dados-labmim/LBM_rain_2024.dat` | 2024-03-18 09:05 | 2025-03-12 13:20 | 103,424 | — | 2024 |
| `LBM_rain_2025.dat` | 2025-03-12 13:25 | 2026-08-15 23:30 | 144,904 | — | 2025 to 2026-08-12 |
---

## Como usar isto

Para reproduzir qualquer número publicado, não leia os `.dat`: use o banco.

```bash
uv run labmim-archive -d data -o output/archive --strict
```

Isso escreve três artefatos e um `archive_report.json` que tabula o que cada
etapa removeu. `station_5min_raw.parquet` é imutável e traz os valores como o
registrador os escreveu, sentinelas inclusive; `station_5min_qc.parquet` é o
frame do qual as médias horárias saem; `station_hourly.parquet` é a agregação.

Para saber qual coluna bruta responde por uma variável num instante, a autoridade
é o código, não este documento:

```python
from micrometeorology.sensors.calibration import load_sensor_switches

switches = load_sensor_switches("configs/micromet/calibrations.yaml")
```

Este arquivo é uma leitura daquele bloco somada à medição do acervo. Se os dois
divergirem, o bloco está certo e este documento está velho.
