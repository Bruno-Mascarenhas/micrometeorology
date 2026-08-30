# Datasets abertos de imagem de céu

Catálogo dos candidatos a fonte de pré-treino para o modelo all-sky desta
estação, ordenado por proximidade ao problema daqui — **Salvador, 13,0°S,
tropical úmido**, estimação de DHI a partir de imagem fisheye.

A fonte do levantamento é Nie et al. (2024), *Open-source sky image datasets for
solar forecasting with deep learning: a comprehensive survey*, Renewable and
Sustainable Energy Reviews 189, arXiv:2211.14709, que cataloga **72 datasets
abertos**, dos quais 45 trazem irradiância medida. Os tamanhos e coberturas
abaixo vêm da Tabela 6 desse survey.

Nenhum dos listados exige pagamento.

| # | dataset | local | latitude | cobertura | tamanho | irradiância | câmera | acesso |
|---|---|---|---|---|---|---|---|---|
| 1 | ARM-TWP Darwin | Austrália | **12,4°S** | 12,5 anos | 222 GB | GHI/DNI/DHI 1 min | TSI | [arm.gov](https://www.arm.gov/) — conta gratuita |
| 2 | ARM-GoAmazon | Manacapuru, Brasil | 3,2°S | 1,9 ano | 54 GB | GHI/DNI/DHI 1 min | TSI | [arm.gov](https://www.arm.gov/) |
| 3 | ARM-TWP Manus / Nauru | PNG / Nauru | 2°S / 0,5°S | 10,5 / 10,8 anos | 144 / 177 GB | GHI/DNI/DHI 1 min | TSI | [arm.gov](https://www.arm.gov/) |
| 4 | ARM-LASIC | Ilha de Ascensão | 8°S | 1,5 ano | 37 GB | GHI/DNI/DHI 1 min | TSI | [arm.gov](https://www.arm.gov/) |
| 5 | SKIPP'D | Stanford, EUA | 37,4°N | 2,7 anos | 1.700 GB | potência PV | fisheye | [purl.stanford.edu/dj417rh1007](https://purl.stanford.edu/dj417rh1007) |
| **6** | **UCSD-Folsom** | Califórnia, EUA | 38,6°N | 3,0 anos | 50 GB | GHI/DNI/DHI 1 min | **fisheye** | [zenodo.org/records/2826939](https://zenodo.org/records/2826939) |
| 7 | SRRL-BMS | Golden, EUA | 39,7°N | décadas | — | GHI/DNI/DHI 1 min | TSI / ASI | [midcdmz.nrel.gov](https://midcdmz.nrel.gov/apps/sitehome.pl?site=BMS) |
| 8 | SIRTA | Palaiseau, França | 48,7°N | anos | — | GHI/DNI/DHI | fisheye | [sirta.ipsl.fr](https://sirta.ipsl.fr/) — cadastro |
| — | SIPM | Rio de Janeiro | 22,9°S | 26 dias, 0,3 GB | — | — | — | pequeno demais para pré-treino |

## O compromisso que decide a escolha

Os sítios da ARM usam **TSI** — espelho hemisférico com uma faixa de sombra que
**oculta o sol** para não saturar o sensor. Folsom, SKIPP'D e SIRTA usam fisheye
direto, como a câmera desta estação.

Isso pesa mais que o clima para esta tarefa. A região circunsolar é o que governa
a partição difusa (Perez et al. 1990), e o braço `sunangle` mediu que dar à rede
a posição do sol derruba a amplitude do viés por banda de elevação de 17,47 para
2,65 W/m². Pré-treinar em imagens onde o sol está tapado ensina o backbone a ler
um céu sem a feição que mais importa aqui.

Daí as duas apostas serem diferentes:

- **clima certo, câmera errada** — ARM-TWP Darwin, 12,4°S e 12,5 anos, o regime
  solar quase idêntico ao daqui;
- **câmera certa, clima errado** — UCSD-Folsom, fisheye com sol visível, DHI
  medido, e o dataset que o benchmark de Varaschin & Silva (2025) usa, o que dá
  comparabilidade com a literatura.

**Escolhido para começar: Folsom.** A transferência move o backbone visual, e o
que ele precisa aprender é a aparência do céu com o sol dentro do quadro.
Diferença de clima o fine-tune nos 55 dias locais corrige; diferença de geometria
óptica, não.

Isto é raciocínio a partir do que foi medido aqui, não resultado medido. Decidir
de verdade exige pré-treinar nos dois e comparar, e fica barato depois que a
ingestão do primeiro existir.

## UCSD-Folsom, o que foi baixado

Zenodo DOI [10.5281/zenodo.2826939](https://doi.org/10.5281/zenodo.2826939),
licença CC BY-NC 4.0 — gratuita, uso não comercial. Citar Pedro, Larson &
Coimbra (2019), *A comprehensive dataset for the accelerated development and
benchmarking of solar forecasting methods*, Journal of Renewable and Sustainable
Energy 11(3), 036102.

Do conjunto completo interessam três arquivos:

| arquivo | tamanho | por que |
|---|---|---|
| `Folsom_irradiance.csv` | 76,5 MB | GHI, DNI e **DHI** a 1 min, o alvo |
| `Folsom_weather.csv` | 138,8 MB | vento, para as features do tier `bare` |
| `Folsom_sky_images_2014.tar.bz2` | 13,8 GB | as imagens; 2015 e 2016 somam mais 35,5 GB |

O resto do registro — features pré-extraídas, satélite, NAM, alvos de previsão e
os scripts de forecast — não serve a este uso: a extração de feature aqui é a
nossa, e os alvos deles são de previsão com horizonte, não de estimação em t=0.

Medido no `Folsom_irradiance.csv`: **1.552.320 linhas**, de 2014-01-02 08:00 a
2016-12-31 07:59 UTC, cadência de 1 min, 618 NaN em `dhi` e **732.122 linhas com
GHI > 20 W/m²**. Contra as 46.014 linhas em 81 dias desta estação, um ano de
Folsom já é uma ordem de grandeza a mais.
