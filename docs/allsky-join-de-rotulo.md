# O join de rótulo do all-sky, e o fatorial que o mediu

Este documento existe para que ninguém precise redescobrir por que o pareamento
imagem↔sensor estava deslocado, nem refazer a medição que quantificou o custo.

Tudo aqui foi medido no acervo do próprio LabMiM, não transcrito de literatura.
Os intervalos são de carimbo **local ingênuo**, como o registrador escreveu.

---

## 1. O CR5000 carimba a média no fim da janela

Uma linha escrita em `t` é a média sobre `(t − 5 min, t]`, não sobre
`[t, t + 5 min)` nem sobre uma janela centrada em `t`.

A verificação usa o único par de tabelas do acervo que amostra o mesmo sinal em
duas cadências ao mesmo tempo:

| tabela | cadência | período |
|---|---|---|
| `dados-labmim/LBM_solar_2024.dat` | 1 min | 2024-03-18 09:01 .. 2024-07-19 15:00 |
| `dados-labmim/LBM_lenta_2024.dat.backup` | 5 min | 2024-03-18 09:05 .. 2024-07-19 15:00 |

O `.backup` é a única fonte desse trecho de 5 minutos — mais um caso da regra
que `docs/acervo-da-estacao.md` já documenta: um glob `*.dat` teria descartado a
evidência.

Reconstruindo a média de 5 minutos a partir da série de 1 minuto sob as duas
convenções e comparando com o que o logger de fato gravou em `CM3Up_Wm2_Avg`:

| convenção | RMS vs. a tabela de 5 min | r | n |
|---|---|---|---|
| **end-stamped** `(t−5, t]` | **0,083 W/m²** | **1,000000** | 35.492 |
| begin-stamped `[t, t+5)` | 64,79 W/m² | 0,974 | 35.492 |

Não é correlação alta: é identidade. A convenção está decidida.

## 2. O custo de parear pelo carimbo cru

`allsky.data.alignment.CenterFrame.pair` faz `np.searchsorted` sobre os carimbos
crus, então o frame do instante τ recebe o rótulo cujo centroide temporal está
2,5 min antes de onde o pipeline o trata.

Medindo contra a verdade instantânea de 1 minuto, em 79.860 amostras diurnas
(a porta é `CM3Up_Wm2_Avg > 20 W/m²` sobre a série de 1 minuto; o pareamento é
`merge_asof(direction="nearest", tolerance=5min)`):

```
offset aplicado ao carimbo    RMS do erro de rótulo (GHI)
        0,00 min                     94,00 W/m²   <- o que o código fazia
       -2,50 min                     74,99 W/m²   <- centroide da janela
                                     -20,2 %
```

A varredura completa de −6 a +2 min tem mínimo **exatamente em −2,50**, o valor
que a convenção end-stamp prevê. Teoria e medição coincidem:

```
-3,50 → 81,15    -2,50 → 74,99    -1,50 → 81,69    0,00 → 94,00
```

Isto é **ruído de rótulo**: erro que nenhuma arquitetura remove.

## 3. A correção

`PrepareSensorConfig.timestamp_offset_minutes` (default `0.0`) soma minutos a
cada carimbo antes do pareamento; `-2.5` move a linha para o centroide da janela
que ela de fato promedia. Desloca **apenas** qual linha é pareada e a
`distance_minutes` registrada — nunca os valores.

O campo entra no hash de resume do manifesto (`"sensor"` pertence a
`_MANIFEST_CONFIG_SECTIONS`), então trocá-lo invalida o manifesto em vez de
reusá-lo em silêncio, e fica gravado no sidecar como
`thresholds.sensor_timestamp_offset_minutes`.

Sobre um mesmo conjunto de frames, o offset muda **49,9 % dos `target_dhi`**
(RMS da diferença 21,6 W/m²) e **14,4 % dos `sky_class`**.

## 4. O fatorial que mediu o efeito a jusante

Desenho: alinhamento × pooling × tarefa, 3 sementes por célula, 24 células.
Os dois braços de alinhamento compartilham uma árvore de frames, um store de
embeddings e o mesmo `split_id` — só os rótulos do manifesto diferem, então a
comparação é pareada de verdade.

Dataset: 46.015 amostras, 81 dias (2026-06-03 .. 08-26), tier `bare`,
split cronológico com gap de 1 dia (treino 30.704 / val 7.003 / teste 7.138).

Teste, DHI RMSE em W/m²:

| align | pooling | tarefa | s42 | s43 | s44 | média | sd |
|---|---|---|---|---|---|---|---|
| raw | cls | multitask | 35,59 | 35,32 | 35,05 | **35,32** | 0,27 |
| raw | cls | só-DHI | 34,43 | 34,52 | 35,06 | 34,67 | 0,34 |
| raw | cls+mean | multitask | 34,58 | 34,58 | 34,92 | 34,69 | 0,20 |
| raw | cls+mean | só-DHI | 33,38 | 33,71 | 32,50 | 33,20 | 0,62 |
| cen | cls | multitask | 32,46 | 31,88 | 32,47 | 32,27 | 0,34 |
| cen | cls | só-DHI | 31,49 | 31,10 | 32,41 | 31,67 | 0,67 |
| cen | cls+mean | multitask | 32,81 | 30,79 | 31,01 | 31,54 | 1,11 |
| **cen** | **cls+mean** | **só-DHI** | 30,21 | 31,19 | 31,18 | **30,86** | 0,56 |

Efeitos principais, pareados (os outros fatores e a semente fixos):

```
alinhamento  raw → centroide    -2,89 ± 0,22 W/m²
pooling      cls → cls+mean     -0,91 ± 0,22
tarefa       multitask → só-DHI -0,86 ± 0,27
```

Os três são reais (|média| > 2·erro-padrão). Entre os fatores testados SOBRE
EMBEDDINGS CONGELADOS, o alinhamento é o maior — maior que os dois efeitos de
readout somados. A seção 8 mostra que descongelar o backbone é maior ainda.

O σ entre sementes na baseline é 0,77 % dela (0,27 de 35,32), então efeitos da
ordem de 1 % são mensuráveis neste bloco de teste. Sem esse número nenhum dos
deltas acima poderia ser afirmado.

## 5. O viés é estruturado pela geometria solar

O MBE agregado da melhor célula congelada é **−4,94 ± 3,09** W/m² sobre 3
sementes (−8,49 / −3,46 / −2,87). Sobre as 24 células do fatorial é
−3,42 ± 3,42, com faixa de −9,62 a +1,86 e 20 das 24 negativas. **O valor
agregado é ruidoso e não deve ser citado de uma semente só.**

O que É robusto é a estrutura por elevação, monotônica em todas as sementes:

| elevação solar | MBE congelado (3 sementes) | n |
|---|---|---|
| 10–20° | +12,62 ± 4,65 | 1.002 |
| 20–35° | +0,25 ± 5,25 | 1.558 |
| 35–50° | −8,16 ± 4,01 | 1.688 |
| 50–90° | −11,95 ± 1,29 | 2.890 |

O modelo **subestima a difusa quando o sol está alto e superestima quando está
baixo**. A leitura por condição de céu (`clear` −9,59, `cloudy` +5,36,
`partly_cloudy_diffuse` −18,12) vem de uma única semente e serve como indício,
não como número.

## 6. Limitações do desenho, explícitas

- O bloco de teste cai **inteiramente em agosto** e é 57 % céu claro. Um número
  medido aqui não transporta para outra estação sem verificação.
- Nada disto é comparável com os 38,4 W/m² publicados em 2026-08-13: janela de
  teste diferente, 81 dias em vez de 65, tier de features diferente e, no braço
  centroide, metade dos rótulos diferente.
- A `skill_persistence` é −1,50 e sempre será negativa nesta tarefa: a estimativa
  é em t=0 e a persistência de 1 minuto é quase a resposta.

## 7. O piso de ruído em difusa, medido e não transportado

A pergunta que decide se vale continuar mexendo no modelo: **quanto do erro
restante é imposto pelo rótulo de 5 minutos?**

Ela foi respondida por medição direta, não por transporte a partir do GHI. A
tabela de 1 minuto de 2024 carrega `CMP21_Wm2_Avg` viva (18.187 valores
distintos), e naquele período a CMP21 *era* o canal de difusa do sítio
(`docs/acervo-da-estacao.md`, faixa 2020-06-01 .. 2025-03-12). Comparando a
difusa instantânea de 1 minuto contra o rótulo de 5 minutos que o pipeline
escolheria, em 79.743 amostras diurnas (mesma porta de dia `CM3Up_Wm2_Avg >
20 W/m²`, mais a porta física na própria difusa indicada em cada linha):

| porta física | carimbo cru | centroide | redução |
|---|---|---|---|
| [0, 800] W/m² | 21,66 | **15,35** | 29,1 % |
| [0, 600] | 21,56 | 14,92 | 30,8 % |
| [5, 500] | 21,25 | 14,73 | 30,7 % |
| [0, 1000] | 22,61 | 16,35 | 27,7 % |

O piso é ~15 W/m², estável entre portas. Com a melhor célula em 30,86 W/m² e
supondo erro de modelo e ruído de rótulo independentes:

```
erro do modelo = sqrt(30,86² − 15,35²) = 26,77 W/m²
fração da VARIÂNCIA do erro que é rótulo = 24,7 %
```

**Três quartos do erro ainda são do modelo.** A arquitetura não está no teto
imposto pelo rótulo, e continuar a melhorá-la tem retorno.

Ressalvas, porque isto é uma medição em outro instrumento e outra época: a CMP21
de 2024 roda sob o programa v19, não o v22 atual, e o sensor de difusa hoje é a
PSP. O número transporta como ordem de grandeza, não como constante calibrada.
Ainda assim é evidência direta, e substitui uma estimativa anterior de ~25 W/m²
obtida multiplicando o valor do GHI por um fator — que superestimava o piso em
cerca de 60 % e, se aceita, teria recomendado abandonar a modelagem cedo demais.

## 8. O fine-tune: o backbone congelado era o gargalo

Modo imagem, join centroide, cabeça única de DHI, 40 épocas com early stopping,
2 sementes por profundidade. `image_size` 224, ~51 s/época, 482 MiB de VRAM.

| profundidade descongelada | s42 | s43 | média | vs. congelado |
|---|---|---|---|---|
| congelado (melhor célula) | — | — | 30,86 | — |
| últimos 2 de 12 blocos | 28,17 | 27,61 | 27,89 | −9,6 % |
| últimos 4 | 24,99 | 26,97 | 25,98 | −15,8 % |
| **todos os 12** | 21,11 | 20,50 | **20,80** | **−32,6 %** |

Monotônico na profundidade, e maior que todos os fatores do fatorial somados.
R² 0,939, MAE 14,5, `skill_clearsky` +0,72. **Nenhum run bateu no limite de 40
épocas** — todos pararam por early stopping entre 12 e 29, então "treinar mais
tempo" não era o lever; o lever era quantos parâmetros podiam aprender.

## 9. O teto se inverteu

Tomando o piso da seção 7 **como ordem de grandeza, não como constante** — ele
foi medido noutro instrumento e noutra época, e a decomposição abaixo supõe
ainda que erro de modelo e ruído de rótulo sejam independentes, o que não foi
verificado:

| piso suposto | erro de modelo implícito | fração da variância que é rótulo |
|---|---|---|
| 14,7 W/m² (porta [5,500]) | 14,7 | 50 % |
| 15,35 (porta [0,800]) | 14,0 | 54 % |
| 16,35 (porta [0,1000]) | 12,8 | 62 % |

Qualquer que seja a porta, a leitura qualitativa é a mesma e é o que importa:
**antes do fine-tune o rótulo respondia por cerca de um quarto da variância do
erro; com 20,80 passou a responder pela maioria.** Isso reordena as prioridades
— restaurar uma tabela de 1 minuto com `PSP_Wm2_Avg` passou de lever de segunda
ordem a primeira, não porque o rótulo piorou, mas porque o modelo melhorou.

O número exato depende de duas suposições não verificadas (transporte entre
instrumentos e independência), então ele orienta prioridade, não serve de meta.

## 10. O viés de −8 W/m² resistiu a tudo que testei

O fine-tune **não removeu** o viés — e, medido sobre 3 sementes em cada lado,
aumentou-o: congelado **−4,94 ± 3,09**, fine-tune de 12 blocos
**−7,57 ± 1,36**. Ao mesmo tempo ele APERTOU a estrutura por elevação (o desvio
entre sementes cai de 1,3–5,3 para 0,7–2,2 W/m²):

| elevação | congelado | fine-tune 12 |
|---|---|---|
| 10–20° | +12,62 ± 4,65 | +5,99 ± 0,72 |
| 20–35° | +0,25 ± 5,25 | −4,16 ± 0,72 |
| 35–50° | −8,16 ± 4,01 | −9,37 ± 2,20 |
| 50–90° | −11,95 ± 1,29 | −13,06 ± 1,68 |

Ou seja: o fine-tune cortou o RMSE pela metade e tornou o viés mais nítido e
mais reprodutível, não menor. Duas hipóteses para a causa foram testadas e
**as duas falharam**.

**Hipótese 1 — regressão à média sob MSE.** Sweep de perda sobre a configuração
vencedora:

| perda | RMSE | MAE | MBE | R² |
|---|---|---|---|---|
| mse (3 sementes) | 20,74 ± 0,32 | 14,44 | −7,57 | 0,939 |
| mae | 20,31 | 14,05 | −8,46 | 0,942 |
| huber | 21,80 | 15,45 | −11,21 | 0,933 |
| heteroscedastic | 32,41 | 23,07 | −17,09 | 0,852 |

O braço `mae` era a falsificação: L1 busca a mediana, então num alvo assimétrico
à direita deveria **piorar** o viés. Não piorou. E a `heteroscedastic`, que a
literatura de cabeças de regressão indicava para subestimação de cauda, foi de
longe a pior. O viés sobrevive a L2, L1, Huber e NLL gaussiana: não é artefato
de família de perda. Nenhuma perda bate a MSE de forma mensurável — a `mae` está
dentro de 1,3 σ com uma semente só.

**Hipótese 2 — deriva do alvo no split cronológico.** Também falha, e no sentido
contrário:

```
DHI média:  treino 136,19   val 130,22   teste 121,60 W/m²
```

O teste tem difusa mais BAIXA que o treino, então um modelo ancorado na média de
treino superestimaria (MBE positivo). O observado é negativo.

**O que se sabe:** o viés é mais negativo em sol alto (−13,06 acima de 50°,
+7,59 entre 10° e 20°) e a janela de teste tem elevação média 42,2° contra 37,4°
no treino, o que amplifica o efeito. **O que não se sabe:** a causa. Fica
registrado como questão aberta. O próximo teste indicado é separar rótulo de
modelo — verificar se a correção de anel de sombra aplicada à PSP em
`sensors/calibration.py` tem dependência residual de elevação, comparando a
difusa corrigida contra a difusa de céu claro modelada por faixa de elevação.

## 11. O que isto implica

Em ordem de tamanho medido, partindo da configuração como estava publicada:

| mudança | efeito | custo |
|---|---|---|
| descongelar o DINOv2 inteiro | **−10,1 W/m²** | config; ~30 min de GPU |
| corrigir o join para o centroide | **−2,9** | 5 linhas de código |
| pooling `cls+mean` | −0,9 | uma linha de config |
| cabeça única de DHI | −0,9 | uma linha de config |
| trocar a perda | nada mensurável | — |

Baseline publicada 35,32 → **20,80 W/m²**, uma redução de 41 %.

A abordagem **não** estava fadada: estava algemada por um backbone congelado. O
achado que a literatura sustenta é exatamente esse — finetuning vs. congelado, e
não uma arquitetura específica.

Daqui em diante o rótulo responde pela maioria da variância do erro (50–62 %
conforme a porta suposta), então o próximo lever é a cadência de aquisição, não
o encoder. E o viés continua sem explicação: ele é estável e monotônico na
elevação solar, cresceu com o fine-tune (−4,94 → −7,57 W/m²), e as duas
hipóteses testadas foram falsificadas.

Ver `docs/acervo-da-estacao.md` para a cronologia das tabelas e
`docs/allsky-archive.md` para o relógio da câmera.

---

## 12. Como o run reportado foi produzido, e o que ficou em aberto

Os configs de cada célula do estudo NÃO são versionados — são derivados
mecanicamente de `configs/allsky/experiments/_base.yaml` mais o fragmento
`models/image_only.yaml`, variando só quatro chaves. Para reconstruir o estudo
basta gerar as combinações:

| eixo | valores | onde |
|---|---|---|
| alinhamento | `timestamp_offset_minutes` 0.0 e −2.5 | cópia de `local_prepare.yaml`, cada uma com seu `output.dataset_dir` |
| pooling | `embeddings.pooling` `cls` e `cls+mean` | idem; o store cls+mean exige `precompute-embeddings -o <dir>/embeddings_clsmean` |
| tarefa | `targets.kindex.enabled` / `targets.sky.enabled` | no config do experimento |
| profundidade | `model.unfreeze_last_n` 2, 4, 12 com `data.input_mode: image` | no config do experimento |

Os dois braços de alinhamento precisam de `output.dataset_dir` distintos: o nome
do manifesto é a constante `DATASET_MANIFEST_FILENAME`, então dois offsets no
mesmo diretório se sobrescrevem em silêncio.

**A execução que gerou os números acima usou um atalho**: uma única árvore de
frames e um único store de embeddings, com o manifesto do braço centroide
construído no mesmo diretório e renomeado. Os rótulos, os splits e o `split_id`
são os mesmos que a receita acima produz; o que o atalho poupou foi reextrair
46 mil frames.

**Gap conhecido, não corrigido aqui:** `allsky.snapshot._sensor_row_near` pareia
a linha do sensor pelo carimbo cru e não conhece
`timestamp_offset_minutes`, então o caminho de inferência não reproduz o join do
treino. Não foi corrigido junto porque o offset correto ali depende da convenção
de carimbo do CSV processado que o snapshot consome, e esse caminho hoje sequer
lê o export de `labmim-sensor-process`: a coluna de tempo do export é o índice
sem cabeçalho, e `SENSOR_TIME_COLUMNS` só aceita `timestamp`/`TIMESTAMP`/
`datetime`/`time`, então a leitura falha antes do pareamento. As duas coisas
devem ser resolvidas juntas, com medição própria.
