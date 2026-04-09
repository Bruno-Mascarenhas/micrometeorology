# `solrad_correction` — Documentação

Pacote para correção de viés da radiação solar difusa do modelo WRF usando aprendizado de máquina. Projetado para ser genérico — funciona com dados de qualquer estação meteorológica e coordenadas geográficas.

---

## Visão Geral

O WRF (Weather Research and Forecasting) frequentemente apresenta viés sistemático na estimativa da radiação solar difusa (`SW_dif`). Este pacote treina modelos de ML para corrigir esse viés usando dados observacionais de estações meteorológicas como referência.

### Modelos Disponíveis

| Modelo | Tipo | Entrada | Quando Usar |
|---|---|---|---|
| **SVM** | Scikit-learn (SVR) | Tabular (1 linha = 1 amostra) | Baseline rápido, poucos dados |
| **LSTM** | PyTorch (RNN) | Janelas temporais (seq_len × features) | Captura dependências temporais |
| **Transformer** | PyTorch (Attention) | Janelas temporais (seq_len × features) | Relações de longo alcance, mais dados |

---

## Estrutura do Pacote

```
src/solrad_correction/
├── __init__.py              # Versão e docstring
├── config.py                # Configuração de experimentos (dataclasses + YAML)
├── cli.py                   # CLI: solrad-run
├── data/
│   ├── loaders.py           # Wrappers para carregar dados de sensor/WRF
│   ├── alignment.py         # Alinhamento temporal sensor ↔ WRF
│   ├── preprocessing.py     # Pipeline com fit apenas no treino (sem leakage)
│   └── splits.py            # Divisão cronológica, walk-forward, K-fold temporal
├── features/
│   ├── engineering.py       # Lags, médias móveis, diferenças
│   ├── temporal.py          # Hora, dia do ano, mês + encoding cíclico (sin/cos)
│   └── sequence.py          # Construção de janelas deslizantes para LSTM/Transformer
├── datasets/
│   ├── tabular.py           # TabularDataset (X, y) + save/load reproduzível
│   └── sequence.py          # SequenceDataset (torch.Dataset) + save/load
├── models/
│   ├── base.py              # BaseRegressorModel (ABC): interface unificada
│   ├── sklearn_base.py      # Wrapper para regressores scikit-learn
│   ├── torch_base.py        # Base para modelos PyTorch (device, transfer learning)
│   ├── svm.py               # SVMRegressor (SVR)
│   ├── lstm.py              # LSTMRegressor + LSTMNet (nn.Module)
│   └── transformer.py       # TransformerRegressor + TimeSeriesTransformer
├── training/
│   ├── trainer.py           # Loop de treino completo
│   ├── loops.py             # train_one_epoch(), evaluate_epoch()
│   ├── callbacks.py         # Early stopping, checkpointing
│   └── progress.py          # Progresso com % por batch, epoch, e ETA
├── evaluation/
│   ├── metrics.py           # Métricas (reutiliza labmim + MAPE)
│   ├── reports.py           # ExperimentReport: salva métricas, config, histórico
│   └── comparison.py        # Tabela comparativa entre experimentos
├── experiments/
│   └── runner.py            # Pipeline completo: config → dados → treino → avaliação
└── utils/
    ├── seeds.py             # Controle de seeds (numpy, torch, random)
    ├── io.py                # Leitura/escrita de JSON, predições CSV
    └── serialization.py     # Serialização: joblib (sklearn) / torch (checkpoint)
```

---

## Instalação

```bash
# Com PyTorch CPU:
pip install -e ".[tcc]"

# Com PyTorch CUDA (instale o torch primeiro):
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[tcc-cuda]"
```

### Verificar se GPU está disponível

```python
from solrad_correction.utils.seeds import get_device
print(get_device())  # "cuda" ou "cpu"
```

---

## Guia Rápido

### 1. Criar um Arquivo de Configuração

```yaml
# configs/tcc/experiments/meu_experimento.yaml
name: svm_baseline_salvador
description: "SVM com dados horários de Salvador"
seed: 42

data:
  hourly_data_path: data/hourly/sensor_data.csv
  target_column: SW_dif
  feature_columns:
    - SWDOWN
    - T2
    - Q2
    - PSFC
  station_lat: -12.95
  station_lon: -38.51

split:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  shuffle: false    # NUNCA usar shuffle em séries temporais

preprocess:
  scaler_type: standard    # "standard", "minmax", "none"
  impute_strategy: drop    # "drop", "ffill", "mean", "interpolate"

features:
  add_temporal: true       # hora, dia, mês
  cyclic_encoding: true    # sin/cos
  lag_steps: []
  rolling_windows: []

model:
  model_type: svm
  svm_kernel: rbf
  svm_c: 10.0
  svm_epsilon: 0.1
  svm_gamma: scale

output_dir: output/experiments
```

### 2. Rodar o Experimento

```bash
solrad-run --config configs/tcc/experiments/meu_experimento.yaml
```

Ou via Python:

```python
from solrad_correction.config import ExperimentConfig
from solrad_correction.experiments.runner import run_experiment

config = ExperimentConfig.from_yaml("configs/tcc/experiments/meu_experimento.yaml")
report = run_experiment(config)
report.print_summary()
```

### 3. Estrutura de Saída

Cada experimento gera um diretório com tudo necessário para reproduzir:

```
output/experiments/svm_baseline_salvador/
├── config.yaml                      # Config exata usada
├── config_resolved.json             # Config como JSON
├── metrics.json                     # Resultados (RMSE, MAE, R², etc.)
├── predictions.csv                  # y_true vs y_pred
├── training_history.csv             # Loss por epoch (se neural)
├── model.joblib (ou model.pt)       # Modelo treinado
├── preprocessing_pipeline.joblib    # Estado do preprocessamento
└── datasets/
    ├── train/                       # Dataset de treino salvo
    └── test/                        # Dataset de teste salvo
```

---

## Usando Cada Modelo

### SVM

```yaml
model:
  model_type: svm
  svm_kernel: rbf       # "rbf", "linear", "poly"
  svm_c: 10.0           # Regularização (maior = menos regularização)
  svm_epsilon: 0.1      # Margem de tolerância
  svm_gamma: scale      # "scale", "auto", ou float
```

### LSTM

```yaml
model:
  model_type: lstm
  lstm_hidden_size: 64       # Neurônios na camada hidden
  lstm_num_layers: 2         # Número de camadas LSTM empilhadas
  lstm_dropout: 0.1          # Dropout entre camadas
  sequence_length: 24        # Tamanho da janela temporal (horas)
  batch_size: 32
  learning_rate: 0.001
  max_epochs: 100
  patience: 10               # Early stopping: para após 10 epochs sem melhora
```

### Transformer

```yaml
model:
  model_type: transformer
  tf_d_model: 64             # Dimensão do embedding
  tf_nhead: 4                # Número de attention heads (d_model deve ser divisível)
  tf_num_encoder_layers: 2   # Número de blocos encoder
  tf_dim_feedforward: 128    # Dimensão do FFN interno
  tf_dropout: 0.1
  sequence_length: 24
  batch_size: 32
  learning_rate: 0.001
  max_epochs: 100
  patience: 10
```

---

## Transfer Learning (Continuar Treinamento)

Treinamento pode ser retomado de um checkpoint anterior:

```yaml
model:
  model_type: lstm
  pretrained_path: output/experiments/lstm_v1/model.pt   # Pesos anteriores
  max_epochs: 50       # Epochs ADICIONAIS
```

Isso carrega os pesos do `lstm_v1` e treina por mais 50 epochs. O checkpoint salva:

- `model_state_dict` (pesos do modelo)
- `optimizer_state_dict` (estado do otimizador)
- `epoch` (epoch em que parou)
- `config` (parâmetros da arquitetura para reconstrução)

---

## Prevenção de Data Leakage

O pacote implementa múltiplas camadas de proteção:

### 1. Divisão Cronológica

```
|←——— treino (70%) ———→|←— val (15%) —→|←— teste (15%) —→|
        passado              presente           futuro
```

O `shuffle=false` é o padrão. Se ativado, um warning é emitido.

### 2. Preprocessamento com Fit no Treino

```python
pipeline = PreprocessingPipeline(scaler_type="standard")
train_pp = pipeline.fit_transform(train_df)   # ← Fit APENAS aqui
val_pp   = pipeline.transform(val_df)          # ← Aplica parâmetros do treino
test_pp  = pipeline.transform(test_df)         # ← Aplica parâmetros do treino
```

A média e desvio-padrão usados para normalizar são calculados **apenas** no treino. Validação e teste usam esses mesmos valores.

### 3. Janelas Deslizantes (Sequence)

Para LSTM/Transformer, cada janela olha apenas para o **passado**:

```
Janela 1: [t₀, t₁, t₂, t₃] → alvo: t₄
Janela 2: [t₁, t₂, t₃, t₄] → alvo: t₅
```

O alvo sempre está **após** o final da janela.

### 4. Pipeline Serializado

O estado do preprocessamento é salvo com cada experimento (`preprocessing_pipeline.joblib`), garantindo que o mesmo transform exato possa ser aplicado em dados futuros.

---

## Comparando Experimentos

```python
from solrad_correction.evaluation.comparison import compare_experiments

df = compare_experiments([
    "output/experiments/svm_baseline",
    "output/experiments/lstm_24h",
    "output/experiments/transformer_48h",
])
print(df)
#                     RMSE     MAE      R²      r      d     MAPE
# svm_baseline      45.23   32.10   0.847  0.921  0.958   18.5
# lstm_24h          38.67   27.45   0.893  0.946  0.972   15.2
# transformer_48h   36.12   25.89   0.908  0.953  0.978   13.8
```

---

## Feature Engineering

### Temporal

```yaml
features:
  add_temporal: true      # Adiciona: hour, day_of_year, month, weekday
  cyclic_encoding: true   # Converte para sin/cos (evita descontinuidade 23→0)
```

**Por que encoding cíclico?** A hora 23 e a hora 0 são adjacentes, mas numericamente distantes. O encoding sin/cos preserva essa proximidade:

```
hour=0  → sin=0.00, cos=1.00
hour=6  → sin=1.00, cos=0.00
hour=12 → sin=0.00, cos=-1.00
hour=23 → sin=-0.26, cos=0.97  ← próximo de hour=0
```

### Lags e Médias Móveis

```yaml
features:
  lag_steps: [1, 3, 6, 12, 24]       # Valores das últimas 1, 3, 6, 12, 24 horas
  rolling_windows: [3, 6, 12, 24]     # Média e desvio-padrão móvel
  rolling_aggs: ["mean", "std"]
```

---

## Progresso do Treinamento

Durante o treinamento neural, o progresso é exibido em tempo real:

```
  Epoch 1/100 [100.0%] ETA epoch: 0.0s | Overall:  1.0%
  Epoch 1/100 — train_loss=0.235412  val_loss=0.198765 (2.3s/epoch, ETA: 3m48s)
  Epoch 2/100 — train_loss=0.189234  val_loss=0.167892 (2.1s/epoch, ETA: 3m25s)
  ...
  Epoch 23/100 — train_loss=0.045123  val_loss=0.052345 (2.2s/epoch, ETA: 2m48s) [EARLY STOP]

✓ Training complete in 50.6s
```

---

## Adicionando um Novo Modelo

1. Escolha a base correta:
   - **Sklearn** → herde de `SklearnRegressorModel`
   - **PyTorch** → herde de `TorchRegressorModel`

2. Implemente a interface:

```python
from solrad_correction.models.sklearn_base import SklearnRegressorModel

class MeuModelo(SklearnRegressorModel):
    @property
    def name(self) -> str:
        return "MeuModelo"

    def __init__(self, param1: float = 1.0) -> None:
        from sklearn.ensemble import RandomForestRegressor
        self._estimator = RandomForestRegressor(n_estimators=100)

    @classmethod
    def from_config(cls, config):
        return cls(param1=config.custom_param)
```

3. Registre no `runner.py`:

```python
elif model_type == "meumodelo":
    model = MeuModelo.from_config(config.model)
```

Para PyTorch, use `TorchRegressorModel` que fornece automaticamente:
- Detecção automática de GPU/CPU
- Transfer learning
- Salvamento de checkpoints
- Integração com o Trainer (progress + early stopping)

---

## Dúvidas Frequentes

### Posso usar para outra variável além de SW_dif?

Sim. Altere `target_column` e `feature_columns` no config YAML. O pacote é genérico.

### Posso treinar com dados de outra cidade?

Sim. Altere `station_lat` e `station_lon` no config e forneça os dados correspondentes. O nome `solrad_correction` é genérico, não está vinculado a nenhuma localização.

### O que é `tolerance="30min"` no alinhamento?

Ao alinhar dados de sensor (observação) com dados WRF (modelo), os timestamps podem não coincidir exatamente. A tolerância permite parear timestamps com até 30 minutos de diferença.

### Preciso de GPU?

Não. O SVM roda apenas em CPU. LSTM e Transformer funcionam em CPU, mas são significativamente mais rápidos com CUDA. O código auto-detecta e usa GPU se disponível.

### Como reproduzir exatamente um experimento?

1. Use o mesmo `seed` no config
2. Use o dataset salvo em `experiments/<nome>/datasets/`
3. Use o config salvo em `experiments/<nome>/config.yaml`

```python
config = ExperimentConfig.from_yaml("output/experiments/lstm_v1/config.yaml")
report = run_experiment(config)
```

### Como ver quais features foram usadas?

O dataset salvo inclui `feature_names.csv`:

```python
from solrad_correction.datasets.tabular import TabularDataset
ds = TabularDataset.load("output/experiments/svm_v1/datasets/train")
print(ds.feature_names)
```

### Como fazer inverse transform das predições?

O pipeline salvo permite desfazer a normalização:

```python
from solrad_correction.data.preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline.load("output/experiments/svm_v1/preprocessing_pipeline.joblib")
y_original = pipeline.inverse_transform_column(y_normalizado, "SW_dif")
```
