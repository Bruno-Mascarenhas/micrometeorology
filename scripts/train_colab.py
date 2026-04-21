"""Google Colab / Remote GPU Training Script.

Este script é altamente otimizado para o Google Colab:
- Suporta Automatic Mixed Precision (AMP) por padrão (ativado no Trainer).
- Dataloaders otimizados e zero-copy tensor loading.
- Aceita passagem de argumentos via linha de comando para automação.

Como rodar no Colab:
!python scripts/train_colab.py --train_path /content/drive/MyDrive/dataset.csv --val_path /content/drive/MyDrive/val.csv --log_dir /content/drive/MyDrive/logs
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from solrad_correction.config import ModelConfig
from solrad_correction.datasets.sequence import SequenceDataset
from solrad_correction.models.lstm import LSTMRegressor
from solrad_correction.models.transformer import TransformerRegressor
from solrad_correction.utils.seeds import get_device

# Configura logs com timestamp
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ColabTrain")


def parse_args() -> argparse.Namespace:
    """Extrai os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description="Treinamento otimizado (Colab)")

    # Dataset e I/O
    parser.add_argument("--train_path", type=str, required=True, help="Caminho do CSV de Treino")
    parser.add_argument("--val_path", type=str, required=True, help="Caminho do CSV de Validação")
    parser.add_argument(
        "--out_model", type=str, default="best_model.pt", help="Onde salvar o modelo (.pt)"
    )
    parser.add_argument(
        "--log_dir", type=str, default="runs/", help="Pasta para os logs do TensorBoard"
    )

    # Features (substitua pelos nomes exatos do seu dataset)
    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        default=["radiacao_obs", "temp", "pressao"],
        help="Colunas de input",
    )
    parser.add_argument("--target", type=str, default="radiacao_real", help="Coluna target")

    # Hiperparâmetros de Treinamento
    parser.add_argument("--model_type", type=str, choices=["lstm", "transformer"], default="lstm")
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Tamanho do Batch (use 512 ou 1024 em T4)"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Máximo de Épocas")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--patience", type=int, default=15, help="Épocas até o Early Stopping")

    # Hiperparâmetros Específicos
    parser.add_argument(
        "--hidden_size", type=int, default=64, help="LSTM Hidden Size ou Transformer D_Model"
    )
    parser.add_argument("--layers", type=int, default=2, help="Num Layers")

    return parser.parse_args()


def load_data(path: str, features: list[str], target: str) -> SequenceDataset:
    """Carrega dados em memória via pandas e converte para SequenceDataset."""
    logger.info(f"Lendo dados de {path}...")
    df = pd.read_csv(path)

    # Aqui pode haver nans, dropamos por segurança
    df = df.dropna(subset=[*features, target])

    x_feat = df[features].to_numpy()
    y_feat = df[target].to_numpy()

    logger.info(f" -> {len(df)} samples carregados.")
    return SequenceDataset(x_feat, y_feat)


def main():
    args = parse_args()
    device = get_device()
    logger.info(f"Hardware Ativo: {device.upper()}")
    if "cuda" not in device:
        logger.warning("Nenhuma GPU detectada! Vá em Runtime > Change runtime type > T4 GPU")

    # 1. Carrega os Dados
    train_dataset = load_data(args.train_path, args.features, args.target)
    val_dataset = load_data(args.val_path, args.features, args.target)

    # 2. Configura a Run
    config = ModelConfig(
        model_type=args.model_type,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        patience=args.patience,
        log_dir=args.log_dir,
    )

    input_size = len(args.features)

    # 3. Inicializa o Modelo escolhido
    logger.info(f"Iniciando {args.model_type.upper()} com AMP (Automatic Mixed Precision)")
    if args.model_type == "lstm":
        config.lstm_hidden_size = args.hidden_size
        config.lstm_num_layers = args.layers
        model = LSTMRegressor(
            input_size=input_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            device=device,
        )
    else:
        config.tf_d_model = args.hidden_size
        config.tf_num_encoder_layers = args.layers
        model = TransformerRegressor(
            input_size=input_size,
            d_model=config.tf_d_model,
            num_encoder_layers=config.tf_num_encoder_layers,
            device=device,
        )

    # 4. Treina
    # O nosso setup interno lida com o pin_memory, non_blocking, compile e mixed precision automático.
    model.fit(train_dataset, val_dataset, config=config)

    # 5. Salva o melhor checkpoint (deve ser em um folder do Google Drive)
    out_path = Path(args.out_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    logger.info(f"Treinamento finalizado. Modelo salvo em {out_path.resolve()}")


if __name__ == "__main__":
    main()
