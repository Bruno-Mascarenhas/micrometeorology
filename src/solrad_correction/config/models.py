"""Model and neural-training configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ModelConfig:
    """Model-specific hyperparameters."""

    model_type: str = "svm"
    log_dir: str | None = None

    svm_kernel: str = "rbf"
    svm_c: float = 1.0
    svm_epsilon: float = 0.1
    svm_gamma: str = "scale"

    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.1

    tf_d_model: int = 64
    tf_nhead: int = 4
    tf_num_encoder_layers: int = 2
    tf_dim_feedforward: int = 128
    tf_dropout: float = 0.1

    sequence_length: int = 24
    batch_size: int = 32

    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 10
    min_delta: float = 1e-4
    evaluation_policy: str = "model_native"
