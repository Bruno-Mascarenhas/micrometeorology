"""Model and neural-training configuration."""

from dataclasses import dataclass


@dataclass(slots=True)
class ModelConfig:
    """Model-specific hyperparameters.

    ``model_type`` must name a model registered in
    ``solrad_correction.models.registry``; the ``svm_*``, ``lstm_*`` and ``tf_*``
    groups apply only to their own architecture and are ignored by the others.

    ``evaluation_policy`` selects what the reported metrics are computed over:
    ``model_native`` scores every model on the rows it can predict, while
    ``common_sequence_horizon`` restricts all models to the rows a sequence model
    with ``sequence_length`` history can reach, so a tabular and a recurrent
    model are compared on the same target rows.

    ``ExperimentConfig.validate`` enforces the cross-field bounds: positive
    ``sequence_length``, ``batch_size``, ``max_epochs``, ``tf_d_model`` and
    ``tf_nhead``, with ``tf_d_model`` divisible by ``tf_nhead`` (each attention
    head takes an equal slice of the model dimension).
    """

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
