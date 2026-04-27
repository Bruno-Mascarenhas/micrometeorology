# solrad_correction open points: architecture, performance, and Colab readiness

Date: 2026-04-27 system date; local repository context reported 2026-04-26 America/Sao_Paulo.  
Scope inspected: `src/solrad_correction/`, TCC tests, `configs/tcc/`, `scripts/train_colab.py`, `notebooks/tcc/02_colab_training.ipynb`, `docs/solrad_correction.md`, `README.md`, `pyproject.toml`, and `.github/workflows/ci.yml`.  
Explicitly not inspected: `data/`.

## 1. Executive summary

`solrad_correction` is already organized into recognizable layers: config, data loading, feature engineering, preprocessing, datasets, model wrappers, training loops, evaluation, reporting, and CLI. The current implementation is good for small and medium CPU experiments and has useful scientific safeguards: chronological splitting, train-only preprocessing fit, explicit sequence target horizon, reproducible seeds, persisted preprocessing state, saved metrics and predictions, and smoke coverage for SVM/LSTM/Transformer.

The main blocker for serious neural-network training is not model correctness. It is pipeline scalability and operational configurability. `run_experiment()` is a single orchestration path that owns loading, feature engineering, splitting, preprocessing, dataset materialization, model selection, training, evaluation, and artifact writing. Sequence training currently materializes sliding windows as a dense contiguous array, then materializes torch tensors again in `SequenceDataset`. This is simple and testable, but it can multiply memory by roughly `sequence_length` for large hourly or WRF-derived tabular datasets.

Before heavy Colab/GPU runs, the highest-value work is to add a validated runtime/training config, introduce lazy or cached sequence datasets, expose DataLoader/device/AMP/resume controls without breaking `solrad-run`, and save richer experiment metadata. Broad model additions should wait until these foundations are stable.

Final recommendation: refactor the training/data plumbing before serious LSTM/Transformer experiments. Continue small CPU smoke experiments locally, but treat current long GPU runs as exploratory only because memory behavior, metadata capture, and resume semantics are not yet strong enough for expensive training.

### Implementation status

- Implemented: P0.1 lazy/cached sequence datasets. Neural training and inference use `WindowedSequenceDataset`; sequence artifacts are saved as base arrays plus window metadata without dense window materialization. Dense `SequenceDatasetMeta` compatibility was removed; lazy cache is the canonical saved sequence format.
- Implemented: P1.1 config validation and resolved-config printing. `ExperimentConfig.validate()` checks model type, evaluation policy, split ratios, positive sequence/batch/epoch values, and Transformer head divisibility. `solrad-run` now supports `--validate-config` and `--print-config`.
- Implemented: P1.3 training history persistence. Torch models expose the last fit history and neural experiment runs write `training_history.csv`.
- Implemented: P1.4 metadata sidecar. Experiment runs write best-effort `metadata.json` with Python/package/device/CUDA/git/timing/model/config summary fields.
- Implemented: P0.2 runtime/training configuration and DataLoader resolution. Optional `runtime:` config now controls device, workers, pin memory, persistent workers, prefetching, AMP, compile override, gradient clipping, profiling, row limits, checkpoint path, checkpoint cadence, and resume path while preserving old YAML defaults.
- Implemented: P0.3 robust torch checkpoints. Neural runner paths now default to `checkpoints/best.pt` and `checkpoints/last.pt`; checkpoints include optimizer/scheduler/scaler state and metadata, and `runtime.resume` resumes full training state.
- Implemented: P0.4 composable experiment pipeline. The public `run_experiment(config)` wrapper remains, while implementation stages now live in `experiments/pipeline.py`.
- Implemented: P0.5 Colab execution unification. `scripts/train_colab.py` now loads an experiment config, applies Colab/runtime overrides, and delegates to the same `run_experiment` path as `solrad-run`.
- Implemented: P1.2 model registry/factory. Existing SVM/LSTM/Transformer constructors are now accessed through `models/registry.py`.
- Implemented: P1.8 CLI operational flags. `solrad-run` supports runtime overrides including `--device`, `--dry-run`, `--smoke-test`, `--limit-rows`, `--profile`, `--num-workers`, `--pin-memory/--no-pin-memory`, `--amp/--no-amp`, `--compile/--no-compile`, `--resume`, and `--output-dir`.
- Implemented: production data-loading layer. Preprocessed hourly inputs now use a format-aware table loader with CSV, Parquet, auto detection, selected-column projection, datetime index parsing, dtype hints, and loader-level row limits for development.
- Implemented: explicit preprocessing state. The preprocessing pipeline now serializes a versioned fitted state with input/output schema, feature/target roles, row counts, imputation values, scaling values, dropped-column reasons, and strict transform schema validation. A human-readable `metadata/preprocessing_state.json` is written next to the executable joblib pipeline.
- Implemented: canonical experiment artifact layout. Runs now write `configs/`, `metrics/`, `predictions/`, `metadata/`, `preprocessing/`, `models/`, `checkpoints/`, `datasets/{train,val,test}`, `logs/`, `profiles/`, and `cache/`, plus `manifest.json` with artifact paths, byte sizes, and checksums.
- Implemented: stable profile artifact. `--profile` / `runtime.profile` writes `profiles/profile.json` with schema version, stage timings, and total stage time.
- Implemented: synthetic benchmark scripts outside `data/` for loading, preprocessing, sequence DataLoader throughput, and artifact/checkpoint serialization. Defaults are CPU-safe and generate synthetic data under `scratch/`.
- Simplification decision: no legacy training-control aliases are maintained. `runtime.resume` replaces `model.pretrained_path`, `runtime.torch_compile` replaces `model.torch_compile`, and `solrad-run --output-dir/-o` is the canonical output override.
- Breaking change: root-level experiment artifacts are no longer duplicated. Use `configs/config.yaml`, `metrics/metrics.json`, `predictions/predictions.csv`, `models/model.*`, and `preprocessing/preprocessing_pipeline.joblib`.

## 2. Current architecture assessment

### What is clean today

- Public CLI name `solrad-run` is preserved in `pyproject.toml` and implemented through `solrad_correction.cli:run_experiment_cli`.
- Package layout follows the documented intent: `data/`, `features/`, `datasets/`, `models/`, `training/`, `evaluation/`, `experiments/`, and `utils/`.
- Config is centralized in dataclasses in `src/solrad_correction/config.py`.
- Temporal split and preprocessing are separated from model code.
- SVM and PyTorch models share a basic `BaseRegressorModel` interface.
- Low-level torch loops are separated from the higher-level `Trainer`.
- Tests use synthetic data and cover split order, preprocessing leakage, sequence target alignment, model interface, training smoke, checkpoint save/load/resume, CLI help, and evaluation row policy.
- CI runs Ruff, MyPy, and pytest on Python 3.14.

### Main architectural concerns

- `src/solrad_correction/experiments/runner.py` is a god-function. It is the only production pipeline and combines data I/O, feature engineering, split policy, preprocessing state, dataset creation, model factory logic, training, prediction, inverse transform, metrics, and artifact writing.
- There is no first-class config validation beyond dataclass construction. Invalid enum values, unsupported model/data combinations, impossible ratios, Transformer `d_model % nhead`, negative lags, and missing required paths are discovered late.
- Model selection is hard-coded in `runner.py` using `if/elif`. Adding GRU, TCN, XGBoost, or LightGBM would require edits in orchestration code.
- Runtime controls are hidden inside `Trainer`: device auto-detection, `num_workers`, `pin_memory`, `prefetch_factor`, AMP behavior, scheduler, gradient clipping, and `torch.compile` cannot be controlled from CLI/config except `torch_compile`.
- The Colab script bypasses the main experiment runner and expects already split CSVs with hard-coded default feature names. It does not produce the same artifact schema as `solrad-run`.
- Output reproducibility is partial. Config, metrics, predictions, preprocessing state, model checkpoint, and datasets are saved, but there is no run metadata file with git commit, environment, device details, CUDA availability, package versions, parameter count, duration, best epoch, or resolved DataLoader settings.
- `ExperimentReport.train_history` is not populated by `run_experiment()` for neural models, because `TorchRegressorModel.fit()` discards the trainer history.

## 3. Critical risks before heavy training

- Dense sequence materialization can cause out-of-memory failures before the GPU is used.
- Current in-memory best-checkpointing deep-copies model, optimizer, scheduler, and scaler states. This is acceptable for small models but expensive for large models and long training.
- There is no configurable checkpoint directory or save cadence. A Colab runtime interruption can lose progress until the final `model.save()`.
- The main runner cannot run dry validation or a synthetic smoke path from config/CLI.
- DataLoader settings are fixed heuristics, not explicit experiment metadata. This makes Colab performance harder to tune and reproduce.
- PyTorch AMP is automatic on CUDA, but not explicitly recorded or controllable.
- `torch.compile` is opt-in, but failure is only debug-logged. The resolved compiled/non-compiled state is not saved.
- `scripts/train_colab.py` uses `SequenceDataset(x_feat, y_feat)` with 2-D features, while `SequenceDataset` documents and expects 3-D sequence tensors. That script is likely not aligned with the package's real sequence pipeline.
- `notebooks/tcc/02_colab_training.ipynb` shows `model = LSTMRegressor(...); model.load(...)`, but `load()` is a classmethod returning a model. The notebook example is likely stale.

## 4. Performance bottleneck candidates

- `load_sensor_hourly()` always uses `pandas.read_csv()` and loads the full file into memory. There is no Parquet path, column projection, dtype control, chunking, or row limiting.
- Feature engineering repeatedly returns new DataFrames via `copy()` and `pd.concat()`. This is fine at small scale but copies full frames after temporal features, cyclic features, lag features, rolling features, and diff features.
- `PreprocessingPipeline.transform()` copies selected columns, creates missing columns, reorders, imputes, and scales by creating new DataFrames/Series operations. This is clear but not memory-frugal.
- `TabularDataset.from_dataframe()` copies a DataFrame subset and converts to `float32` arrays.
- `create_sequences()` uses `sliding_window_view()` initially, but then calls `np.ascontiguousarray(...)`, materializing all windows.
- `SequenceDataset.__init__()` calls `np.ascontiguousarray(...)` again and converts to torch tensors, creating another copy if the input is not already exactly contiguous `float32`.
- Resolved: sequence artifacts now use `windowed_sequences.npz` with base arrays and window metadata instead of dense `sequences.npz`.
- `TorchRegressorModel.predict()` wraps existing dataset tensors in a `TensorDataset`, uses a DataLoader with default `num_workers=0` and no explicit pinned memory, and accumulates predictions in a Python list before concatenating.
- In-memory best checkpointing in `Trainer` deep-copies state dicts on every validation improvement.

## 5. Refactor open points grouped by priority

### P0: must fix before serious training

#### P0.1 - Replace dense sequence windows with lazy or cached sequence datasets [IMPLEMENTED]

- Current problem: `create_sequences()` materializes `(n - sequence_length, sequence_length, n_features)` as contiguous `float32`, and `SequenceDataset` converts that to torch tensors. This multiplies memory by the window length.
- Why it matters: Long Colab/GPU runs will be limited by host RAM and Drive/disk I/O before model compute. A 1,000,000 row, 40 feature, 24-step dataset is no longer a 160 MB feature matrix; dense windows are several GB before overhead.
- Affected files: `src/solrad_correction/features/sequence.py`, `src/solrad_correction/datasets/sequence.py`, `src/solrad_correction/experiments/runner.py`, `tests/tcc/test_sequence_dataset.py`.
- Suggested implementation: Add a `WindowedSequenceDataset` that stores base 2-D `features`, 1-D `target`, `sequence_length`, and optional target offset. `__len__` returns `len(features) - sequence_length`; `__getitem__` slices `features[idx:idx + sequence_length]` and returns `target[idx + sequence_length]`. Support numpy arrays, torch tensors, and `np.memmap`. Keep `create_sequences()` for backward compatibility and tests, but make runner choose lazy dataset by default for neural models.
- Risk level: Medium. Target alignment and prediction index semantics must remain unchanged.
- Estimated complexity: Medium.
- Required tests: Window content/target alignment, equality with current `create_sequences()`, no copy behavior using `np.shares_memory` where applicable, DataLoader batch shape, short data error, prediction index compatibility.
- Delegation: Yes. Can be implemented independently if public sequence semantics are specified first.

#### P0.2 - Add validated runtime/training configuration [IMPLEMENTED]

- Current problem: `ModelConfig` contains model hyperparameters but not explicit runtime controls for device, workers, pin memory, AMP, compile behavior, checkpointing, or profiling. `Trainer` chooses these internally.
- Why it matters: Colab GPU training and local CPU development need different defaults, but both must be reproducible from saved config.
- Affected files: `src/solrad_correction/config.py`, `src/solrad_correction/training/trainer.py`, `src/solrad_correction/training/loops.py`, `src/solrad_correction/cli.py`, `configs/tcc/experiments/*.yaml`, docs/tests.
- Suggested implementation: Add backward-compatible optional fields, either in `ModelConfig` or a new `RuntimeConfig`: `device: auto|cpu|cuda`, `num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`, `amp`, `torch_compile`, `gradient_clip`, `scheduler`, `checkpoint_dir`, `checkpoint_every`, and `resume`. Defaults should preserve current behavior. Validate unsupported combinations, especially `prefetch_factor` when `num_workers=0`.
- Risk level: Medium.
- Estimated complexity: Medium.
- Required tests: Config load defaults, explicit CPU config, mocked CUDA config, DataLoader args resolution, invalid enum/negative worker validation, CLI override precedence.
- Delegation: Yes. Coordinate with P0.3 and P1.1 because they consume the same config.

#### P0.3 - Implement robust checkpoint save/resume during training [IMPLEMENTED]

- Current problem: Checkpoints are saved only when `model.save()` is called after training, while best weights are held in memory during training. Runtime interruption in Colab can lose progress.
- Why it matters: Colab sessions are interruptible. Heavy training needs durable checkpoints on Google Drive and safe resume from model/optimizer/scheduler/scaler states.
- Affected files: `src/solrad_correction/training/trainer.py`, `src/solrad_correction/training/callbacks.py`, `src/solrad_correction/models/torch_base.py`, `src/solrad_correction/utils/serialization.py`, `tests/tcc/test_training_smoke.py`.
- Implemented shape: `Trainer` writes `last.pt` and `best.pt` during training. Resume is through `runtime.resume` / `--resume`; `model.pretrained_path` was removed to keep one canonical resume mechanism.
- Risk level: Medium to high. Resume epoch semantics are easy to get wrong.
- Estimated complexity: Medium.
- Required tests: Save best/last, resume increments epochs correctly, optimizer state restored, incompatible optimizer gracefully warns, CPU-only checkpoint works, prediction equality after load.
- Delegation: Yes, but should follow P0.2 config names.

#### P0.4 - Split `run_experiment()` into composable pipeline stages [IMPLEMENTED]

- Current problem: One function in `experiments/runner.py` owns the entire workflow and model-specific branches.
- Why it matters: Hard to add dry-run, profile, smoke-test, cached datasets, model registry, or alternate Colab entrypoint without duplicating or complicating the god-function.
- Affected files: `src/solrad_correction/experiments/runner.py`, `src/solrad_correction/cli.py`, tests for runner/evaluation policy.
- Suggested implementation: Extract pure or near-pure stages: `load_data(config)`, `build_features(df, config)`, `split_data(df, config)`, `fit_preprocessor(train, config)`, `transform_splits(...)`, `build_datasets(...)`, `build_model(...)`, `train_model(...)`, `evaluate_model(...)`, `write_artifacts(...)`. Preserve `run_experiment(config)` as the public compatibility wrapper.
- Risk level: Medium. Behavioral drift in metrics/predictions is the main risk.
- Estimated complexity: High if done with full test coverage.
- Required tests: Golden synthetic end-to-end for SVM and LSTM, artifact names unchanged, prediction schema unchanged, evaluation policy unchanged.
- Delegation: Partially. Better as one coordinating agent or a staged refactor to avoid conflicts.

#### P0.5 - Unify Colab execution with the main experiment runner [IMPLEMENTED]

- Current problem: `scripts/train_colab.py` is separate from `solrad-run`, uses independent arguments, expects split CSVs, likely passes 2-D arrays into `SequenceDataset`, and writes only a model artifact.
- Why it matters: Heavy training should produce the same metrics, predictions, preprocessing state, config snapshot, checkpoints, and metadata as local runs.
- Affected files: `scripts/train_colab.py`, `notebooks/tcc/02_colab_training.ipynb`, `src/solrad_correction/cli.py`, docs.
- Suggested implementation: Convert the Colab path to call `solrad-run --config ...` or a thin Python wrapper around `ExperimentConfig.from_yaml()` and `run_experiment()`. Keep `scripts/train_colab.py` as a compatibility shim if needed, but make it generate/load an experiment config and use the same runner. Add explicit Drive paths for data/config/output/checkpoints.
- Risk level: Medium.
- Estimated complexity: Medium.
- Required tests: CLI help, `--validate-config`/dry-run path, script argument parsing with synthetic temp CSV/config, no real data.
- Delegation: Yes, after P0.2 and P0.4 interfaces are defined.

### P1: should fix soon

#### P1.1 - Add config validation and resolved-config printing [IMPLEMENTED]

- Current problem: Dataclasses accept arbitrary invalid values until downstream code fails.
- Why it matters: Bad configs waste Colab GPU time and can produce subtly invalid scientific comparisons.
- Affected files: `src/solrad_correction/config.py`, `src/solrad_correction/cli.py`, tests.
- Suggested implementation: Add `ExperimentConfig.validate()` and `to_dict()`/`resolved()` helpers, or migrate internally to Pydantic models while keeping YAML schema compatible. Validate model type, evaluation policy, split ratios, positive sequence length/batch size/epochs, scaler/imputer enums, rolling aggregation names, Transformer divisibility, CUDA request when unavailable if strict mode is enabled, and required data paths for real runs. Add CLI flags `--validate-config` and `--print-config`.
- Risk level: Low to medium.
- Estimated complexity: Medium.
- Required tests: Valid configs pass, invalid enum fails clearly, Transformer divisibility, missing path behavior can be skipped during dry-run/smoke but enforced for real run.
- Delegation: Yes.

#### P1.2 - Add model registry/factory [IMPLEMENTED]

- Current problem: SVM/LSTM/Transformer selection is hard-coded in `runner.py`.
- Why it matters: New models require modifying orchestration code and tests in multiple places.
- Affected files: `src/solrad_correction/models/__init__.py`, new `src/solrad_correction/models/registry.py`, `src/solrad_correction/experiments/runner.py`, docs/tests.
- Suggested implementation: Add a registry mapping `model_type` to factory functions and model kind (`tabular` or `sequence`). Keep existing `from_config()` constructors. Use registry in the runner. Do not add new model classes yet.
- Risk level: Low.
- Estimated complexity: Small to medium.
- Required tests: Registry contains existing names, unknown model error, factory produces expected model type, runner still works.
- Delegation: Yes.

#### P1.3 - Preserve and expose training history [IMPLEMENTED]

- Current problem: `Trainer.train()` returns history, but `TorchRegressorModel.fit()` discards it and `run_experiment()` creates `ExperimentReport` with empty `train_history`.
- Why it matters: Loss curves are essential for diagnosing underfit/overfit and for reproducibility of neural experiments.
- Affected files: `src/solrad_correction/models/torch_base.py`, `src/solrad_correction/experiments/runner.py`, `src/solrad_correction/evaluation/reports.py`, tests.
- Suggested implementation: Store history on the model or return a `TrainingResult` object from fit/training stage. Populate `ExperimentReport.train_history` while preserving report JSON and CSV names.
- Risk level: Low.
- Estimated complexity: Small.
- Required tests: Neural run writes `training_history.csv`, row count matches epochs, SVM does not require history.
- Delegation: Yes.

#### P1.4 - Save richer run metadata [IMPLEMENTED]

- Current problem: Artifacts do not include git commit, dirty state, environment, Python/package versions, device, CUDA availability, hostname, parameter count, duration, resolved DataLoader settings, AMP/compile status, best epoch, or dataset shape summaries.
- Why it matters: Scientific reproducibility requires knowing exactly what trained the model and where.
- Affected files: `src/solrad_correction/evaluation/reports.py`, `src/solrad_correction/experiments/runner.py`, new `src/solrad_correction/utils/metadata.py`, docs/tests.
- Implemented shape: write `metadata/metadata.json` in the v2 artifact layout. Root-level legacy artifact paths are not duplicated; metrics, predictions, configs, preprocessing, and models live in their canonical subdirectories. Metadata capture remains best-effort and does not fail the experiment if git or CUDA metadata is unavailable.
- Risk level: Low.
- Estimated complexity: Medium.
- Required tests: Metadata file exists in synthetic run, contains required keys, gracefully handles missing git.
- Delegation: Yes.

#### P1.5 - Improve artifact layout [IMPLEMENTED]

- Current problem: Current layout is flat and partially documented. Checkpoints/logs/metadata/profiles/caches do not have a stable place.
- Why it matters: Long experiments produce multiple artifacts, and Colab Drive outputs need predictable organization.
- Affected files: `src/solrad_correction/evaluation/reports.py`, `src/solrad_correction/experiments/runner.py`, docs.
- Implemented shape: root-level legacy files were removed in favor of one canonical v2 layout: `configs/`, `metrics/`, `predictions/`, `metadata/`, `preprocessing/`, `models/`, `checkpoints/`, `datasets/{train,val,test}`, `logs/`, `profiles/`, and `cache/`. Runs write `manifest.json` with schema version, artifact paths, byte sizes, and checksums. Validation and docs now describe the migration.
- Risk level: Low.
- Estimated complexity: Medium.
- Required tests: manifest exists, canonical paths exist, validation and prediction schema still pass on synthetic SVM and LSTM runs.
- Delegation: Yes.

#### P1.6 - Add data loading backends and development limits

- Current problem: The loader supports full CSV only for hourly data and does not expose column projection, dtype mapping, Parquet, Arrow, chunking, or `limit_rows`.
- Why it matters: Large WRF-derived or merged tabular files need memory-efficient loading and fast iterative CPU development.
- Affected files: `src/solrad_correction/data/loaders.py`, `src/solrad_correction/config.py`, `src/solrad_correction/cli.py`, tests/docs.
- Suggested implementation: Add config fields for `file_format: auto|csv|parquet`, `columns`, `dtype`, `parse_dates`, `limit_rows` for development only, and `cache_dir`. Prefer Parquet/Arrow for heavy workflows. Keep existing `hourly_data_path` behavior unchanged.
- Risk level: Medium.
- Estimated complexity: Medium.
- Required tests: Synthetic CSV and Parquet loading, column projection, limit rows, dtype behavior, no data directory access.
- Delegation: Yes.

#### P1.7 - Make preprocessing state more explicit and memory-aware

- Current problem: Preprocessing stores scaler/imputer state but not schema version, fitted row count, original dtypes, dropped-column reasons, or target/feature distinction. Transform copies the full frame.
- Why it matters: Future inference and scientific auditing need to know exactly what happened to each column. Memory copies become expensive at large scale.
- Affected files: `src/solrad_correction/data/preprocessing.py`, tests/docs.
- Suggested implementation: Add a serializable `PreprocessingState` dataclass. Record fitted columns, dropped columns with NA ratios, fill values, scaler stats, source feature/target names, and row counts. For performance, keep DataFrame API but add array-oriented transform methods after schema is fitted.
- Risk level: Medium.
- Estimated complexity: Medium.
- Required tests: Save/load compatibility with existing joblib state, dropped-column metadata, inverse transform unchanged.
- Delegation: Yes.

#### P1.8 - Add CLI operational flags without breaking old usage [IMPLEMENTED]

- Original problem: `solrad-run` had only `--config`, `--name`, and an output override.
- Why it matters: Local CPU development and Colab runs need dry-run, config validation, device override, worker tuning, profiling, smoke mode, and resume controls.
- Affected files: `src/solrad_correction/cli.py`, `src/solrad_correction/config.py`, tests/docs.
- Implemented shape: `--device auto|cpu|cuda`, `--dry-run`, `--validate-config`, `--print-config`, `--smoke-test`, `--limit-rows`, `--profile`, `--num-workers`, `--pin-memory/--no-pin-memory`, `--amp/--no-amp`, `--compile/--no-compile`, `--resume`, and canonical `--output-dir/-o`.
- Risk level: Low to medium.
- Estimated complexity: Medium.
- Required tests: Help includes flags, old invocation still parses, flag overrides config deterministically, dry-run does not load real data.
- Delegation: Yes, after P0.2 config fields are settled.

### P2: nice to have

#### P2.1 - Add synthetic benchmark suite [IMPLEMENTED]

- Current problem: There are no benchmarks for preprocessing, feature engineering, sequence generation, DataLoader throughput, training loop overhead, or checkpoint serialization.
- Why it matters: Refactors should prove performance improvements without touching real data.
- Affected files: new `benchmarks/` or `tests/benchmarks/`, docs, optional pyproject tooling.
- Implemented shape: `benchmarks/solrad_correction/` now contains synthetic scripts for loading, preprocessing, lazy sequence DataLoader throughput, and artifact/checkpoint serialization. Scripts generate data under `scratch/`, expose row/feature/window knobs, and run quickly by default.
- Risk level: Low.
- Estimated complexity: Medium.
- Required tests: Benchmark scripts import and run with small synthetic defaults.
- Delegation: Yes.

#### P2.2 - Add optional advanced tabular/tree models later

- Current problem: SVM is a useful baseline but does not scale as well to large tabular datasets, and neural sequence models may not always be the best scientific baseline.
- Why it matters: XGBoost/LightGBM/HistGradientBoosting can provide strong tabular baselines with faster training than SVR on larger data.
- Affected files: models registry, configs, docs/tests.
- Suggested implementation: Do not add now. After registry/config validation, consider `HistGradientBoostingRegressor` first because it is already in scikit-learn. Add XGBoost/LightGBM only if dependency policy accepts them.
- Risk level: Low to medium.
- Estimated complexity: Medium.
- Required tests: Synthetic fit/predict, save/load, metric row policy.
- Delegation: Yes after P1.2.

#### P2.3 - Add optional additional sequence architectures

- Current problem: LSTM and Transformer cover two important families, but GRU/TCN/N-BEATS/TFT-like models may be useful later.
- Why it matters: Architecture experimentation is easier once dataset and training runtime are stable.
- Affected files: model registry, configs, model modules, tests.
- Suggested implementation: Add GRU first because it is close to LSTM and low-risk. TCN next if convolutional temporal baselines are needed. N-BEATS/TFT-like models should wait for stronger experiment management and feature schema support.
- Risk level: Medium.
- Estimated complexity: Medium to high depending on architecture.
- Required tests: CPU smoke, shape checks, save/load, deterministic seed where feasible.
- Delegation: Yes after P0/P1 training foundations.

#### P2.4 - Strengthen static typing with Protocols and result dataclasses

- Current problem: Interfaces use `Any` in the model base and runner casts. `metrics: dict[str, callable]` is weakly typed.
- Why it matters: Stronger types would catch dataset/model mismatches like passing 2-D features to `SequenceDataset` in the Colab script.
- Affected files: `src/solrad_correction/models/base.py`, dataset modules, runner/training modules.
- Suggested implementation: Add `Protocol`s for tabular and sequence datasets, `TrainingResult`, `PredictionResult`, `ArtifactPaths`, and precise metric callable types. Keep runtime behavior unchanged.
- Risk level: Low.
- Estimated complexity: Medium.
- Required tests: MyPy remains clean, no runtime schema changes.
- Delegation: Yes.

#### P2.5 - Add lightweight profiling hooks [IMPLEMENTED]

- Current problem: No built-in timing/memory profiling for stages.
- Why it matters: Performance work needs stage-level evidence: load, feature engineering, preprocessing, sequence dataset creation, DataLoader throughput, training, prediction, save.
- Affected files: runner, new profiling utility, CLI.
- Implemented shape: `--profile` / `runtime.profile` writes `profiles/profile.json` with schema version, per-stage wall-clock seconds, and total stage seconds. The current profiler remains dependency-free and does not include memory deltas.
- Risk level: Low.
- Estimated complexity: Small to medium.
- Required tests: Profile file exists in synthetic dry/smoke run, contains stage durations.
- Delegation: Yes.

## 6. Proposed target architecture

Keep the existing public package and CLI names, but reorganize internals around explicit stages and registries:

```text
src/solrad_correction/
  config.py
    ExperimentConfig, DataConfig, SplitConfig, PreprocessConfig,
    FeatureConfig, ModelConfig, RuntimeConfig, ArtifactConfig

  data/
    loaders.py          # CSV/Parquet/Arrow loading, projection, dtype, limit_rows
    alignment.py        # sensor/WRF alignment
    splits.py           # chronological and walk-forward split policies
    preprocessing.py    # fit/transform state, schema metadata

  features/
    temporal.py
    engineering.py
    sequence.py         # dense helper plus canonical lazy window dataset/cache

  datasets/
    tabular.py
    sequence.py         # lazy/tensor/memmap-backed sequence datasets
    cache.py            # optional materialized/cache manifest helpers

  models/
    registry.py         # model_type -> factory + dataset kind
    svm.py
    lstm.py
    transformer.py

  training/
    dataloaders.py      # resolve DataLoader settings from runtime config
    trainer.py          # checkpoint-aware trainer
    loops.py
    callbacks.py

  experiments/
    pipeline.py         # stage functions
    runner.py           # compatibility wrapper around pipeline
    artifacts.py        # artifact paths, manifest, metadata

  evaluation/
    metrics.py
    reports.py
    comparison.py

  cli.py
```

The key design rule: stage functions should be independently testable with synthetic data, and `run_experiment(config)` should remain as the stable API that preserves old behavior unless new options are explicitly enabled.

## 7. Proposed implementation phases split for multiple agents

### Phase 0 - Guardrails and baselines

- Agent A: Add config validation and `--validate-config`/`--print-config`.
- Agent B: Add synthetic benchmark skeleton with small defaults.
- Agent C: Add metadata capture and training history persistence.

These can run independently because they mostly touch config/CLI, benchmarks, and reporting.

### Phase 1 - Sequence scalability

- Agent A: Implement lazy `WindowedSequenceDataset`.
- Agent B: Add memmap/cache save/load support for base arrays and lazy windows.
- Agent C: Update runner dataset creation to use lazy datasets while preserving dense helper compatibility.

Coordinate target alignment and artifact schema before starting.

### Phase 2 - Runtime and checkpointing

- Agent A: Add `RuntimeConfig` and DataLoader setting resolution.
- Agent B: Implement checkpoint save/resume with `best.pt` and `last.pt`.
- Agent C: Add CLI override flags for device/workers/AMP/compile/resume/profile.

These share config names, so one short design note should be agreed first.

### Phase 3 - Runner modularization

- Agent A: Extract loading/features/splitting/preprocessing stages.
- Agent B: Extract dataset/model/train/evaluate stages and introduce model registry.
- Agent C: Add dry-run/smoke-test path using synthetic data only.

This phase has higher merge risk because all agents may touch `runner.py`; split by file after a first mechanical extraction.

### Phase 4 - Colab operational workflow

- Agent A: Rewrite `scripts/train_colab.py` as a compatibility wrapper around config + runner.
- Agent B: Update `notebooks/tcc/02_colab_training.ipynb` with Drive mount, install, CUDA check, smoke test, resume, and artifact sync.
- Agent C: Update docs with local CPU and Colab GPU workflows.

### Phase 5 - Optional model expansion

- Add model registry-backed GRU or scikit-learn `HistGradientBoostingRegressor` only after Phases 0-4 are stable.

## 8. Files/modules likely affected

- `src/solrad_correction/config.py`: validation, runtime/artifact config, compatibility defaults.
- `src/solrad_correction/cli.py`: new flags and dry-run/validate/print behavior.
- `src/solrad_correction/experiments/runner.py`: stage extraction, dataset/model factory use, metadata/history wiring.
- `src/solrad_correction/data/loaders.py`: Parquet/Arrow/column projection/limit rows.
- `src/solrad_correction/data/preprocessing.py`: explicit state and metadata.
- `src/solrad_correction/features/sequence.py`: dense helper plus shared sequence index math.
- `src/solrad_correction/datasets/sequence.py`: lazy/memmap/tensor-backed datasets.
- `src/solrad_correction/models/base.py`: stronger interfaces or Protocols.
- `src/solrad_correction/models/lstm.py` and `transformer.py`: config/load metadata consistency.
- `src/solrad_correction/models/torch_base.py`: training history, resume distinction, prediction DataLoader controls.
- `src/solrad_correction/models/registry.py`: new module.
- `src/solrad_correction/training/trainer.py`: runtime config, checkpoints, best/last persistence.
- `src/solrad_correction/training/loops.py`: AMP and gradient clipping config.
- `src/solrad_correction/evaluation/reports.py`: metadata, manifest, history.
- `src/solrad_correction/utils/serialization.py`: checkpoint schema versioning.
- `src/solrad_correction/utils/seeds.py`: deterministic-mode options and metadata.
- `scripts/train_colab.py`: align with `solrad-run`.
- `notebooks/tcc/02_colab_training.ipynb`: update workflow.
- `configs/tcc/experiments/*.yaml`: optional new config fields with old defaults preserved.
- `docs/solrad_correction.md` and `README.md`: migration and workflow docs.
- `tests/tcc/*`: synthetic tests for all new behavior.
- `.github/workflows/ci.yml`: optional benchmark/lint/test matrix updates.

## 9. Tests that must be added

- Config validation:
  - valid existing SVM/LSTM configs load unchanged;
  - invalid model type fails clearly;
  - invalid evaluation policy fails clearly;
  - split ratios must sum to 1;
  - Transformer `d_model` must be divisible by `nhead`;
  - negative lags/windows/workers rejected.
- Chronological split/no leakage:
  - existing tests are good; add unsorted index input and duplicate timestamp policy.
- Preprocessing fit only on train:
  - existing tests are good; add serialized state roundtrip and dropped-column metadata.
- Sequence window correctness:
  - lazy dataset matches dense `create_sequences()`;
  - target index starts at `index[sequence_length]`;
  - no dense copy for lazy numpy input where feasible;
  - memmap-backed dataset returns correct batches.
- Reproducibility:
  - fixed seed gives stable small CPU neural predictions within tolerance;
  - metadata captures seed and deterministic flags.
- Checkpoint save/load/resume:
  - best and last checkpoints written;
  - resume starts from saved epoch and increments correctly;
  - optimizer/scheduler/scaler state restored when compatible;
  - old checkpoint format still loads.
- CPU training smoke:
  - keep current LSTM/Transformer smoke tests;
  - add smoke with lazy sequence dataset.
- CLI help and dry-run:
  - `solrad-run --help` includes new flags;
  - `--validate-config` does not train;
  - `--print-config` emits resolved config;
  - `--dry-run` validates stages without real data when paired with synthetic/smoke mode.
- Prediction schema compatibility:
  - `predictions.csv` keeps `y_true` and `y_pred`;
  - optional timestamp index behavior remains compatible with evaluation policy.
- Metric row alignment policies:
  - existing policy tests are good; add sequence model prediction index length and SVM common-horizon row counts in an end-to-end synthetic run.
- Colab/GPU behavior:
  - mock CUDA availability for config/device resolution;
  - guard actual CUDA tests with `pytest.mark.skipif(not torch.cuda.is_available())`.
- Data loading:
  - synthetic CSV and Parquet fixtures only;
  - column projection and `limit_rows` behavior.
- Benchmark importability:
  - small benchmark commands run on CPU with synthetic data.

## 10. Synthetic benchmark plan

Benchmarks must not read `data/`. Suggested location: `benchmarks/solrad_correction/` or `tests/benchmarks/` with opt-in execution.

- Preprocessing large DataFrame:
  - Generate `n_rows` x `n_features` DataFrame with controlled NaNs.
  - Measure `fit()`, `transform()`, peak memory, and output shape.
  - Compare current DataFrame path with future array-oriented path.
- Feature engineering lags/rolling:
  - Generate DatetimeIndex and base numeric features.
  - Sweep lags `[1, 3, 6, 24]`, rolling windows `[3, 24, 168]`, and feature counts.
  - Measure number of copies indirectly through time and memory.
- Sequence generation copy vs lazy indexing:
  - Compare current dense `create_sequences()`, lazy `WindowedSequenceDataset`, and memmap-backed lazy dataset.
  - Record construction time, RSS delta, first-batch latency, and batch throughput.
- DataLoader throughput CPU:
  - Use synthetic lazy and dense datasets.
  - Sweep `batch_size`, `num_workers`, `pin_memory`, `persistent_workers`, and `prefetch_factor`.
  - Keep Windows local default at `num_workers=0`; benchmark Linux/Colab separately.
- Small training loop CPU:
  - Train tiny LSTM/Transformer for 1-2 epochs on synthetic data.
  - Measure epoch time and confirm no GPU dependency.
- Checkpoint serialization:
  - Save/load synthetic LSTM/Transformer checkpoints with optimizer/scheduler/scaler absent or present.
  - Measure file size and time.

Example command shape:

```bash
conda run -n labmim python benchmarks/solrad_correction/sequence_dataloader.py --rows 200000 --features 32 --sequence-length 24
```

## 11. Colab readiness checklist

- Confirm Colab runtime has GPU enabled:
  - `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"`
- Mount Google Drive and define stable paths:
  - repo checkout path;
  - config directory;
  - input dataset path;
  - experiment output directory;
  - checkpoint directory.
- Install package without breaking Colab CUDA torch:
  - install the correct PyTorch CUDA wheel first if needed;
  - install this package in editable mode without reinstalling CPU torch unexpectedly.
- Copy or generate configs:
  - keep `configs/tcc/experiments/*.yaml` as source templates;
  - write run-specific configs under Drive for reproducibility.
- Run smoke tests before expensive training:
  - `python -m pytest tests/tcc -q` or a shorter `solrad-run --smoke-test`.
- Validate config:
  - `solrad-run --config ... --validate-config`;
  - `solrad-run --config ... --print-config`.
- Run heavy training:
  - use `--device cuda`, `--amp`, suitable `--batch-size`/config batch size, tuned `--num-workers`, and Drive-backed checkpoint directory.
- Resume:
  - resume from `checkpoints/last.pt` after interruption;
  - keep `best.pt` for evaluation/deployment.
- Save artifacts back to Drive:
  - `configs/config.yaml`, `configs/config_resolved.json`, `metadata/metadata.json`, `metadata/preprocessing_state.json`, `metrics/metrics.json`, `predictions/predictions.csv`, `metrics/training_history.csv`, `preprocessing/preprocessing_pipeline.joblib`, `models/model.pt`, `checkpoints/`, `logs/`, and `manifest.json`.
- Keep local CPU default:
  - configs should run on CPU with small data/smoke mode without requiring CUDA.

## 12. Backward compatibility and migration notes

- Do not rename or remove `solrad-run`.
- Keep `--config`, `--name`, and `--output-dir/-o` working for `solrad-run`.
- Keep old YAML files valid. New config fields must have defaults that reproduce current behavior.
- Preserve default `model.evaluation_policy: model_native` and current metric row semantics.
- Keep `predictions.csv` columns `y_true` and `y_pred`. If timestamps become explicit columns in the future, preserve the current indexed CSV behavior or document a schema version.
- Breaking change: root-level experiment artifacts are no longer written. Use the canonical v2 layout under `configs/`, `metrics/`, `predictions/`, `metadata/`, `preprocessing/`, `models/`, `checkpoints/`, `datasets/{train,val,test}`, `profiles/`, `logs/`, and `cache/`.
- Keep dense `create_sequences()` available for compatibility, tests, and small experiments. Add lazy datasets as the scalable default for neural training.
- Use only `runtime.resume` / `--resume` for full optimizer/scheduler/scaler continuation.
- If preprocessing state format changes, loader must support old joblib state keys.
- If using Pydantic internally, preserve YAML field names and dataclass-like access or provide a compatibility layer.
- README currently says Python 3.11+, while `pyproject.toml` requires `>=3.14` and CI uses 3.14. Align documentation before publishing install instructions.

## 13. Commands run and results

No command inspected `data/`.

Latest current-pass validation:

```bash
conda run -n labmim python -m pytest tests/ -q
```

Result:

```text
174 passed, 2 warnings in 5.31s
```

```bash
conda run -n labmim ruff check src tests scripts benchmarks
```

Result:

```text
All checks passed!
```

```bash
conda run -n labmim python -m mypy src
```

Result:

```text
Success: no issues found in 79 source files
```

```bash
conda run -n labmim python benchmarks/solrad_correction/sequence_dataloader.py --rows 512 --features 8 --sequence-length 12 --batch-size 32 --max-batches 3
```

Result:

```text
{'benchmark': 'sequence_dataloader', 'rows': 512, 'windows': 500, 'samples': 96, 'batches': 3, 'seconds': 0.001724, 'samples_per_second': 55690.915}
```

Latest implementation validation:

```bash
conda run -n labmim python -m pytest tests/ -q
```

Result:

```text
164 passed in 5.01s
```

```bash
conda run -n labmim ruff check src tests scripts\train_colab.py
```

Result:

```text
All checks passed!
```

```bash
conda run -n labmim python -m mypy src
```

Result:

```text
Success: no issues found in 78 source files
```

Previous implementation validation:

```bash
conda run -n labmim python -m pytest tests/ -q
```

Result:

```text
165 passed in 4.99s
```

```bash
conda run -n labmim ruff check src tests scripts\train_colab.py
```

Result:

```text
All checks passed!
```

```bash
conda run -n labmim python -m mypy src
```

Result:

```text
Success: no issues found in 78 source files
```

Previous implementation validation:

```bash
conda run -n labmim python -m pytest tests/ -q
```

Result:

```text
147 passed in 5.07s
```

```bash
conda run -n labmim ruff check src tests
```

Result:

```text
All checks passed!
```

```bash
conda run -n labmim python -m mypy src
```

Result:

```text
Success: no issues found in 75 source files
```

Original analysis validation:

```bash
conda run -n labmim python -m pytest tests/ -q
```

Result:

```text
123 passed in 4.97s
```

```bash
conda run -n labmim ruff check src tests
```

Result:

```text
All checks passed!
```

```bash
conda run -n labmim python -m mypy src
```

Result:

```text
Success: no issues found in 74 source files
```

Additional inspection commands used:

```bash
rg --files -g '!data/**'
git status --short
Get-Content pyproject.toml
Get-Content README.md
Get-Content docs/solrad_correction.md
Get-Content configs/tcc/experiments/lstm_hourly.yaml
Get-Content configs/tcc/experiments/svm_hourly.yaml
Get-Content scripts/train_colab.py
Get-Content .github/workflows/ci.yml
Get-Content src/solrad_correction/... selected modules
Get-Content tests/tcc/... selected tests
```

Notes:

- `rg --files -g '!data/**'` reported access denied for `scratch/pytest-audit` and `scratch/pytest-training`. This did not block the analysis.
- `git status --short` also reported those scratch permission warnings and no tracked file changes before this report was added.

## 14. Final recommendation

Refactor now, but do it in controlled phases rather than a broad rewrite. The current implementation is scientifically sensible for small experiments, and the existing tests are a good base. However, starting serious LSTM/Transformer Colab training before fixing sequence materialization, checkpoint durability, runtime configurability, and metadata capture risks wasting GPU sessions and producing artifacts that are difficult to audit or resume.

The safest path is:

1. Add validation, runtime config, metadata, and training history.
2. Replace dense sequence training with lazy/cached datasets while keeping dense helpers for compatibility.
3. Add durable checkpoints and resume.
4. Modularize the runner and align Colab with `solrad-run`.
5. Only then expand model families or run long comparative experiments.
