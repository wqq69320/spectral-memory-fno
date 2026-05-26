# Experiment Protocol

This document defines the intended rules for fair comparison.

## Data Splits

Use fixed train, validation, and test splits generated from a documented random seed. All models should use the same split files for a given dataset.

## Rollout Horizon

Evaluate each model on the same one-step and long-horizon rollout settings. Report errors per step and aggregated across the full horizon.

## Training Budget

Baselines should use comparable training budgets where possible, including epoch count, batch size, optimizer, learning rate schedule, and early stopping rules.

## Metrics

Report accuracy and cost metrics:

- Relative L2 error
- Long-horizon rollout error
- Inference time
- Memory usage
- Energy error
- Fourier spectrum error

## Reporting

Do not report a model as outperforming another without reproducible configs, seeds, dataset generation details, and saved evaluation outputs.
