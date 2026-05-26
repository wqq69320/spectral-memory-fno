# Contributing

This project is a research-oriented PyTorch codebase for attention-free

long-horizon PDE forecasting with spectral neural operators and state space

memory. Contributions should keep the repository easy to audit, reproduce, and

extend.

## Guidelines

- Keep code modular and place reusable logic under `src/sm_fno`.

- Add or update plain YAML configs for new experiments.

- Do not commit large datasets, generated outputs, model checkpoints, figures,

  metric tables, or logs.

- Document new baselines, PDE tasks, metrics, or protocol changes in `README.md`

  and `docs/`.

- Add tests for new metrics, tensor shapes, data generators, rollout behavior,

  and config compatibility.

- Keep experiment outputs under ignored local paths such as `data/processed/`,

  `outputs/`, and `results/`.

- Avoid claiming performance improvements without reproducible experiments,

  frozen configs, metrics, plots, and baseline comparisons.

- Clearly label smoke runs, protocol-scale sanity checks, and benchmark-scale

  experiments.

## Development

Install the development dependencies:

```bash
pip install -e ".[dev]"
```

Run checks before opening a pull request:

```bash
pytest
ruff check .
black --check .
```
