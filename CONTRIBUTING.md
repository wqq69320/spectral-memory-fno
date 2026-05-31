# Contributing

This project is currently a research scaffold. Contributions should keep the repository easy to audit, reproduce, and extend.

## Guidelines

- Keep code modular and place reusable logic under `src/sm_fno`.
- Add or update plain YAML configs for new experiments.
- Do not commit large datasets, generated outputs, model checkpoints, or logs.
- Document new baselines in `README.md` and `docs/`.
- Add tests for new metrics, tensor shapes, and rollout behavior.
- Avoid claiming performance improvements without reproducible experiments, configs, metrics, and baseline comparisons.

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
