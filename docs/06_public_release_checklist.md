# Public Release Checklist

Use this checklist before making the repository public.

- Confirm all source files have clear docstrings and type hints where useful.
- Remove private paths, local machine references, and temporary notes.
- Verify `.gitignore` excludes large data, checkpoints, logs, and generated outputs.
- Run tests with `pytest`.
- Run format and lint checks.
- Document all implemented models and baselines.
- Document dataset generation steps and expected storage layout.
- Publish exact configs for reported experiments.
- Include tables and figures only when generated from reproducible runs.
- Update `CITATION.cff` with final authors, release date, and repository URL.
- Replace `TODO` fields in the license and citation metadata.
