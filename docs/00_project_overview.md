# Project Overview

This repository studies attention-free long-horizon PDE forecasting using the proposed SM-FNO architecture: a Spectral Memory Fourier Neural Operator.

The project is organized around a conservative research question: can spectral spatial modeling and state space temporal memory provide a cost-efficient alternative to attention-based temporal models for PDE rollout?

## High-Level Architecture

SM-FNO is planned as a composition of:

1. A spatial encoder based on Fourier Neural Operator layers.
2. A temporal memory module based on a State Space Model.
3. A decoder that maps latent states back to PDE fields.

The repository separates data generation, model definitions, training, evaluation, and visualization so that baselines can be added without rewriting the experiment pipeline.

## Research Principles

- Compare against simple and strong baselines.
- Use the same data splits and rollout horizons across models.
- Report accuracy and cost metrics.
- Avoid performance claims until experiments are reproducible.
