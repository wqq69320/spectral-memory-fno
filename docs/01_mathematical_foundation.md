# Mathematical Foundation

PDE forecasting seeks to approximate the evolution operator that maps one or more observed field states to future field states:

```text
u(t_0), ..., u(t_k) -> u(t_{k+1}), ..., u(t_{k+H})
```

where `u(t)` is a spatial field and `H` is the rollout horizon.

## Long-Horizon Rollout

In autoregressive forecasting, prediction errors can compound over time because each predicted state may be used as input for the next step. Evaluation should therefore include both one-step error and rollout error across many future steps.

## Planned PDE Roles

- **1D Heat equation:** a dissipative system useful for validating smooth dynamics and numerical stability.
- **1D Burgers equation:** a nonlinear advection-diffusion system useful for testing sharper dynamics.
- **2D Navier-Stokes:** a later extension for richer spatial structure and more demanding long-horizon behavior.

The scaffold does not yet provide production-quality solvers. Data generation modules contain placeholders for future numerical methods.
