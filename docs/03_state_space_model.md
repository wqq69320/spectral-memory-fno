# State Space Model

State Space Models maintain a hidden state that evolves over time. A simple linear recurrence can be written as:

```text
h_t = A h_{t-1} + B x_t
y_t = C h_t + D x_t
```

where `x_t` is the input at time `t`, `h_t` is the memory state, and `y_t` is the output.

## Temporal Scaling

Unlike full self-attention over a sequence, recurrent or structured state space updates can scale linearly with sequence length during rollout. This makes them appealing for long-horizon PDE forecasting, where the number of temporal steps can be large.

## Planned Use in SM-FNO

The SSM component will receive latent representations from the FNO spatial encoder and update temporal memory across rollout steps. Initial versions may use a simple recurrent state update before moving to more expressive structured SSM variants.
