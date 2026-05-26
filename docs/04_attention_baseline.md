# Attention Baseline

Attention-based sequence models are a useful baseline because they can model long-range interactions across time by comparing tokens with one another.

## Mechanism

Self-attention computes pairwise interactions between sequence elements. For temporal sequence length `T`, the dense attention matrix has a typical cost that scales as:

```text
O(T^2)
```

This cost can become significant for long rollout horizons.

## Why Include It?

The Transformer baseline provides a reference point for expressiveness and cost. SM-FNO should be evaluated against this baseline using the same data, rollout horizon, training budget, and metrics before making any conclusions about quality or efficiency.
