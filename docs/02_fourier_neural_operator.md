# Fourier Neural Operator

Fourier Neural Operators model spatial fields by learning transformations in the frequency domain. They are attractive for PDE tasks because many physical fields have structure that can be represented compactly using low-frequency spectral modes.

## Spectral Convolution Intuition

A typical FNO block follows this pattern:

```text
field -> FFT -> learned spectral weights -> inverse FFT -> updated field
```

The learned spectral weights act on selected Fourier modes, while pointwise layers preserve local channel mixing. This can provide a global spatial receptive field without using attention over all grid points.

## Planned Use in SM-FNO

In SM-FNO, FNO layers will be used to encode and update spatial structure at each time step. Temporal memory will be handled separately by an SSM module so that the model can scale to longer sequences without relying on quadratic temporal attention.
