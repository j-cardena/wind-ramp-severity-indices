# MATLAB implementation

This folder provides a MATLAB implementation of the core ramp severity indices in `src/ramp_indices.py`:

- RAI: Ramp Acceleration Index
- RSCI: Ramp Shape Complexity Index
- OSI: Operational Stress Index
- GIP: Grid Impact Potential
- ECSI: Enhanced Composite Severity Index (weighted sum of normalized components)

## Requirements

- MATLAB R2022b+ (tested target)
- No toolboxes required for the core computations.
- Plotting uses base MATLAB.

## Quick start

From the repository root in MATLAB:

```matlab
cd matlab
run('scripts/run_all.m')
```

This will:
1. Load `../data/sample_ramps.csv`
2. Compute all indices for each ramp
3. Write `../results/matlab_ramp_indices.csv`
4. Save a couple of figures in `../results/figures/`

## Notes on numerical parity

- The formulas match the Python reference implementation.
- Small floating-point differences are expected.
- ECSI uses batch min-max normalization bounds computed from the provided ramps (same two-pass approach as Python `calculate_batch`).

## File layout

- `scripts/run_all.m`: runnable entry point
- `src/`: core functions
- `tests/`: minimal unit tests (MATLAB `matlab.unittest`)
