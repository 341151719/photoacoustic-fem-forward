# Photoacoustic FEM Forward

[![CI](https://github.com/341151719/photoacoustic-fem-forward/actions/workflows/ci.yml/badge.svg)](https://github.com/341151719/photoacoustic-fem-forward/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

An independent, auditable two-dimensional finite-element forward model for photoacoustics:

```text
Gaussian laser fluence -> absorbed energy -> initial pressure
-> heterogeneous transient pressure acoustics
-> finite-aperture receiver -> causal bandwidth response
```

This repository reproduces the **front-end physics**, not the compressed sensing or image-reconstruction algorithm from the motivating paper.

**[Chinese documentation](README_CN.md) | [modelling plan](docs/reference/Photoacoustic_FEM_Method_CN.md) | [visual validation report](docs/VALIDATION_CN.md)**

![Transient pressure propagation](docs/assets/02_wave_propagation.png)

## Implemented physics

- Gmsh/OpenCASCADE conforming water, tissue, absorber, and line-receiver geometry.
- meshio audit of physical groups, SI units, mesh quality, duplicate nodes, and interfaces.
- scikit-fem P1 discretization of the conservative heterogeneous pressure equation.
- Positive conservative mass-lumped L2 projection of initial pressure.
- First-order Sommerfeld absorbing boundary condition on the full exterior.
- Newmark average-acceleration transient integration with reused sparse factorization.
- Finite line-aperture pressure averaging and a documented causal bandwidth response.
- Reproducible NPZ, VTU, JSON, PNG, and GIF products with versions and mesh hashes.

## Quick start

```bash
sudo apt-get update
sudo apt-get install -y libglu1-mesa

git clone https://github.com/341151719/photoacoustic-fem-forward.git
cd photoacoustic-fem-forward
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pytest -q
pa-fem self-test
```

Run the 1 MHz cases:

```bash
# Fast pipeline/debug case; intentionally under-resolved.
pa-fem solve --config configs/debug.json --outdir results/debug_1MHz

# Reference case: about 88k DOFs, 40 um mesh, 5 ns step, 12 us record.
pa-fem solve --config configs/reference.json --outdir results/reference_1MHz
```

The 10 MHz configuration is an explicitly labelled scaling example. It is not claimed to represent the paper's unreported transducer transfer function.

## Validation result

| Check | Result |
|---|---:|
| Projected initial-pressure peak | 9.941 kPa (10 kPa estimate) |
| Centre-path travel time | 3.624 us |
| Raw FEM peak time | 3.595 us (0.80% difference) |
| Reference mesh | 88,313 DOFs / 175,523 triangles |
| Reference time step | 5 ns; CFL indicator 0.305 |
| 40 vs 30 um, detected 1 MHz waveform | 1.35% relative L2 difference |
| 10 vs 5 ns, detected 1 MHz waveform | 0.88% relative L2 difference |
| Actual p0 vs 2*p0 linearity residual | 0 |
| Reference hard checks | 6/6 passed |

The unfiltered broadband trace converges more slowly (9.02% between 40 and 30 um) because it retains frequencies outside the modelled receiver band. The current convergence claim is intentionally limited to the specified 1 MHz detected bandwidth.

![Numerical validity](docs/assets/04_numerical_validity.png)

See the [complete validation report](docs/VALIDATION_CN.md) and [machine-readable summary](docs/reference/visual_validation_summary.json).

## Reference inputs

The original modelling plan is versioned under `docs/reference`. The supplied paper and reference loudspeaker-FEM archives are available as assets on the [`reference-inputs-v1` release](https://github.com/341151719/photoacoustic-fem-forward/releases/tag/reference-inputs-v1), keeping large independently licensed inputs out of Git history. See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution, licensing notes, and SHA256 checksums.

## Scope and limitations

- The 2-D model represents an out-of-plane infinite line source and line receiver.
- Tissue is an inviscid fluid; shear and power-law attenuation are omitted.
- Energy decay verifies a dissipative ABC implementation, not a -30 dB reflection bound for every incidence angle.
- The default transducer response is an engineering assumption; measured impulse-response data should replace it for experiment-level comparison.
- No coded acoustic aperture, piezoelectric coupling, compression, or reconstruction is included.

## License

Code is MIT licensed. Reference materials retain their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
