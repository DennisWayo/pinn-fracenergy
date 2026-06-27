[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Build Status](https://img.shields.io/badge/PINN-yes-green)
![Build Status](https://img.shields.io/badge/FDM-yes-blue)
![Contributions](https://img.shields.io/badge/contributions-welcome-gold)
![GitHub issues](https://img.shields.io/github/issues/DennisWayo/pinn-fracenergy)
![GitHub forks](https://img.shields.io/github/forks/DennisWayo/pinn-fracenergy)
![GitHub stars](https://img.shields.io/github/stars/DennisWayo/pinn-fracenergy)

# Physics-Guided Neural Regression for Hydraulic-Fracture Energy Proxies

This repository provides a physics-guided neural regression benchmark for learning a
synthetic hydraulic-fracture energy proxy across 1D, 2D, and 3D input spaces. The setup is
designed to study dimensional scaling, training stability, monotonic physical consistency,
and surrogate accuracy under a controlled target. It is not a strict PINN because no PDE
residual is included in the loss.

## Energy proxy

$$
E = \frac{P W}{\mu} \exp\left(-\frac{L}{5}\right)
$$

where:
- `L` is fracture length (m)
- `W` is fracture width (m)
- `P` is pressure gradient (MPa/m)
- `mu` is viscosity (Pa*s)

Input cases:
- 1D: `L`
- 2D: `(L, W)` at fixed `P, mu`
- 3D: `(L, W, P, log(mu))`

## Run everything

The root wrapper checks the Python environment, runs the selected workflow, writes logs to
`run_logs/`, and writes generated figures to the repository root.

```bash
bash run_all.sh
```

Useful modes:

```bash
bash run_all.sh --quick            # smoke test only; metrics are not paper-quality
bash run_all.sh --learning-curves  # long sample-size learning-curve study
bash run_all.sh --all              # main run plus learning curves
```

Required Python packages:

```bash
python3 -m pip install numpy pandas scikit-learn matplotlib seaborn tensorflow
```

You can also choose a specific Python environment:

```bash
PYTHON_BIN=/path/to/python3 bash run_all.sh
```

## Featured results

![comparison.png](comparison.png)

Highlight: One slice across `L` at fixed `W, P, mu` shows how 1D/2D/3D PINNs track the synthetic target.

![fdm_instability.png](fdm_instability.png)

Highlight: A compact stability contrasts stable vs. unstable explicit steps, underscoring why training dynamics can diverge.

## Quick start

```bash
python pinn_fracenergy.py
```

The direct Python entry point is useful for manual experimentation. For reproducible runs,
prefer `bash run_all.sh`. Edit `pinn_fracenergy_config.py` or use environment variables to
change dataset size, training schedule, plot generation, and 3D sweep options.

## Results (reproduced CPU run)

| Model | MSE | NRMSE | R^2 |
|---|---:|---:|---:|
| 1D | 0.0031 | 0.0004 | 1.0000 |
| 2D | 1.4580 | 0.0069 | 1.0000 |
| 3D | 10917.8261 | 0.0918 | 0.9916 |

The reported metrics are computed on inverse-transformed, noise-free proxy test targets
after training on noisy transformed targets.

## Outputs

Key figures produced by `pinn_fracenergy_plots.py`:

- `comparison.png`
- `fdm_instability.png`
- `trainloss.png`
- `contour_map.png`
- `3D_surface.png`
- `error_distribution.png`
- `kgd_baseline.png`
- `confusion_*.png`
- `roc_curves.png`
- `learning_curve_*.png`

## Repo layout

- `pinn_fracenergy.py`: main training/evaluation entrypoint
- `pinn_fracenergy_config.py`: experiment configuration
- `pinn_fracenergy_data.py`: synthetic data generation and scaling
- `pinn_fracenergy_model.py`: model construction, training, metrics
- `pinn_fracenergy_plots.py`: plotting utilities
- `learning_curve_study.py`: repeated sample-size/seed sweep
- `plot_learning_curves.py`: learning-curve plotting utility
- `run_all.sh`: one-command runner for the main workflow and learning curves
- `README.md`, `LICENSE`

## Citation

If you use this work, please cite the accompanying manuscript.

## Funders

- The National Key R&D Program of China (2023YFE0110900)
- National Natural Science Foundation of China (52074040)
- Nazarbayev University’s Collaborative Research Project (111024CRP2014)
