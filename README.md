# SIMBI

```
  ███████╗██╗███╗   ███╗██████╗ ██╗
  ██╔════╝██║████╗ ████║██╔══██╗██║
  ███████╗██║██╔████╔██║██████╔╝██║
  ╚════██║██║██║╚██╔╝██║██╔══██╗██║
  ███████║██║██║ ╚═╝ ██║██████╔╝██║
  ╚══════╝╚═╝╚═╝     ╚═╝╚═════╝ ╚═╝
```

<div align="center">

**High-performance 3D relativistic magneto-gas dynamic code for astrophysical fluid simulations**

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-orange.svg?style=for-the-badge&logo=c%2B%2B)](https://en.cppreference.com/w/cpp/20)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/downloads/)
[![CUDA Support](https://img.shields.io/badge/CUDA-Supported-76B900.svg?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![AMD Support](https://img.shields.io/badge/AMD-Supported-ED1C24.svg?style=for-the-badge&logo=amd)](https://rocm.docs.amd.com/)

**[Quick Start](#quick-start) • [Installation](#installation) • [Usage](#usage) • [Publications](#publications)**

</div>

---

## Overview

SIMBI is a finite volume code for astrophysical fluid simulations. Results from SIMBI simulations have been published in *The Astrophysical Journal* and *The Astrophysical Journal Letters*, studying relativistic jets, shock morphology, and stellar explosions.

**Features:**
- Special Relativistic Magnetohydrodynamics (SRMHD), Special Relativistic Hydrodynamics (SRHD), and Newtonian Hydrodynamics
- GPU acceleration via CUDA (NVIDIA) and HIP (AMD)
- High-resolution shock capturing with HLLE, HLLC, and HLLD Riemann solvers
- Immersed boundary method (Peskin 2002) for solid objects in the computational domain
- Adaptive mesh refinement with Berger-Colella subcycling
- Entity-component-system architecture for partition-aware multi-device execution
- Type-safe Python configuration with automatic CLI generation
- Afterglow radiation transport and observables

---

## Simulation Gallery

<div align="center">

| Relativistic Jet Evolution | Relativistic Shock Tube | Rayleigh-Taylor Instability |
|:---:|:---:|:---:|
| [Animation](https://user-images.githubusercontent.com/29236993/145315802-c8d7e8c5-7beb-488c-b496-b9edf404be2e.mp4) | [Animation](https://user-images.githubusercontent.com/29236993/212521070-0e2a7ced-cd5f-4006-9039-be67f174fb07.mp4) | [Animation](https://github.com/EigenDev/simbi/assets/29236993/818d930d-d993-4e5d-8ed4-47a9bae11a7f) |

| Moving Mesh Techniques | Magnetic Turbulence |
|:---:|:---:|
| [Animation](https://user-images.githubusercontent.com/29236993/205418982-943af187-8ae3-4401-92d5-e09a4ea821e2.mp4) | [Animation](https://github.com/user-attachments/assets/9e5b8c42-ce3e-4c23-a380-7903eec52b92) |

</div>

---

## Quick Start

Run the Marti & Müller relativistic shock tube test:

**CPU:**
```bash
# install
CC=gcc CXX=g++ python dev.py install

# run test problem
simbi run marti-muller --mode cpu --resolution 400

# visualize
simbi plot data/1000.chkpt.000_400.h5 --setup "Marti & Muller Problem 1" --field rho v p
```

**GPU (auto-detects architecture):**
```bash
CC=gcc CXX=g++ python dev.py install --gpu
simbi run marti-muller --mode gpu --resolution 1024
```

---

## Installation

### Requirements

**Minimum:**
- gcc ≥ 8 or clang ≥ 10
- Python 3.10+
- 8 GB RAM
- Linux/macOS

**Recommended:**
- Latest stable compiler
- Python 3.11+
- 32+ GB RAM for large 3D simulations

### Dependencies

- **Build**: Meson ≥ 1.4.0, Ninja
- **Libraries**: pybind11, HDF5, OpenMP
- **Python**: pydantic, rich, hdf5

### Installation Commands

**Standard (editable install):**
```bash
CC=gcc CXX=g++ python dev.py install -e
```

**With visualization tools:**
```bash
CC=gcc CXX=g++ python dev.py install --visual-extras
```

**GPU compilation (NVIDIA, auto-detects architecture):**
```bash
CC=gcc CXX=g++ python dev.py install --gpu
```

**GPU compilation (explicit architecture):**
```bash
# V100 (compute capability 7.0)
CC=gcc CXX=g++ python dev.py install --gpu --device-arch sm_70

# A100 (compute capability 8.0)
CC=gcc CXX=g++ python dev.py install --gpu --device-arch sm_80
```

**GPU compilation (AMD):**
```bash
# MI100 (gfx908)
CC=gcc CXX=g++ python dev.py install --gpu --device-arch gfx908
```

### Advanced Build Options

| Option | Description |
|--------|-------------|
| `--precision single\|double` | Floating point precision (default: double) |
| `--column-major` | Use column-major data layout |
| `--four-velocity` | Use four-velocity as primitive variable |
| `--unified-memory` | CUDA unified memory (default: device memory) |
| `--build-tests` | Build test suite |
| `--linker mold\|lld\|gold\|bfd` | Select linker (auto-detects fastest) |
| `--gpu-jobs N` | Parallel jobs for GPU compilation |
| `--timeout N` | Build timeout in seconds |
| `--reconfigure` | Force meson reconfiguration |

**Clean and rebuild:**
```bash
python dev.py clean --all
python dev.py install --gpu
```

---

## Usage

### CLI Commands

SIMBI provides three main commands:

```bash
simbi run        # run simulations
simbi plot       # visualize checkpoint data
simbi afterglow  # radiation transport and observables
```

### Running Simulations

```bash
# basic usage
simbi run marti-muller --mode gpu --resolution 400

# list available parameters for a problem
simbi run <problem> --info

# list all available problem configs
simbi run --configs

# custom config path
simbi run simbi_configs/examples/kh.py --mode cpu --resolution 512

# resume from checkpoint
simbi run <problem> --checkpoint data/checkpoint.h5
```

**Common options:**
- `--mode cpu|gpu` - execution mode
- `--resolution N` or `--resolution N M` or `--resolution N M K` - grid resolution
- `--adiabatic-index` - ratio of specific heats
- `--end-time` - simulation end time
- `--data-directory` - output directory

### Visualization

```bash
# plot checkpoint fields
simbi plot data/checkpoint.h5 --setup "Problem Name" --field rho v p

# plot with body diagnostics
simbi plot data/checkpoint.h5 --bodies

# create animation
simbi plot data/*.h5 --animate --field rho

# generate example config
simbi plot --generate-config
```

### Afterglow Analysis

Generate synthetic observables from simulation data:

```bash
# generate photon events from hydro snapshots
simbi afterglow generate data/*.h5 --output events.h5 --max-events 1000000

# compute observer lightcurve
simbi afterglow lightcurve events.h5 --observer-angle 0.1 --frequencies 1e9 1e14 1e18

# generate sky intensity map
simbi afterglow skymap events.h5 --observer-time 1e5

# compute polarization evolution
simbi afterglow polarization events.h5 --observer-angle 0.1

# generate spectrum
simbi afterglow spectrum events.h5 --observer-time 1e5
```

---

## Configuration System

SIMBI uses type-safe configuration with automatic CLI generation. Problems inherit from `SimbiProblem` and use `ProblemParam` for field metadata.

### Basic Structure

```python
from pathlib import Path
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import (
    BoundaryCondition,
    CoordSystem,
    Regime,
    Solver,
)

class KelvinHelmholtz(SimbiProblem):
    """kelvin-helmholtz instability in newtonian fluid."""

    # physics parameters
    adiabatic_index: Annotated[
        float, ProblemParam(5.0 / 3.0, description="adiabatic index")
    ]
    rhoL: Annotated[float, ProblemParam(2.0, description="density in central layer")]
    rhoR: Annotated[float, ProblemParam(1.0, description="density in outer regions")]

    # domain configuration
    resolution: Annotated[
        tuple[int, int],
        ProblemParam((256, 256), cli=True, description="grid resolution"),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(-0.5, 0.5), (-0.5, 0.5)], description="domain boundaries"),
    ]
    coord_system: Annotated[
        CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="coordinate system")
    ]
    regime: Annotated[Regime, ProblemParam(Regime.NEWTONIAN, description="physics regime")]

    # numerics
    boundary_conditions: Annotated[
        BoundaryCondition, ProblemParam(BoundaryCondition.PERIODIC)
    ]
    solver: Annotated[Solver, ProblemParam(Solver.HLLC, description="riemann solver")]

    # simulation control
    data_directory: Annotated[
        Path,
        ProblemParam(Path("data/kh"), cli=True, checkpoint_safe=True),
    ]
    end_time: Annotated[
        float, ProblemParam(20.0, cli=True, checkpoint_safe=True)
    ]

    def initial_primitive_state(self):
        """generate initial conditions."""
        def gas_state():
            nx, ny = self.resolution
            for jj in range(ny):
                for ii in range(nx):
                    # compute y coordinate
                    y = self.bounds[1][0] + jj * (self.bounds[1][1] - self.bounds[1][0]) / ny
                    if abs(y) < 0.25:
                        yield (self.rhoL, 0.5, 0.0, 2.5)  # rho, vx, vy, p
                    else:
                        yield (self.rhoR, -0.5, 0.0, 2.5)
        return gas_state
```

### ProblemParam Options

| Option | Description |
|--------|-------------|
| `cli=True` | Expose as CLI argument |
| `checkpoint_safe=True` | Can override when resuming from checkpoint |
| `description="..."` | Help text for CLI |
| `ge=`, `le=`, `gt=`, `lt=` | Validation constraints |

### Source Terms

Add gravity or custom hydro sources via expression graphs:

```python
import simbi

@computed_field
@property
def gravity_source_expressions(self):
    graph = simbi.Expr.Graph()
    x_comp = simbi.Expr.constant(0.0, graph)
    y_comp = simbi.Expr.constant(-0.1, graph)  # constant downward gravity
    terms = graph.compile([x_comp, y_comp])
    return terms.serialize()
```

### Immersed Bodies

Define solid objects in the computational domain:

```python
from simbi.types import ImmersedBodyConfig, BodyCapability, GravitationalProperties

@computed_field
@property
def immersed_bodies(self) -> list[ImmersedBodyConfig]:
    return [
        ImmersedBodyConfig(
            name="central_mass",
            capabilities=[BodyCapability.GRAVITATIONAL],
            gravitational=GravitationalProperties(
                mass=1.0,
                softening_length=0.01,
            ),
            position=(0.0, 0.0),
            velocity=(0.0, 0.0),
        )
    ]
```

### Dynamic Mesh Motion

For expanding or contracting domains:

```python
@computed_field
@property
def scale_factor(self):
    return lambda time: 1.0 + 0.1 * time

@computed_field
@property
def scale_factor_derivative(self):
    return lambda time: 0.1
```

---

## Physics

### Regimes

| Regime | Description | Use Cases |
|--------|-------------|-----------|
| `SRMHD` | Special Relativistic Magnetohydrodynamics | AGN jets, pulsar wind nebulae, magnetic reconnection |
| `SRHD` | Special Relativistic Hydrodynamics | Gamma-ray bursts, relativistic shocks, stellar explosions |
| `NEWTONIAN` | Classical Hydrodynamics | Stellar winds, ISM dynamics, classical turbulence |

### Coordinate Systems

- `CARTESIAN` - x, y, z
- `SPHERICAL` - r, θ, φ
- `CYLINDRICAL` - r, φ, z
- `AXIS_CYLINDRICAL` - cylindrical with axis symmetry
- `PLANAR_CYLINDRICAL` - 2D cylindrical in r-z plane

### Numerical Methods

**Riemann Solvers:**
- `HLLE` - HLL solver with entropy fix
- `HLLC` - HLL Contact solver (hydrodynamics)
- `HLLD` - HLL Discontinuities solver (magnetohydrodynamics)

**Grid Spacing:**
- `LINEAR` - uniform spacing
- `LOGARITHMIC` - log spacing (useful for spherical)

**Boundary Conditions:**
- `PERIODIC` - wrap around
- `REFLECTING` - mirror symmetry
- `OUTFLOW` - zero gradient
- `DYNAMIC` - user-defined expressions

**Time Integration:**
- `EULER` - Forward Euler
- `RK2` - Second-order Runge-Kutta (Berger-Colella for AMR)

**Additional Options:**
- `plm_theta` - PLM reconstruction parameter (0-2, default 1.5)
- `use_quirk_smoothing` - Quirk's carbuncle fix
- `use_fleischmann_limiter` - Low-Mach fix for HLLC

### Static Mesh Refinement

```python
# enable refinement
refinement_enabled: Annotated[bool, ProblemParam(True)]
refinement_max_levels: Annotated[int, ProblemParam(3)]
refinement_regions: Annotated[
    list[list[tuple[float, float]]],
    ProblemParam([[(-0.1, 0.1), (-0.1, 0.1)], [(-0.05, 0.05), (-0.05, 0.05)]]),
]
refinement_ratios: Annotated[list[int], ProblemParam([2, 2])]
refinement_subcycling_mode: Annotated[
    SubCycleMode, ProblemParam(SubCycleMode.STANDARD)
]
```

**Subcycling Modes:**
- `NONE` - all levels advance with same timestep
- `STANDARD` - subcycle by refinement ratio
- `MANUAL` - user-specified substeps per level
- `ADAPTIVE` - (not yet implemented)

---

## Architecture

### Entity-Component-System Design

SIMBI uses an ECS architecture for partition-aware multi-device execution:

**Components:**
- `simulation_t<Rank, Regime, CoordSystem>` - top-level simulation state
- `partition_t<Rank>` - device assignment + execution stream for one domain partition
- `level_decomposition_t<Rank>` - all partitions + halo graph for one AMR level
- `partition_fields_t` - hydro fields (cons, prim, flux, bfield, efield) for one partition
- `flux_register_component_t` - AMR flux correction registers

**Systems:**
- `timestep_system_t` - compute CFL-limited timesteps with subcycling
- `ghost_fill_system_t` - fill ghost cells via physical BCs or coarse grid prolongation
- `c2p_system_t` - conservative to primitive variable conversion
- `flux_system_t` - compute numerical fluxes via Riemann solver
- `euler_system_t` / `rk2_stage1_system_t` / `rk2_stage2_system_t` - time integration
- `restriction_system_t` - inject fine grid interior to coarse grid
- `prolongation_system_t` - interpolate coarse grid to fine grid boundaries
- `reflux_system_t` - apply flux correction at AMR boundaries
- `body_effects_system_t` - immersed boundary forces and diagnostics

Each system operates on partitions using per-partition executors for async kernel dispatch.

---

## Example Configurations

SIMBI includes ~24 example configurations in `simbi_configs/examples/`:

| Example | Description |
|---------|-------------|
| `sod.py` | Newtonian shock tube |
| `marti_muller.py` | SRHD shock tube (1D, 3D variants) |
| `kh.py` | Kelvin-Helmholtz instability |
| `rt.py` | Rayleigh-Taylor instability (with gravity) |
| `sedov.py` | Sedov-Taylor explosion (spherical) |
| `thermal_bomb.py` | Thermal bomb (2D, 3D variants) |
| `magnetic_blast.py` | MHD blast wave |
| `magnetic_shock_tube.py` | 1D MHD shock |
| `orszag_tang.py` | SRMHD Orszag-Tang vortex |
| `kepler.py` | Keplerian disk with central mass |
| `uniform_sphere.py` | Uniform sphere with immersed body |
| `quad_shocktube.py` | 2D multi-region shock |

Run any example:
```bash
simbi run sedov --mode gpu --resolution 256
simbi run kepler --mode cpu --resolution 128 128
```

---

## Publications

SIMBI has been used in the following publications:

| Year | Publication |
|------|-------------|
| **2024** | [DuPont, M. et al. - "Strong Bow Shocks: Turbulence and An Exact Self-Similar Asymptotic"](https://iopscience.iop.org/article/10.3847/1538-4357/ad5adc) |
| **2023** | [DuPont, M. et al. - "Explosions in Roche-lobe Distorted Stars: Relativistic Bullets in Binaries"](https://iopscience.iop.org/article/10.3847/1538-4357/ad284e) |
| **2023** | [DuPont, M. & MacFadyen A. - "Stars Bisected By Relativistic Blades"](https://iopscience.iop.org/article/10.3847/2041-8213/ad132c) |
| **2022** | [DuPont, M. et al. - "Ellipsars: Ring-like Explosions from Flattened Stars"](https://iopscience.iop.org/article/10.3847/2041-8213/ac6ded) |

---

## Citation

```bibtex
@article{simbi2023,
  title={SIMBI: A high-performance 3D relativistic magneto-gas dynamic
         code for astrophysical fluid simulations},
  author={Eigen, J. and others},
  journal={Journal of Computational Physics},
  volume={456},
  pages={111-123},
  year={2023},
  publisher={Elsevier}
}
```

---

## Version History

| Version | Changes |
|---------|---------|
| **v0.8.0** | Minimized compiler warnings |
| **v0.7.0** | Added mypy type checking, immersed boundary method |
| **v0.6.0** | Fixed git tag ordering, code refactoring |
| **v0.5.0** | Performance optimizations |
| **v0.4.0** | Code restructuring |
| **v0.3.0** | Improved C++ organization |
| **v0.2.0** | Memory contiguity optimizations |
| **v0.1.0** | Initial release |

---

## Support

Report bugs and request features at [GitHub Issues](https://github.com/EigenDev/simbi/issues).

**Common issues:**

Installation problems:
```bash
gcc --version   # check ≥ 8
python --version  # check ≥ 3.10
nvidia-smi      # verify GPU (NVIDIA)
rocm-smi        # verify GPU (AMD)
```

Runtime issues:
```bash
simbi run <problem> --info  # check available options
simbi run --configs         # list available problems
```

---

## License

SIMBI is distributed under the [MIT License](https://opensource.org/licenses/MIT).

---

<div align="center">

**[Report Bug](https://github.com/EigenDev/simbi/issues) • [Request Feature](https://github.com/EigenDev/simbi/issues)**

</div>
