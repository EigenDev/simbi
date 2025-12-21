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
- Python configuration system with automatic CLI generation

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
# install with virtual environment
CC=gcc CXX=g++ python dev.py install --create-venv yes

# activate environment
source .simbi-venv/bin/activate

# run test problem
simbi run marti_muller --mode cpu --resolution 400

# visualize
simbi plot data/1000.chkpt.000_400.h5 --setup "Marti & Muller Problem 1" --field rho v p
```

**GPU (NVIDIA V100, compute capability 7.0):**
```bash
CC=gcc CXX=g++ python dev.py install --gpu-compilation --dev-arch 70
simbi run marti_muller --mode gpu --resolution 1024
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
- Ubuntu 20.04+

### Dependencies

- **Build**: Meson, Ninja
- **Libraries**: pybind11, HDF5
- **Python**: mypy, halo, pydantic, rich

**UV Package Manager (optional):**
```bash
curl -sSf https://install.astral.sh | sh
# or
pip install uv
```

UV provides faster dependency resolution. SIMBI automatically uses it when available.

### Installation Commands

**Standard:**
```bash
CC=gcc CXX=g++ python dev.py install --create-venv yes
```

**With visualization tools:**
```bash
CC=gcc CXX=g++ python dev.py install --visual-extras --create-venv yes
```

**GPU compilation (NVIDIA):**
```bash
# V100 (compute capability 7.0)
CC=gcc CXX=g++ python dev.py install --gpu-compilation --dev-arch 70

# A100 (compute capability 8.0)
CC=gcc CXX=g++ python dev.py install --gpu-compilation --dev-arch 80
```

**GPU compilation (AMD):**
```bash
# MI100 (gfx908)
CC=gcc CXX=g++ python dev.py install --gpu-compilation --gpu-platform hip --dev-arch gfx908
```

**Environment activation:**
```bash
source <installation_path>/.venv/bin/activate
```

---

## Usage

### Running Simulations

```bash
# basic usage
simbi run marti_muller --mode gpu --resolution 400 --adiabatic-index 1.4

# custom config path
simbi run simbi_configs/examples/marti_muller.py --mode cpu --resolution 1024

# with uv
uv run simbi run marti_muller --mode gpu --resolution 512

# view available parameters
simbi run <problem> --info
```

**Common options:**
- `--mode` → cpu/gpu execution
- `--resolution` → grid resolution
- `--adiabatic-index` → ratio of specific heats

### Visualization

```bash
simbi plot data/1000.chkpt.000_400.h5 --setup "Marti & Muller Problem 1" --field rho v p
uv run simbi plot data/checkpoint.h5 --setup "Physics Setup" --field rho v p
```

### Creating Custom Simulations

```bash
simbi generate --name my_simulation
# edit simbi_configs/my_simulation.py
simbi run my_simulation --mode gpu
```

---

## Physics

### Regimes

**SRMHD** - Special Relativistic Magnetohydrodynamics
- AGN jets, pulsar wind nebulae, magnetic reconnection

**SRHD** - Special Relativistic Hydrodynamics
- Gamma-ray bursts, relativistic shocks, stellar explosions

**Classical** - Newtonian Hydrodynamics
- Stellar winds, ISM dynamics, classical turbulence

### Configuration

SIMBI uses type-safe configuration with automatic CLI generation:

```python
from pathlib import Path
from simbi.core.config.base_config import SimbiBaseConfig
from simbi.core.config.fields import SimbiField
from simbi.core.types.input import Regime, Solver

class KelvinHelmholtz(SimbiBaseConfig):
    """kelvin-helmholtz instability in newtonian fluid"""

    resolution: tuple[int, int] = SimbiField(
        (256, 256), description="number of zones in x and y"
    )
    bounds: list[tuple[float, float]] = SimbiField(
        [(-0.5, 0.5), (-0.5, 0.5)], description="domain boundaries"
    )
    regime: Regime = SimbiField(Regime.CLASSICAL, description="physics regime")
    solver: Solver = SimbiField(Solver.HLLC, description="riemann solver")
    adiabatic_index: float = SimbiField(5.0/3.0, description="ratio of specific heats")
    rhoL: float = SimbiField(2.0, description="density in central layer")
    rhoR: float = SimbiField(1.0, description="density in outer regions")
    end_time: float = SimbiField(20.0, description="end time")
    data_directory: Path = SimbiField(Path("data/kh_config"), description="output directory")

    def initial_primitive_state(self):
        """generate initial conditions"""
        def gas_state():
            # yields (rho, vx, vy, p) for each grid cell
            pass
        return gas_state
```

**Dynamic meshes:**
```python
@computed_field
@property
def scale_factor(self) -> Callable[float, float]:
    return lambda time: 1.0 + 0.1 * time
```

**Source terms:**
```python
@computed_field
@property
def gravity_source_expressions(self):
    graph = simbi.Expr.Graph()
    x_comp = simbi.Expr.constant(0.0, graph)
    y_comp = simbi.Expr.constant(-0.1, graph)
    terms = graph.compile([x_comp, y_comp])
    return terms.serialize()
```

**Immersed boundaries:**
```python
@computed_field
@property
def body_system(self) -> BodySystemConfig:
    # define solid objects in computational domain
    # based on peskin (2002) immersed boundary method
    pass
```

### Numerical Methods

**Riemann Solvers:**
- `HLLE` - HLL solver with entropy fix
- `HLLC` - HLL Contact solver for hydrodynamics
- `HLLD` - HLL Discontinuities solver (magnetohydrodynamics)

**Coordinate Systems:**
- Cartesian, Spherical, Cylindrical, Axis-cylindrical, Planar-cylindrical

**Grid Spacing:**
- Linear (uniform)
- Logarithmic

**Boundary Conditions:**
- PERIODIC, REFLECTING, OUTFLOW, DYNAMIC

**Time Integration:**
- Forward Euler
- RK2 (Berger-Colella for AMR)

**Adaptive Mesh Refinement:**
- Berger-Colella subcycling
- Flux correction at coarse-fine boundaries
- Restriction and prolongation operators
- Three subcycling modes: NONE, STANDARD, MANUAL, ADAPTIVE

---

## Architecture

### Entity-Component-System Design

SIMBI uses an ECS architecture for partition-aware multi-device execution:

**Core Components:**
- `simulation_t<Rank, Regime, CoordSystem>` - Top-level simulation state
- `partition_t<Rank>` - Device assignment + execution stream for one domain partition
- `level_decomposition_t<Rank>` - All partitions + halo graph for one AMR level
- `partition_fields_t` - Hydro fields (cons, prim, flux, bfield, efield) for one partition
- `flux_register_component_t` - AMR flux correction registers

**Systems:**
- `timestep_system_t` - Compute CFL-limited timesteps with subcycling
- `ghost_fill_system_t` - Fill ghost cells via physical BCs or coarse grid prolongation
- `c2p_system_t` - Conservative to primitive variable conversion
- `flux_system_t` - Compute numerical fluxes via Riemann solver
- `euler_system_t` / `rk2_stage1_system_t` / `rk2_stage2_system_t` - Time integration
- `restriction_system_t` - Inject fine grid interior to coarse grid
- `prolongation_system_t` - Interpolate coarse grid to fine grid boundaries
- `reflux_system_t` - Apply flux correction at AMR boundaries
- `body_effects_system_t` - Immersed boundary forces and diagnostics

Each system operates on partitions, using per-partition executors for async kernel dispatch.

**Geometry Handling:**
Geometry operations use `with_block_geometry<CoordSystem>()` to dispatch specialized implementations per coordinate system at compile time.

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
gcc --version  # check ≥ 8
python --version  # check ≥ 3.10
nvidia-smi  # verify GPU (NVIDIA)
rocm-smi    # verify GPU (AMD)
```

Runtime issues:
```bash
source .simbi-venv/bin/activate  # activate environment
simbi run <problem> --info  # check available options
```

---

## License

SIMBI is distributed under the [MIT License](https://opensource.org/licenses/MIT).

---

<div align="center">

**[Report Bug](https://github.com/EigenDev/simbi/issues) • [Request Feature](https://github.com/EigenDev/simbi/issues)**

</div>
