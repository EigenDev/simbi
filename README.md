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

**A high-performance 3D relativistic magneto-gas dynamic code for astrophysical fluid simulations**

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-orange.svg?style=for-the-badge&logo=c%2B%2B)](https://en.cppreference.com/w/cpp/20)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/downloads/)
[![CUDA Support](https://img.shields.io/badge/CUDA-Supported-76B900.svg?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![AMD Support](https://img.shields.io/badge/AMD-Supported-ED1C24.svg?style=for-the-badge&logo=amd)](https://rocm.docs.amd.com/)

[Quick Start](#quick-start) • [Installation](#installation) • [Physics Capabilities](#physics-capabilities) • [Publications](#publications)

</div>

---

## Overview

SIMBI enables state-of-the-art astrophysical fluid simulations with cutting-edge numerics and physics. The code handles Special Relativistic Magnetohydrodynamics (SRMHD), Special Relativistic Hydrodynamics (SRHD), and Newtonian Hydrodynamics across both CPU and GPU architectures.

**Key Features:**
- ⚛️ Full 3D physics with high-resolution shock capturing methods
- 🚀 GPU acceleration supporting both NVIDIA (CUDA) and AMD (HIP/ROCm) platforms
- 🐍 Python-driven configuration system for complex simulations
- 🌊 Dynamic mesh capabilities with adaptive expansion/contraction
- 🔬 Immersed boundary method based on Peskin (2002)
- 📊 Passive scalar tracking and customizable source terms

**Research Applications:**
SIMBI has powered interesting research in relativistic jets, stellar explosions, and magnetized plasma dynamics, with results published in leading astrophysics journals including ApJ and ApJL.

---

## Simulation Gallery

<div align="center">

| Relativistic Jet Evolution | Relativistic Shock Tube | Rayleigh-Taylor Instability |
|:---:|:---:|:---:|
| [![Jet](https://img.shields.io/badge/View-Animation-ff0000?style=flat-square&logo=youtube)](https://user-images.githubusercontent.com/29236993/145315802-c8d7e8c5-7beb-488c-b496-b9edf404be2e.mp4) | [![Shock](https://img.shields.io/badge/View-Animation-ff0000?style=flat-square&logo=youtube)](https://user-images.githubusercontent.com/29236993/212521070-0e2a7ced-cd5f-4006-9039-be67f174fb07.mp4) | [![RT](https://img.shields.io/badge/View-Animation-ff0000?style=flat-square&logo=youtube)](https://github.com/EigenDev/simbi/assets/29236993/818d930d-d993-4e5d-8ed4-47a9bae11a7f) |

| Moving Mesh Techniques | Magnetic Turbulence |
|:---:|:---:|
| [![Mesh](https://img.shields.io/badge/View-Animation-ff0000?style=flat-square&logo=youtube)](https://user-images.githubusercontent.com/29236993/205418982-943af187-8ae3-4401-92d5-e09a4ea821e2.mp4) | [![Turbulence](https://img.shields.io/badge/View-Animation-ff0000?style=flat-square&logo=youtube)](https://github.com/user-attachments/assets/9e5b8c42-ce3e-4c23-a380-7903eec52b92) |

</div>

---

## 🚀 Quick Start

Get SIMBI running with the Marti & Müller relativistic shock tube problem:

```bash
# 1. Install SIMBI with virtual environment
CC=gcc CXX=g++ python dev.py install --create-venv yes

# 2. Activate environment
source .simbi-venv/bin/activate

# 3. Run the classic test problem
simbi run marti_muller --mode cpu --resolution 400

# 4. Visualize results
simbi plot data/1000.chkpt.000_400.h5 --setup "Marti & Muller Problem 1" --field rho v p
```

**GPU Acceleration** (NVIDIA V100 example):
```bash
# Install with GPU support (compute capability 7.0)
CC=gcc CXX=g++ python dev.py install --gpu-compilation --dev-arch 70

# Run on GPU
simbi run marti_muller --mode gpu --resolution 1024
```

## 📦 Installation

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Compiler** | gcc ≥ 8, clang ≥ 10 | Latest stable with C++20 support |
| **Python** | 3.10+ | 3.11+ |
| **Memory** | 8 GB | 32+ GB for large 3D simulations |
| **GPU** (optional) | CUDA 11.0+, ROCm 4.0+ | Latest drivers |

### Essential Dependencies

SIMBI requires several core libraries and build tools:

- **Build Systems**: Meson, Ninja
- **Libraries**: pybind11, HDF5 libraries
- **Python Packages**: mypy, halo, pydantic, rich

### Recommended: UV Package Manager

For the best experience with SIMBI, we strongly recommend installing UV first. UV provides faster dependency resolution and more reliable package management:

```bash
# Install UV (Unix-like systems)
curl -sSf https://install.astral.sh | sh

# Or with pip
pip install uv
```

UV significantly improves dependency resolution and installation speed. When UV is installed, SIMBI will automatically detect and use it for dependency management.

### Installation Options

**Basic CPU Installation:**
```bash
CC=gcc CXX=g++ python dev.py install
```

**With Virtual Environment (Recommended):**
```bash
CC=gcc CXX=g++ python dev.py install --create-venv yes
```

**Including Visualization Tools:**
```bash
CC=gcc CXX=g++ python dev.py install --visual-extras
```

**GPU Compilation:**

For NVIDIA GPUs, you must specify your GPU's compute capability:
```bash
# Example for V100 (compute capability 7.0 - note no decimal point)
CC=gcc CXX=g++ python dev.py install --gpu-compilation --dev-arch 70
```

For AMD GPUs:
```bash
# Example for MI100 (gfx908)
CC=gcc CXX=g++ python dev.py install --gpu-compilation --gpu-platform hip --dev-arch gfx908
```

**Complete Installation with All Options:**
```bash
CC=gcc CXX=g++ python dev.py install --create-venv yes --visual-extras --cli-extras --gpu-compilation --dev-arch 70
```

### Virtual Environment Management

SIMBI can create and manage its own virtual environment:

```bash
# Always create a virtual environment
python dev.py install --create-venv yes

# Specify custom environment path
python dev.py install --create-venv yes --venv-path /custom/path

# Skip virtual environment creation
python dev.py install --create-venv no
```

After installation with a virtual environment, activate it before using SIMBI:
```bash
# Linux/macOS
source .simbi-venv/bin/activate

# Windows
.simbi-venv\Scripts\activate
```

### Build Configuration Options

```bash
# Debug build with symbols
python dev.py install --debug

# Optimized release build
python dev.py install --release

# View all available options
python dev.py install --help
```

---

## 💻 Usage

### Running Simulations

SIMBI uses a Python-driven configuration system. You can run predefined examples or create custom simulations:

```bash
# Run with full path
simbi run simbi_configs/examples/marti_muller.py --mode gpu --resolution 400 --adiabatic-index 1.4

# Shorthand notation (searches simbi_configs/ directory)
simbi run marti_muller --mode gpu --resolution 400 --adiabatic-index 1.4

# Dash-case also works
simbi run marti-muller --mode gpu --resolution 400 --adiabatic-index 1.4

# Using UV for environment isolation (recommended)
uv run simbi run marti_muller --mode gpu --resolution 400
```

**Global CLI Options:**
- `--mode`: Execution mode (cpu/gpu)
- `--resolution`: Grid resolution
- `--adiabatic-index`: Ratio of specific heats

Note: Problem-specific options are dynamically parsed based on the configuration script. Use `simbi run <problem> --info` to see all available options.

### Plotting and Analysis

```bash
# Plot specific fields
simbi plot data/1000.chkpt.000_400.h5 --setup "Marti & Muller Problem 1" --field rho v p

# General plotting format
simbi plot <checkpoint_file> --setup "<physics_setup_name>" --field <field_names>

# Using UV (optional)
uv run simbi plot data/1000.chkpt.000_400.h5 --setup "Custom Setup" --field rho v p
```

### Creating Custom Simulations

```bash
# Generate configuration template
simbi generate --name custom_simulation

# Using UV
uv run simbi generate --name custom_simulation
```

This creates a skeleton configuration file in the `simbi_configs/` directory that you can customize for your specific physics setup.

### Why Use UV?

Using UV with SIMBI (`uv run simbi ...`) provides several advantages:

- **Faster dependency resolution**: UV resolves and installs dependencies much faster than pip
- **Reliable isolation**: Ensures your SIMBI environment doesn't conflict with other Python projects
- **Reproducible environments**: Easier to recreate the exact same environment across systems
- **Compatible with conda**: Works within conda environments if you prefer conda

### Shell Aliases

To reduce typing with UV, consider creating aliases:

```bash
# Add to your .bashrc, .zshrc, etc.
alias simbi-run="uv run simbi run"
alias simbi-plot="uv run simbi plot"
alias simbi-generate="uv run simbi generate"
```

---

## Physics Capabilities

### Available Regimes

SIMBI supports three main physics regimes:

| Regime | Description | Applications |
|--------|-------------|--------------|
| **SRMHD** | Special Relativistic Magnetohydrodynamics | AGN jets, pulsar wind nebulae |
| **SRHD** | Special Relativistic Hydrodynamics | Gamma-ray bursts, relativistic shocks |
| **Classical** | Newtonian Hydrodynamics | Stellar winds, ISM dynamics |

### Configuration Example

SIMBI uses a modern, type-safe configuration system with automatic CLI generation:

```python
from pathlib import Path
import numpy as np
from simbi.core.config.base_config import SimbiBaseConfig
from simbi.core.config.fields import SimbiField
from simbi.core.types.input import CoordSystem, Regime, Solver, BoundaryCondition

class KelvinHelmholtz(SimbiBaseConfig):
    """Kelvin Helmholtz instability in Newtonian fluid"""

    # Grid configuration
    resolution: tuple[int, int] = SimbiField(
        (256, 256), description="Number of zones in x and y dimensions"
    )
    bounds: list[tuple[float, float]] = SimbiField(
        [(-0.5, 0.5), (-0.5, 0.5)], description="Domain boundaries"
    )

    # Physics setup
    regime: Regime = SimbiField(Regime.CLASSICAL, description="Physics regime")
    adiabatic_index: float = SimbiField(5.0/3.0, description="Adiabatic index")
    solver: Solver = SimbiField(Solver.HLLC, description="Numerical solver")

    # Physical parameters
    rhoL: float = SimbiField(2.0, description="Density in central layer")
    rhoR: float = SimbiField(1.0, description="Density in outer regions")

    # Simulation control
    end_time: float = SimbiField(20.0, description="End time for simulation")
    data_directory: Path = SimbiField(Path("data/kh_config"), description="Output directory")

    def initial_primitive_state(self):
        """Generate initial conditions with random perturbations"""
        def gas_state():
            # Implementation yields (rho, vx, vy, p) for each cell
            pass
        return gas_state
```

### Advanced Features

**Dynamic Meshes:**
```python
def scale_factor(self, time):
    return 1.0 + 0.1 * time  # Linear expansion

def scale_factor_derivative(self, time):
    return 0.1  # Rate of expansion
```

**Source Terms:**
```python
@property
def gravity_source_expressions(self):
    return ["0", "-G*M/r^2", "0"]  # Gravity in y-direction

@property
def hydro_source_expressions(self):
    return ["0", "0", "0", "0", "cooling_function(T)"]  # Custom source
```

**Boundary Sources:**
```python
def bx1_inner_expressions(self):
    # Custom boundary source at x1 minimum
    pass
```

**Immersed Boundary Method:**
```python
@property
def body_system(self):
    # Define solid objects in the computational domain
    # Implementation based on Peskin (2002)
    pass
```

**Passive Scalar Tracking:**
```python
def initial_conditions(self, x, y, z):
    # Standard: (density, velocity, pressure)
    # With scalar: (density, velocity, pressure, concentration)
    return (rho, v, p, scalar)
```

### Boundary Conditions

Available boundary condition types:
- `BoundaryCondition.PERIODIC`
- `BoundaryCondition.REFLECTING`
- `BoundaryCondition.OUTFLOW`
- `BoundaryCondition.DYNAMIC`

### Numerical Methods

SIMBI provides robust numerical schemes for relativistic fluid dynamics:

**Riemann Solvers:**
- `Solver.HLLE` - HLL solver with entropy fix
- `Solver.HLLC` - HLL Contact solver for hydrodynamics
- `Solver.HLLD` - HLL Discontinuities solver (magnetohydrodynamics only)

**Coordinate Systems:**
- `CoordSystem.CARTESIAN` - Cartesian coordinates
- `CoordSystem.SPHERICAL` - Spherical coordinates
- `CoordSystem.CYLINDRICAL` - Cylindrical coordinates

**Grid Spacing:**
- `CellSpacing.LINEAR` - Uniform grid spacing
- `CellSpacing.LOGARITHMIC` - Logarithmic spacing

The code employs high-resolution shock capturing with multiple reconstruction schemes, constrained transport for magnetic field evolution, and adaptive time stepping for numerical stability.

---

## 📚 Publications

SIMBI has enabled breakthrough research published in leading astrophysics journals:

| Year | Publication | Focus |
|------|-------------|-------|
| 2024 | [DuPont, M. et al. - "Strong Bow Shocks: Turbulence and An Exact Self-Similar Asymptotic"](https://iopscience.iop.org/article/10.3847/1538-4357/ad5adc) | Shock wave physics |
| 2023 | [DuPont, M. et al. - "Explosions in Roche-lobe Distorted Stars: Relativistic Bullets in Binaries"](https://iopscience.iop.org/article/10.3847/1538-4357/ad284e) | Binary stellar systems |
| 2023 | [DuPont, M. & MacFadyen A. - "Stars Bisected By Relativistic Blades"](https://iopscience.iop.org/article/10.3847/2041-8213/ad132c) | High-energy astrophysics |
| 2022 | [DuPont, M. et al. - "Ellipsars: Ring-like Explosions from Flattened Stars"](https://iopscience.iop.org/article/10.3847/2041-8213/ac6ded) | Stellar explosion dynamics |

## 📖 Citation

If you use SIMBI in your research, please cite:

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

## 🛠️ Development

### Version History

| Version | Focus | Key Changes |
|---------|-------|-------------|
| v0.8.0 | Code quality | Refactored to minimize compiler warnings |
| v0.7.0 | Features | Added static type checking with mypy, implemented immersed boundary method |
| v0.6.0 | Stability | Fixed Git tag ordering & code refactoring |
| v0.5.0 | Performance | Code refactoring and improvements |
| v0.4.0 | Architecture | Code refactoring and improvements |
| v0.3.0 | Readability | Improved C++ code readability and maintainability |
| v0.2.0 | Performance | Optimized memory contiguity with flattened std::vector |
| v0.1.0 | Initial release | Basic features |

### Roadmap

**Short Term:**
- Enhanced immersed boundary methods
- Additional reconstruction schemes
- Improved visualization tools

**Medium Term:**
- Multi-GPU support
- Extended equation of state options

**Long Term:**
- MPI support for distributed computing
- General relativistic extensions

---

## 🆘 Support

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/EigenDev/simbi/issues) for bug reports and feature requests
- **Discussions**: [GitHub Discussions](https://github.com/EigenDev/simbi/discussions) for community Q&A

### Common Issues

**Installation Problems:**
- Ensure your compiler supports C++20
- Verify Python version is 3.10+
- Check that all dependencies are installed

**GPU Detection Issues:**
- Verify GPU drivers are up to date
- Ensure CUDA/ROCm versions are compatible
- Check GPU compute capability matches `--dev-arch` specification

**Virtual Environment Issues:**
- Remember to activate your environment: `source .simbi-venv/bin/activate`
- If using UV, you can run commands with `uv run simbi ...`

---

## 📜 License

SIMBI is distributed under the [MIT License](https://opensource.org/licenses/MIT).

---

<div align="center">

**Built for computational astrophysics research**

[Report Bug](https://github.com/EigenDev/simbi/issues) • [Request Feature](https://github.com/EigenDev/simbi/issues) • [Contribute](https://github.com/EigenDev/simbi/contribute)

</div>
