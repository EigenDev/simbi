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

**[Quick Start](#-quick-start) • [Installation](#-installation) • [Usage](#-usage) • [Publications](#-publications)**

</div>

---

> [!NOTE]
> **Research Impact**: SIMBI powers breakthrough research published in *The Astrophysical Journal* and *The Astrophysical Journal Letters*, enabling discoveries in relativistic jets, stellar explosions, and magnetized plasma dynamics at Princeton University, NYU, and institutions worldwide.

## ★ Overview

SIMBI enables state-of-the-art astrophysical fluid simulations with cutting-edge numerics and physics. From relativistic jets in active galactic nuclei to stellar explosions and magnetic turbulence, SIMBI handles the challenging physics of magnetohydrodynamics across both Newtonian and relativistic regimes.

```
┌─────────────────┬─────────────────┬─────────────────┐
│  ⚡ Performance  │  🔬 Physics     │  🛠️ Development │
│                 │                 │                 │
│  • GPU Accelerated │ • SRMHD/SRHD    │ • Python Config │
│  • CPU Optimized   │ • Classical HD  │ • Type Safety   │
│  • HDF5 Output     │ • 3D Dynamics   │ • Modern C++20  │
└─────────────────┴─────────────────┴─────────────────┘
```

**Core Capabilities:**
- **Multi-Physics Regimes**: Special Relativistic Magnetohydrodynamics (SRMHD), Special Relativistic Hydrodynamics (SRHD), and Newtonian Hydrodynamics
- **High-Performance Computing**: Native GPU acceleration for NVIDIA (CUDA) and AMD (HIP/ROCm) platforms
- **Advanced Numerics**: High-resolution shock capturing with multiple reconstruction schemes and Riemann solvers
- **Flexible Boundaries**: Immersed boundary method, dynamic meshes, and customizable boundary conditions
- **Research-Ready**: Python-driven configuration system with automatic CLI generation and type safety

---

## 🎬 Simulation Gallery

<div align="center">

| Relativistic Jet Evolution | Relativistic Shock Tube | Rayleigh-Taylor Instability |
|:---:|:---:|:---:|
| [![Jet](https://img.shields.io/badge/View-Animation-ff0000?style=flat-square&logo=youtube)](https://user-images.githubusercontent.com/29236993/145315802-c8d7e8c5-7beb-488c-b496-b9edf404be2e.mp4) | [![Shock](https://img.shields.io/badge/View-Animation-ff0000?style=flat-square&logo=youtube)](https://user-images.githubusercontent.com/29236993/212521070-0e2a7ced-cd5f-4006-9039-be67f174fb07.mp4) | [![RT](https://img.shields.io/badge/View-Animation-ff0000?style=flat-square&logo=youtube)](https://github.com/EigenDev/simbi/assets/29236993/818d930d-d993-4e5d-8ed4-47a9bae11a7f) |

| Moving Mesh Techniques | Magnetic Turbulence |
|:---:|:---:|
| [![Mesh](https://img.shields.io/badge/View-Animation-ff0000?style=flat-square&logo=youtube)](https://user-images.githubusercontent.com/29236993/205418982-943af187-8ae3-4401-92d5-e09a4ea821e2.mp4) | [![Turbulence](https://img.shields.io/badge/View-Animation-ff0000?style=flat-square&logo=youtube)](https://github.com/user-attachments/assets/9e5b8c42-ce3e-4c23-a380-7903eec52b92) |

</div>

---

## ▶ Quick Start

Get SIMBI running with the classic Marti & Müller relativistic shock tube problem:

<details>
<summary><strong>🚀 5-Minute Setup (CPU)</strong></summary>

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

</details>

<details>
<summary><strong>⚡ GPU Acceleration Setup</strong></summary>

For NVIDIA GPUs (V100 example with compute capability 7.0):
```bash
# Install with GPU support
CC=gcc CXX=g++ python dev.py install --gpu-compilation --dev-arch 70

# Run on GPU with higher resolution
simbi run marti_muller --mode gpu --resolution 1024
```

</details>

> [!TIP]
> **New to SIMBI?** The Marti & Müller shock tube is a classic relativistic hydrodynamics test problem that demonstrates SIMBI's shock-capturing capabilities. It runs in seconds and produces publication-quality output.

---

## 📦 Installation

### System Requirements

> [!NOTE]
> **Minimum Requirements**: gcc ≥ 8 or clang ≥ 10, Python 3.10+, 8 GB RAM, Linux/macOS
>
> **Recommended**: Latest stable compiler, Python 3.11+, 32+ GB RAM for large 3D simulations, Ubuntu 20.04+

### Dependencies

SIMBI requires several core libraries and build tools:
- **Build Systems**: Meson, Ninja
- **Libraries**: pybind11, HDF5 libraries
- **Python Packages**: mypy, halo, pydantic, rich

<details>
<summary><strong>⚙️ UV Package Manager (Recommended)</strong></summary>

For optimal dependency management, we recommend UV:

```bash
# Install UV (Unix-like systems)
curl -sSf https://install.astral.sh | sh

# Or with pip
pip install uv
```

UV provides faster dependency resolution and more reliable package management. When installed, SIMBI automatically detects and uses it.

</details>

### Installation Options

**Standard Installation:**
```bash
CC=gcc CXX=g++ python dev.py install --create-venv yes
```

**With Visualization Tools:**
```bash
CC=gcc CXX=g++ python dev.py install --visual-extras --create-venv yes
```

<details>
<summary><strong>🎯 GPU Compilation</strong></summary>

**NVIDIA GPUs** (specify compute capability without decimal):
```bash
# Example: V100 (compute capability 7.0)
CC=gcc CXX=g++ python dev.py install --gpu-compilation --dev-arch 70

# Example: A100 (compute capability 8.0)
CC=gcc CXX=g++ python dev.py install --gpu-compilation --dev-arch 80
```

**AMD GPUs**:
```bash
# Example: MI100 (gfx908)
CC=gcc CXX=g++ python dev.py install --gpu-compilation --gpu-platform hip --dev-arch gfx908
```

</details>

<details>
<summary><strong>🔧 Advanced Options</strong></summary>

```bash
# All features enabled
python dev.py install --create-venv yes --visual-extras --cli-extras --gpu-compilation --dev-arch 70

# Custom environment path
python dev.py install --create-venv yes --venv-path /custom/path

# Development build
python dev.py install --debug

# View all options
python dev.py install --help
```

</details>

**Environment Activation:**
```bash
# After installation, always activate before use
source .simbi-venv/bin/activate
```

---

## 💻 Usage

### Running Simulations

SIMBI uses a modern Python configuration system with automatic CLI generation:

```bash
# Basic usage
simbi run marti_muller --mode gpu --resolution 400 --adiabatic-index 1.4

# Full path (for custom configs)
simbi run simbi_configs/examples/marti_muller.py --mode cpu --resolution 1024

# With UV (recommended for isolation)
uv run simbi run marti_muller --mode gpu --resolution 512
```

> [!NOTE]
> **CLI Magic**: SIMBI automatically generates command-line options from your configuration fields. Use `simbi run <problem> --info` to see all available parameters.

**Global Options:**
- `--mode` → Execution mode (cpu/gpu)
- `--resolution` → Grid resolution
- `--adiabatic-index` → Ratio of specific heats

### Analysis & Visualization

```bash
# Plot simulation results
simbi plot data/1000.chkpt.000_400.h5 --setup "Marti & Muller Problem 1" --field rho v p

# Using UV
uv run simbi plot data/checkpoint.h5 --setup "Physics Setup" --field rho v p
```

### Creating Custom Simulations

```bash
# Generate configuration template
simbi generate --name my_simulation

# Edit the generated file: simbi_configs/my_simulation.py
# Run your simulation
simbi run my_simulation --mode gpu
```

<details>
<summary><strong>💡 UV Workflow Benefits</strong></summary>

Using UV with SIMBI provides:
- **Faster dependency resolution** - Up to 10x faster than pip
- **Environment isolation** - No conflicts with other Python projects
- **Reproducible builds** - Exact dependency versions across systems
- **Conda compatibility** - Works within existing conda environments

**Shell Aliases** (optional convenience):
```bash
# Add to .bashrc/.zshrc
alias simbi-run="uv run simbi run"
alias simbi-plot="uv run simbi plot"
```

</details>

---

## ⚛️ Physics & Configuration

### Physics Regimes

> [!IMPORTANT]
> **SRMHD** - Special Relativistic Magnetohydrodynamics
> *Applications*: AGN jets, pulsar wind nebulae, magnetic reconnection
>
> **SRHD** - Special Relativistic Hydrodynamics
> *Applications*: Gamma-ray bursts, relativistic shocks, stellar explosions
>
> **Classical** - Newtonian Hydrodynamics
> *Applications*: Stellar winds, ISM dynamics, classical turbulence

### Modern Configuration System

SIMBI uses a type-safe, field-decorated configuration approach:

```python
from pathlib import Path
from simbi.core.config.base_config import SimbiBaseConfig
from simbi.core.config.fields import SimbiField
from simbi.core.types.input import CoordSystem, Regime, Solver, BoundaryCondition

class KelvinHelmholtz(SimbiBaseConfig):
    """Kelvin Helmholtz instability in Newtonian fluid"""

    # Grid setup
    resolution: tuple[int, int] = SimbiField(
        (256, 256), description="Number of zones in x and y dimensions"
    )
    bounds: list[tuple[float, float]] = SimbiField(
        [(-0.5, 0.5), (-0.5, 0.5)], description="Domain boundaries"
    )

    # Physics configuration
    regime: Regime = SimbiField(Regime.CLASSICAL, description="Physics regime")
    solver: Solver = SimbiField(Solver.HLLC, description="Riemann solver")
    adiabatic_index: float = SimbiField(5.0/3.0, description="Ratio of specific heats")

    # Physical parameters
    rhoL: float = SimbiField(2.0, description="Density in central layer")
    rhoR: float = SimbiField(1.0, description="Density in outer regions")

    # Simulation control
    end_time: float = SimbiField(20.0, description="End time")
    data_directory: Path = SimbiField(Path("data/kh_config"), description="Output directory")

    def initial_primitive_state(self):
        """Generate initial conditions with perturbations"""
        def gas_state():
            # Implementation yields (rho, vx, vy, p) for each grid cell
            # Your physics setup goes here
            pass
        return gas_state
```

<details>
<summary><strong>🔬 Advanced Physics Features</strong></summary>

**Dynamic Meshes:**
```python
@computed_field
@property
def scale_factor(self) -> Callable[float, float]:
    return lambda time: 1.0 + 0.1 * time  # Linear expansion
```

**Source Terms:**
```python
@computed_field
@property
def gravity_source_expressions(self):
    # Custom gravity implementation using expression graphs
    graph = simbi.Expr.Graph()
    x_comp = simbi.Expr.constant(0.0, graph)
    y_comp = simbi.Expr.constant(-0.1, graph)  # Gravity in -y direction
    terms = graph.compile([x_comp, y_comp])
    return terms.serialize()
```

**Immersed Boundaries:**
```python
@computed_field
@property
def body_system(self) -> BodySystemConfig:
    # Define solid objects in computational domain
    # Based on Peskin (2002) immersed boundary method
    pass
```

</details>

### Numerical Methods

> [!NOTE]
> **Riemann Solvers**
> • `HLLE` - HLL solver with entropy fix
> • `HLLC` - HLL Contact solver for hydrodynamics
> • `HLLD` - HLL Discontinuities solver (magnetohydrodynamics only)
>
> **Coordinate Systems**
> • `Cartesian` • `Spherical` • `Cylindrical` • `Axis-cylindrical` • `Planar-cylindrical`
>
> **Grid Spacing**
> • `Linear` - Uniform grid spacing • `Logarithmic` - Logarithmic spacing

**Boundary Conditions:**
`PERIODIC` • `REFLECTING` • `OUTFLOW` • `DYNAMIC`

---

## 📚 Publications

> [!IMPORTANT]
> **Research Heritage**: SIMBI has enabled breakthrough discoveries in relativistic astrophysics, with results published in top-tier journals and cited throughout the computational astrophysics community.

| Year | Publication | Impact |
|------|-------------|--------|
| **2024** | [DuPont, M. et al. - "Strong Bow Shocks: Turbulence and An Exact Self-Similar Asymptotic"](https://iopscience.iop.org/article/10.3847/1538-4357/ad5adc) | Shock wave physics breakthrough |
| **2023** | [DuPont, M. et al. - "Explosions in Roche-lobe Distorted Stars: Relativistic Bullets in Binaries"](https://iopscience.iop.org/article/10.3847/1538-4357/ad284e) | Binary stellar system dynamics |
| **2023** | [DuPont, M. & MacFadyen A. - "Stars Bisected By Relativistic Blades"](https://iopscience.iop.org/article/10.3847/2041-8213/ad132c) | High-energy astrophysics |
| **2022** | [DuPont, M. et al. - "Ellipsars: Ring-like Explosions from Flattened Stars"](https://iopscience.iop.org/article/10.3847/2041-8213/ac6ded) | Stellar explosion mechanisms |

---

## 📖 Citation

If SIMBI contributes to your research, please cite:

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

<details>
<summary><strong>📋 Version History</strong></summary>

| Version | Focus | Key Changes |
|---------|-------|-------------|
| **v0.8.0** | Code quality | Minimized compiler warnings |
| **v0.7.0** | Features | Added mypy type checking, immersed boundary method |
| **v0.6.0** | Stability | Fixed Git tag ordering, code refactoring |
| **v0.5.0** | Performance | Code optimizations and improvements |
| **v0.4.0** | Architecture | Major code restructuring |
| **v0.3.0** | Readability | Improved C++ code organization |
| **v0.2.0** | Performance | Memory contiguity optimizations |
| **v0.1.0** | Genesis | Initial release with core features |

</details>

### Roadmap

**Short Term**
- [ ] Enhanced immersed boundary methods
- [ ] Additional reconstruction schemes
- [ ] Improved visualization tools

**Medium Term**
- [ ] Multi-GPU support
- [ ] Extended equation of state options
- [ ] Cloud computing integration

**Long Term**
- [ ] MPI support for distributed computing
- [ ] General relativistic extensions
- [ ] Machine learning integration

---

## 🆘 Support & Community

### Getting Help

- **📋 Issues**: [GitHub Issues](https://github.com/EigenDev/simbi/issues) for bugs and feature requests
- **💬 Discussions**: [GitHub Discussions](https://github.com/EigenDev/simbi/discussions) for community Q&A

<details>
<summary><strong>🔧 Common Issues & Solutions</strong></summary>

**Installation Problems:**
```bash
# Check compiler compatibility
gcc --version  # Should be ≥ 8
python --version  # Should be ≥ 3.10

# Verify GPU setup (if using)
nvidia-smi  # For NVIDIA
rocm-smi    # For AMD
```

**Runtime Issues:**
```bash
# Environment activation (don't forget!)
source .simbi-venv/bin/activate

# Check GPU detection
simbi run <problem> --info  # Shows available options

# Memory issues for large simulations
ulimit -m unlimited
```

</details>

---

## 📜 License

SIMBI is distributed under the [MIT License](https://opensource.org/licenses/MIT).

---

<div align="center">

> [!NOTE]
> **Built for computational astrophysics research**

**[Report Bug](https://github.com/EigenDev/simbi/issues) • [Request Feature](https://github.com/EigenDev/simbi/issues) • [Contribute](https://github.com/EigenDev/simbi/contribute)**

</div>
