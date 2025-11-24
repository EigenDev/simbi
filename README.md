SIMBI

   ███████╗██╗███╗   ███╗██████╗ ██╗
   ██╔════╝██║████╗ ████║██╔══██╗██║
   ███████╗██║██╔████╔██║██████╔╝██║
   ╚════██║██║██║╚██╔╝██║██╔══██╗██║
   ███████║██║██║ ╚═╝ ██║██████╔╝██║
   ╚══════╝╚═╝╚═╝     ╚═╝╚═════╝ ╚═╝


<div align="center">

A 3D relativistic magneto-gas dynamic code for astrophysical fluid simulations

Quick Start • Installation • Usage • Publications

</div>

[!NOTE]
Research Usage: SIMBI has been used in research published in The Astrophysical Journal and The Astrophysical Journal Letters, studying relativistic jets, shock morphology, and stellar explosions.

Overview

SIMBI is designed for astrophysical fluid simulations. It handles magnetohydrodynamics across both Newtonian and relativistic regimes, from relativistic jets in active galactic nuclei to stellar explosions and magnetic turbulence.

Key Capabilities

Multi-Physics Regimes: Special Relativistic Magnetohydrodynamics (SRMHD), Special Relativistic Hydrodynamics (SRHD), and Newtonian Hydrodynamics.

Computing: Native GPU acceleration for NVIDIA (CUDA) and AMD (HIP/ROCm) platforms.

Numerics: Shock capturing with multiple reconstruction schemes and Riemann solvers.

Boundaries: Immersed boundary method, dynamic meshes, and customizable boundary conditions.

Configuration: Python-driven configuration system with automatic CLI generation and type safety.

Simulation Gallery

<div align="center">

Relativistic Jet Evolution

Relativistic Shock Tube

Rayleigh-Taylor Instability







Moving Mesh Techniques

Magnetic Turbulence





</div>

Quick Start

Get SIMBI running with the classic Marti & Müller relativistic shock tube problem.

<details>
<summary><strong>Standard Setup (CPU)</strong></summary>

# 1. Install SIMBI with virtual environment
CC=gcc CXX=g++ python dev.py install --create-venv yes

# 2. Activate environment
source .simbi-venv/bin/activate

# 3. Run the classic test problem
simbi run marti_muller --mode cpu --resolution 400

# 4. Visualize results
simbi plot data/1000.chkpt.000_400.h5 --setup "Marti & Muller Problem 1" --field rho v p


</details>

<details>
<summary><strong>GPU Acceleration Setup</strong></summary>

For NVIDIA GPUs (V100 example with compute capability 7.0):

# Install with GPU support
CC=gcc CXX=g++ python dev.py install --gpu-compilation --dev-arch 70

# Run on GPU with higher resolution
simbi run marti_muller --mode gpu --resolution 1024


</details>

[!TIP]
The Marti & Müller shock tube is a standard relativistic hydrodynamics test problem that demonstrates SIMBI's shock-capturing capabilities.

Installation

System Requirements

[!NOTE]
Minimum Requirements: gcc ≥ 8 or clang ≥ 10, Python 3.10+, 8 GB RAM, Linux/macOS

Recommended: Latest stable compiler, Python 3.11+, 32+ GB RAM for large 3D simulations, Ubuntu 20.04+

Dependencies

SIMBI requires several core libraries and build tools:

Build Systems: Meson, Ninja

Libraries: pybind11, HDF5 libraries

Python Packages: mypy, halo, pydantic, rich

<details>
<summary><strong>UV Package Manager (Recommended)</strong></summary>

For faster dependency management, we recommend UV:

# Install UV (Unix-like systems)
curl -sSf [https://install.astral.sh](https://install.astral.sh) | sh

# Or with pip
pip install uv


UV provides faster dependency resolution. When installed, SIMBI automatically detects and uses it.

</details>

Installation Options

Standard Installation

CC=gcc CXX=g++ python dev.py install --create-venv yes


With Visualization Tools

CC=gcc CXX=g++ python dev.py install --visual-extras --create-venv yes


<details>
<summary><strong>GPU Compilation</strong></summary>

NVIDIA GPUs (specify compute capability without decimal):

# Example: V100 (compute capability 7.0)
CC=gcc CXX=g++ python dev.py install --gpu-compilation --dev-arch 70

# Example: A100 (compute capability 8.0)
CC=gcc CXX=g++ python dev.py install --gpu-compilation --dev-arch 80


AMD GPUs

# Example: MI100 (gfx908)
CC=gcc CXX=g++ python dev.py install --gpu-compilation --gpu-platform hip --dev-arch gfx908


</details>

<details>
<summary><strong>Advanced Options</strong></summary>

# Full feature set
python dev.py install --create-venv yes --visual-extras --cli-extras --gpu-compilation --dev-arch 70

# Custom environment path
python dev.py install --create-venv yes --venv-path /custom/path

# View all options
python dev.py install --help


</details>

Environment Activation

# After installation, always activate before use
source <wherever_you_installed_simbi>/.venv/bin/activate


Usage

Running Simulations

SIMBI uses a Python configuration system with automatic CLI generation:

# Basic usage
simbi run marti_muller --mode gpu --resolution 400 --adiabatic-index 1.4

# Full path (for custom configs)
simbi run simbi_configs/examples/marti_muller.py --mode cpu --resolution 1024

# With UV (recommended for isolation)
uv run simbi run marti_muller --mode gpu --resolution 512


[!NOTE]
SIMBI automatically generates command-line options from your configuration fields. Use simbi run <problem> --info to see all available parameters.

Global Options

--mode → Execution mode (cpu/gpu)

--resolution → Grid resolution

--adiabatic-index → Ratio of specific heats

Analysis & Visualization

# Plot simulation results
simbi plot data/1000.chkpt.000_400.h5 --setup "Marti & Muller Problem 1" --field rho v p

# Using UV
uv run simbi plot data/checkpoint.h5 --setup "Physics Setup" --field rho v p


Creating Custom Simulations

# Generate configuration template
simbi generate --name my_simulation

# Edit the generated file: simbi_configs/my_simulation.py
# Run your simulation
simbi run my_simulation --mode gpu


<details>
<summary><strong>UV Workflow Benefits</strong></summary>

Using UV with SIMBI provides:

Faster dependency resolution

Environment isolation

Reproducible builds

Conda compatibility

Shell Aliases (optional convenience):

# Add to .bashrc/.zshrc
alias simbi-run="uv run simbi run"
alias simbi-plot="uv run simbi plot"


</details>

Physics & Configuration

Physics Regimes

[!IMPORTANT]
SRMHD - Special Relativistic Magnetohydrodynamics
Applications: AGN jets, pulsar wind nebulae, magnetic reconnection

SRHD - Special Relativistic Hydrodynamics
Applications: Gamma-ray bursts, relativistic shocks, stellar explosions

Classical - Newtonian Hydrodynamics
Applications: Stellar winds, ISM dynamics, classical turbulence

Configuration System

SIMBI uses a type-safe, field-decorated configuration approach:

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


<details>
<summary><strong>Advanced Physics Features</strong></summary>

Dynamic Meshes

@computed_field
@property
def scale_factor(self) -> Callable[float, float]:
    return lambda time: 1.0 + 0.1 * time  # Linear expansion


Source Terms

@computed_field
@property
def gravity_source_expressions(self):
    # Custom gravity implementation using expression graphs
    graph = simbi.Expr.Graph()
    x_comp = simbi.Expr.constant(0.0, graph)
    y_comp = simbi.Expr.constant(-0.1, graph)  # Gravity in -y direction
    terms = graph.compile([x_comp, y_comp])
    return terms.serialize()


Immersed Boundaries

@computed_field
@property
def body_system(self) -> BodySystemConfig:
    # Define solid objects in computational domain
    pass


</details>

Numerical Methods

[!NOTE]
Riemann Solvers
• HLLE - HLL solver with entropy fix
• HLLC - HLL Contact solver for hydrodynamics
• HLLD - HLL Discontinuities solver (magnetohydrodynamics only)

Coordinate Systems
• Cartesian • Spherical • Cylindrical • Axis-cylindrical • Planar-cylindrical

Grid Spacing
• Linear - Uniform grid spacing • Logarithmic - Logarithmic spacing

Boundary Conditions
PERIODIC • REFLECTING • OUTFLOW • DYNAMIC

Publications

SIMBI has been utilized in the following studies:

Year

Publication

Topic

2024

DuPont, M. et al. - "Strong Bow Shocks: Turbulence and An Exact Self-Similar Asymptotic"

Shock wave physics

2023

DuPont, M. et al. - "Explosions in Roche-lobe Distorted Stars: Relativistic Bullets in Binaries"

Binary stellar system dynamics

2023

DuPont, M. & MacFadyen A. - "Stars Bisected By Relativistic Blades"

High-energy astrophysics

2022

DuPont, M. et al. - "Ellipsars: Ring-like Explosions from Flattened Stars"

Stellar explosion mechanisms

Citation

If SIMBI contributes to your research, please cite:

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


Development

<details>
<summary><strong>Version History</strong></summary>

Version

Focus

Key Changes

v0.8.0

Code quality

Minimized compiler warnings

v0.7.0

Features

Added mypy type checking, immersed boundary method

v0.6.0

Stability

Fixed Git tag ordering, code refactoring

v0.5.0

Performance

Code optimizations and improvements

v0.4.0

Architecture

Major code restructuring

v0.3.0

Readability

Improved C++ code organization

v0.2.0

Performance

Memory contiguity optimizations

v0.1.0

Genesis

Initial release with core features

</details>

Roadmap

Short Term

[ ] Enhanced immersed boundary methods

[ ] Additional reconstruction schemes

[ ] Improved visualization tools

Medium Term

[ ] Multi-GPU support

[ ] Extended equation of state options

[ ] Cloud computing integration

Long Term

[ ] MPI support for distributed computing

[ ] General relativistic extensions

[ ] Machine learning integration

Support & Community

Getting Help

Issues: GitHub Issues for bugs and feature requests

<details>
<summary><strong>Common Issues & Solutions</strong></summary>

Installation Problems

# Check compiler compatibility
gcc --version  # Should be ≥ 8
python --version  # Should be ≥ 3.10

# Verify GPU setup (if using)
nvidia-smi  # For NVIDIA
rocm-smi    # For AMD


Runtime Issues

# Environment activation (don't forget!)
source .simbi-venv/bin/activate

# Check GPU detection
simbi run <problem> --info  # Shows available options

# Memory issues for large simulations
ulimit -m unlimited


</details>

License

SIMBI is distributed under the MIT License.

<div align="center">

Built for computational astrophysics research

Report Bug • Request Feature • Contribute

</div>
