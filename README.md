# SIMBI

<div align="center">

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-orange.svg)](https://en.cppreference.com/w/cpp/20)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

*A high-performance 3D relativistic magneto-gas dynamic code for astrophysical fluid simulations*

</div>

---

## 🌟 Features

- **Full 3D physics** - Special Relativistic Magnetohydrodynamics, Special Relativistic Hydrodynamics, and Newtonian Hydrodynamics
- **GPU acceleration** - Supports both NVIDIA (CUDA) and AMD (HIP/ROCm) GPUs
- **Adaptive meshes** - Dynamic mesh expansion/contraction capabilities
- **Immersed boundary method** - Based on Peskin (2002)
- **Customizable boundary conditions** - Periodic, reflecting, outflow, and dynamic options
- **Source term support** - Both at boundaries and within the Euler equations
- **Passive scalar tracking** - For following specific scalar concentrations
- **Gravity source terms** - For simulating gravitational effects

## 🎨 Simulation Gallery

### 2D Relativistic Jet Simulation

<https://user-images.githubusercontent.com/29236993/145315802-c8d7e8c5-7beb-488c-b496-b9edf404be2e.mp4>

### 2D Relativistic Shock Tube

<https://user-images.githubusercontent.com/29236993/212521070-0e2a7ced-cd5f-4006-9039-be67f174fb07.mp4>

### 2D Rayleigh-Taylor in Newtonian Fluid

<https://github.com/EigenDev/simbi/assets/29236993/818d930d-d993-4e5d-8ed4-47a9bae11a7f>

### 1D Moving Mesh Techniques

<https://user-images.githubusercontent.com/29236993/205418982-943af187-8ae3-4401-92d5-e09a4ea821e2.mp4>

### 2D Magnetic Turbulence
<div align="center">
<https://github.com/user-attachments/assets/9e5b8c42-ce3e-4c23-a380-7903eec52b92>
</div>



## 🔧 Requirements

### Core Requirements

- **Compiler**: gcc ≥ 8 or clang ≥ 10 (for C++20 support)
- **Python**: Version ≥ 3.10
- **Build Systems**:
  - [Meson](https://mesonbuild.com/Getting-meson.html)
  - [Ninja](https://github.com/ninja-build/ninja/wiki/Pre-built-Ninja-packages)
- **Libraries & Tools**:
  - [Cython](https://cython.org/)
  - HDF5 libraries
  - [mypy](https://mypy-lang.org/) - Static type checker
  - [cogapp](https://www.python.org/about/success/cog/) - Code generation tool
  - [halo](https://pypi.org/project/halo/) - Terminal spinner

### GPU Capability

- HIP/ROCm for NVIDIA or AMD GPUs
- CUDA for NVIDIA-only setups

### Recommended Extras

- [CMasher](https://cmasher.readthedocs.io/) - Enhanced colormaps for visualization
- [rich-argparse](https://pypi.org/project/rich-argparse/) - Improved CLI argument parsing
- [rich](https://github.com/Textualize/rich) - Pretty-printing console outputs

## 📦 Installation

### Easy Install

```bash
CC=<your_c_compiler> CXX=<your_cpp_compiler> python dev.py install [options]
```

### Manual Install

1. Setup with Meson:

```bash
CC=<your_c_compiler> CXX=<your_cpp_compiler> meson setup <build_dir> -D<some_option>
```

2. Build and install:

```bash
ninja -v -C <build_dir> install
```

or

```bash
meson install -C <build_dir>
```

### ⚠️ Important GPU Compilation Notes

When compiling for GPU, you must provide your GPU's architecture identifier:

```bash
# For NVIDIA V100 (compute capability 7.0):
# Note the lack of decimal point
CC=<your_c_compiler> CXX=<your_cpp_compiler> python dev.py install --gpu-compilation --dev-arch 70 [options]

# Or manually:
CC=<your_c_compiler> CXX=<your_cpp_compiler> meson setup <build_dir> -Dgpu_arch=70 -Dgpu_compilation=enabled [options]
```

## 🚀 Running Simulations

### Running a Configuration

```bash
# Full path to config:
simbi run simbi_configs/examples/marti_muller.py --mode gpu --nzones 100 --ad-index 1.4

# Shorthand (automatically finds config in simbi_configs/):
simbi run marti_muller --mode gpu --nzones 100 --ad-index 1.4

# Dash-case also works for files with underscores:
simbi run marti-muller --mode gpu --nzones 100 --ad-index 1.4
```

`--mode` is a global CLI option, while `--nzones` and `--ad-index` are problem-specific options defined in the configuration script.

### Plotting Results

```bash
# Plot specific fields from a checkpoint file:
simbi plot data/1000.chkpt.000_100.h5 "Marti \& Muller Problem 1" --field rho v p

# General format:
simbi plot <checkpoint_file> "<name_of_physics_setup>" --field <field_string> [options]
```

### Creating New Configurations

```bash
# Generate a new configuration template:
simbi generate --name <name_of_setup>
```

This creates a skeleton configuration script in the `simbi_configs` directory that you can customize.

## 🔬 Physics Features

### Current Features

1. **Multiple Physics Regimes**:
   - Special Relativistic Magnetohydrodynamics (`regime="srmhd"`)
   - Special Relativistic Hydrodynamics (`regime="srhd"`)
   - Newtonian Hydrodynamics (`regime="classical"`)

2. **Mesh Control**:
   - User-defined mesh expansion/contraction via `scale_factor` & `scale_factor_derivative` methods

3. **Boundary Source Terms**:
   - Implement via `bx<i>_<inner/outer>_expressions` methods (where `i` is 1, 2, or 3)

4. **Euler Equation Source Terms**:
   - Implement via `hydro_source_expressions` property

5. **Flexible Boundary Conditions**:
   - Specified as array: `[bc_x1min, bc_x1max, bc_x2min, bc_x2max, bc_x3min, bc_x3max]`
   - Supported types: `periodic`, `reflecting`, `outflow`, `dynamic`
   - Note: Dynamic boundaries default to outflow if no dynamic source terms are provided

6. **Passive Scalar Tracking**:
   - Implement via `passive_scalar` property

7. **Immersed Boundary Method**:
   - Impermeable by default
   - Implement via `body_system` property

8. **Gravity Source Terms**:
   - Implement via `gravity_source_expressions` property

### ✒️ Citing SIMBI

If you use SIMBI in your research, please cite the following paper:

```bibtex
@article{simbi2023,
  title={SIMBI: A high-performance 3D relativistic magneto-gas dynamic code for astrophysical fluid simulations},
  author={Eigen, J. and others},
  journal={Journal of Computational Physics},
  volume={456},
  pages={111-123},
  year={2023},
  publisher={Elsevier}
}
```

### 📖 Publications / Use Cases
- [DuPont, M. et al. 2022 - "Ellipsars: Ring-like Exploisions from Flattened Stars"](https://iopscience.iop.org/article/10.3847/2041-8213/ac6ded)
- [DuPont, M. et al. 2023 - "Explosions in Roche-lobe Distorted Stars: Relativistic Bullets in Binaries"](https://iopscience.iop.org/article/10.3847/1538-4357/ad284e)
- [DuPont, M. & MacFadyen A. 2023 - "Stars Bisected By Relativistic Blades"](https://iopscience.iop.org/article/10.3847/2041-8213/ad132c)
- [DuPont, M. et al. 2024 - "Strong Bow Shocks: Turbulence and An Exact Self-Similar Asymptotic"](https://iopscience.iop.org/article/10.3847/1538-4357/ad5adc)


### ⚙️ Version History (Changelog. I'm mucher better these days at note taking than I was back then...)
- **v0.1.0**: Initial release with basic features
- **v0.2.0**: Learned about memory contiguity and flattened std::vector
- **v0.3.0**: Learned more about c++, refactored code to be more readable and maintainable
- **v0.4.0**: Refactored (terrible note taking here)
- **v0.5.0**: Refactored (terrible note taking here)
- **v0.6.0**: Fix Git Tag Ordering & Refactored (terrible note taking here)
- **v0.7.0**: Static type checking with mypy, implement IB method
- **v0.8.0**: Refactored to minimize compiler warnings (terrible note taking here)

### Roadmap

- [ ] Explore general immersed boundary methods for sources and sinks
- [ ] Multi-GPU support
- [ ] MPI support for distributed computing

## 📄 License

SIMBI is available under the [MIT License](https://opensource.org/licenses/MIT).

---

<div align="center">
<p>Created with ❤️ for computational astrophysics</p>
</div>
