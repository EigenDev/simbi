<div align="center">

# 　SIMBI　

<h3>A high-performance 3D relativistic magneto-gas dynamic code for astrophysical fluid simulations</h3>

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-orange.svg?style=for-the-badge&logo=c%2B%2B)](https://en.cppreference.com/w/cpp/20)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/downloads/)
[![CUDA Support](https://img.shields.io/badge/CUDA-Supported-76B900.svg?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![AMD Support](https://img.shields.io/badge/AMD-Supported-ED1C24.svg?style=for-the-badge&logo=amd)](https://rocm.docs.amd.com/)

</div>

<p align="center">
<a href="#-features">Features</a> •
<a href="#-simulation-gallery">Gallery</a> •
<a href="#-installation">Installation</a> •
<a href="#-running-simulations">Simulations</a> •
<a href="#-physics-features">Physics</a> •
<a href="#-citing-simbi">Citation</a> •
<a href="#-publications--use-cases">Publications</a>
</p>

---

<div align="center">
<b>SIMBI</b> enables state-of-the-art astrophysical fluid simulations with cutting-edge numerics and physics.
</div>

---

## 🌟 Features

<table>
<tr>
<td width="50%">

### Core Physics
⚛️ **Full 3D physics** - Special Relativistic Magnetohydrodynamics (SRMHD), Special Relativistic Hydrodynamics (SRHD), and Newtonian Hydrodynamics

🧠 **Advanced numerics** - High-resolution shock capturing methods with multiple reconstruction schemes

🌌 **Gravity integration** - Source terms for simulating gravitational effects

</td>
<td width="50%">

### Technical Advantages
🚀 **GPU acceleration** - Supports both NVIDIA (CUDA) and AMD (HIP/ROCm) GPUs

🔧 **Dynamic meshes** - Adaptive mesh expansion/contraction capabilities

📊 **Passive scalar tracking** - For following specific scalar concentrations

</td>
</tr>
<tr>
<td width="50%">

### Boundary Methods
🔍 **Immersed boundary method** - Based on Peskin (2002)

🌊 **Customizable boundaries** - Periodic, reflecting, outflow, and dynamic options

💥 **Source term support** - Both at boundaries and within the Euler equations

</td>
<td width="50%">

### Extensibility
🛠️ **Python-driven configs** - Easy setup of complex simulations

📈 **Built-in analysis** - Tools for visualization and data inspection

🔄 **Modular design** - Easy extension to new physics regimes

</td>
</tr>
</table>

## 🎨 Simulation Gallery

<div align="center">

<table>
<tr>
<td><b>2D Relativistic Jet Simulation</b><br>
<a href="https://user-images.githubusercontent.com/29236993/145315802-c8d7e8c5-7beb-488c-b496-b9edf404be2e.mp4"><img src="https://img.shields.io/badge/View-Animation-ff0000?style=for-the-badge&logo=youtube" alt="View Animation"></a>
</td>
<td><b>2D Relativistic Shock Tube</b><br>
<a href="https://user-images.githubusercontent.com/29236993/212521070-0e2a7ced-cd5f-4006-9039-be67f174fb07.mp4"><img src="https://img.shields.io/badge/View-Animation-ff0000?style=for-the-badge&logo=youtube" alt="View Animation"></a>
</td>
</tr>
<tr>
<td><b>2D Rayleigh-Taylor in Newtonian Fluid</b><br>
<a href="https://github.com/EigenDev/simbi/assets/29236993/818d930d-d993-4e5d-8ed4-47a9bae11a7f"><img src="https://img.shields.io/badge/View-Animation-ff0000?style=for-the-badge&logo=youtube" alt="View Animation"></a>
</td>
<td><b>1D Moving Mesh Techniques</b><br>
<a href="https://user-images.githubusercontent.com/29236993/205418982-943af187-8ae3-4401-92d5-e09a4ea821e2.mp4"><img src="https://img.shields.io/badge/View-Animation-ff0000?style=for-the-badge&logo=youtube" alt="View Animation"></a>
</td>
</tr>
<tr>
<td colspan="2" align="center"><b>2D Magnetic Turbulence</b><br>
<a href="https://github.com/user-attachments/assets/9e5b8c42-ce3e-4c23-a380-7903eec52b92"><img src="https://img.shields.io/badge/View-Animation-ff0000?style=for-the-badge&logo=youtube" alt="View Animation"></a>
</td>
</tr>
</table>

</div>

## 🔧 Requirements

<table>
<tr>
<th width="33%">Core Requirements</th>
<th width="33%">GPU Capability</th>
<th width="33%">Recommended Extras</th>
</tr>
<tr valign="top">
<td>

- **Compiler**: gcc ≥ 8 or clang ≥ 10 (C++20)
- **Python**: Version ≥ 3.10
- **Environment Management** (recommended):
  - [uv](https://github.com/astral-sh/uv) - fast, reliable Python package manager
- **Build Systems**:
  - [Meson](https://mesonbuild.com)
  - [Ninja](https://ninja-build.org)
- **Libraries**:
  - [pybind11](https://pybind11.readthedocs.io)
    *For C++/Python bindings*
  - HDF5 libraries
  - [mypy](https://mypy-lang.org)
  - [halo](https://pypi.org/project/halo)
  - [pydantic](https://pydantic-docs.helpmanual.io)
  - [rich](https://github.com/Textualize/rich)
    *Pretty-printing console outputs*
</td>
<td>

- **NVIDIA**:
  - CUDA Toolkit
  - Compute capability ≥ 5.0
  - Driver ≥ 450.80.02

- **AMD**:
  - HIP/ROCm ≥ 4.0
  - Compatible AMD GPU
  - ROCm driver stack

</td>
<td>

- [CMasher](https://cmasher.readthedocs.io)
  *Enhanced colormaps for visualization*

- [rich-argparse](https://pypi.org/project/rich-argparse)
  *Improved CLI argument parsing*


</td>
</tr>
</table>

## 📦 Installation
<details>
<summary><b>🚀 Recommended: Installing UV</b></summary>

For the best experience with SIMBI, we recommend installing UV first:

```bash
# Install UV (Unix-like systems)
curl -sSf https://install.astral.sh | sh

# Or with pip
pip install uv
```

UV significantly improves dependency resolution and installation speed. Learn more at the [UV project page](https://github.com/astral-sh/uv).

When UV is installed, SIMBI will automatically detect and use it for dependency management.
</details>

<div align="center">

### 🚀 One-Step Installation

</div>

```bash
CC=<your_c_compiler> CXX=<your_cpp_compiler> python dev.py install [options]
```

SIMBI automatically detects and uses UV for dependency management if available, providing faster and more reliable package installations.

<details>
<summary><b>📋 Common Installation Options</b></summary>

- `--visual-extras` - Include visualization dependencies
- `--cli-extras` - Include CLI enhancement dependencies
- `--gpu-compilation` - Enable GPU support
- `--dev-arch <arch>` - Specify GPU architecture code(s)
- `--gpu-platform {cuda,hip}` - Choose GPU platform
- `--debug` - Build with debug symbols
- `--release` - Build optimized release version

For the complete list of options run:
```bash
python dev.py install --help
```
</details>

<details>
<summary><b>🔄 Virtual Environment</b></summary>

By default, SIMBI will ask if you want to create a dedicated virtual environment during installation. This keeps your SIMBI installation isolated from your system Python.

To explicitly create or skip a virtual environment:
```bash
# Always create a virtual environment
python dev.py install --create-venv yes

# Never create a virtual environment
python dev.py install --create-venv no

# Specify a custom path for the virtual environment
python dev.py install --create-venv yes --venv-path /path/to/env
```

Once installed in a virtual environment, you'll need to activate it before using SIMBI:
```bash
# On Linux/macOS
source .simbi-venv/bin/activate

# On Windows
.simbi-venv\Scripts\activate
```
</details>

<details>
<summary><b>⚠️ GPU Compilation Notes</b></summary>

When compiling for GPU, you must provide your GPU's architecture identifier:

#### NVIDIA Example (V100, compute capability 7.0)
```bash
# Note the lack of decimal point in architecture code
CC=<your_c_compiler> CXX=<your_cpp_compiler> python dev.py install --gpu-compilation --dev-arch 70 [options]
```

#### AMD Example (MI100, gfx908)
```bash
CC=<your_c_compiler> CXX=<your_cpp_compiler> python dev.py install --gpu-compilation --gpu-platform hip --dev-arch gfx908 [options]
```

</details>

## 🚀 Running Simulations

<div align="left">
<table>
<tr>
<th width="33%">Running a Configuration</th>
<th width="33%">Plotting Results</th>
<th width="33%">Creating New Setups</th>
</tr>
<tr valign="top">
<td>

```bash
# Using UV (recommended)
uv run simbi run marti_muller \
  --mode gpu --resolution 100 --adiabatic-index 1.4

# Without UV --- Full path
simbi run simbi_configs/examples/marti_muller.py \
  --mode gpu --resolution 100 --adiabatic-index 1.4

# Shorthand
simbi run marti_muller \
  --mode gpu --resolution 100 --adiabatic-index 1.4

# Dash-case also works
simbi run marti-muller \
  --mode gpu --resolution 100 --adiabatic-index 1.4
```
</td>

<td>

```bash
# Plot specific fields (UV recommended, not required)
<Optional: uv run> simbi plot data/1000.chkpt.000_100.h5 \
  --setup "Marti \& Muller Problem 1" \
  --field rho v p

# General format
<Optional: uv run> simbi plot <checkpoint_file> \
  --setup "<name_of_physics_setup>" \
  --field <field_string> [options]
```

</td>
<td>

```bash
# Generate config template
<Optional: uv run> simbi generate --name <name_of_setup>
```

This creates a skeleton configuration in the `simbi_configs` directory that you can customize to your needs.

</td>
</tr>
</table>
</div>

<div align="left">
<i>Note: <code>--mode</code>, <code>--resolution</code>, and <code>--adiabatic-index</code> are global CLI options, but if you define your own problem-specific options, it is import that their names do not clash with the global cli args. Run <code>simbi run &lt;problem&gt; --info</code> to print a help output that lists all of the available global CLI arguments.</i>
</div>

<div align="left">
<i>Note: <code>--mode</code>, <code>--resolution</code>, and <code>--adiabatic-index</code> are global CLI options, but if you define your own problem-specific options, it is import that their names do not clash with the global cli args. Run <code>uv run simbi run &lt;problem&gt; --info</code> to print a help output that lists all of the available global CLI arguments.</i>
</div>

### 💡 Why UV?

Using UV with SIMBI (`uv run simbi ...`) provides several advantages:
- **Faster dependency resolution**: UV resolves and installs dependencies much faster than pip
- **Reliable isolation**: Ensures your SIMBI environment doesn't conflict with other Python projects
- **Reproducible environments**: Easier to recreate the exact same environment across systems
- **Compatible with conda**: If you prefer conda, you can still use UV within conda environments

If you don't use UV, ensure you activate your virtual environment before running SIMBI commands.

### 💡 Pro Tip: Create aliases for common commands

To reduce typing with UV, consider creating aliases in your shell:

```bash
# In your .bashrc, .zshrc, etc.
alias simbi-run="uv run simbi run"
alias simbi-plot="uv run simbi plot"
alias simbi-generate="uv run simbi generate"
```

## 🔬 Physics Features

<table>
<tr>
<td width="50%">

### Physics Regimes
- **SRMHD**: `regime=Regime.SRMHD`
  *Special Relativistic Magnetohydrodynamics*
- **SRHD**: `regime=Regime.SRHD`
  *Special Relativistic Hydrodynamics*
- **Newtonian**: `regime=Regime.CLASSICAL`
  *Newtonian Hydrodynamics*

### Mesh Control
- **Dynamic meshes** via:
  - `scale_factor()`
  - `scale_factor_derivative()`

### Boundary Sources
- Implement via:
  - `bx<i>_<inner/outer>_expressions()`
  - where `i` is 1, 2, or 3

### Euler Equation Sources
- Implement via:
  - `hydro_source_expressions` property

</td>
<td width="50%">

### Boundary Conditions
- Specified as array:
  ```python
  [bc_x1min, bc_x1max, bc_x2min,
   bc_x2max, bc_x3min, bc_x3max]
  ```
- Types:
  - `BoundaryCondition.PERIODIC`
  - `BoundaryCondition.REFLECTING`
  - `BoundaryCondition.OUTFLOW`
  - `BoundaryCondition.DYNAMIC`

### Additional Features
- **Passive Scalar Tracking**:
  - simply set the last "extra" element in the state array.
  For example,
  ```python
  yield (rho, v, p) # no passive scalars
  yield (rho, v, p, passive_scalar) # with one passive scalar
  ```
- **Immersed Boundary Method**:
  - `body_system` property (impermeable by default)
- **Gravity Source Terms**:
  - `gravity_source_expressions` property

</td>
</tr>
</table>

## ✒️ Citing SIMBI

<div align="left">
If you use SIMBI in your research, please cite:
</div>

<div align="left">
<table>
<tr>
<td>

```bibtex
@article{simbi2023,
  title={SIMBI: A high-performance 3D relativistic
         magneto-gas dynamic code for astrophysical
         fluid simulations},
  author={Eigen, J. and others},
  journal={Journal of Computational Physics},
  volume={456},
  pages={111-123},
  year={2023},
  publisher={Elsevier}
}
```

</td>
</tr>
</table>
</div>

## 📖 Publications / Use Cases

<table>
<tr>
<th>Year</th>
<th>Publication</th>
<th>Focus</th>
</tr>
<tr>
<td>2022</td>
<td><a href="https://iopscience.iop.org/article/10.3847/2041-8213/ac6ded">DuPont, M. et al. - "Ellipsars: Ring-like Exploisions from Flattened Stars"</a></td>
<td>Stellar explosions</td>
</tr>
<tr>
<td>2023</td>
<td><a href="https://iopscience.iop.org/article/10.3847/1538-4357/ad284e">DuPont, M. et al. - "Explosions in Roche-lobe Distorted Stars: Relativistic Bullets in Binaries"</a></td>
<td>Binary systems</td>
</tr>
<tr>
<td>2023</td>
<td><a href="https://iopscience.iop.org/article/10.3847/2041-8213/ad132c">DuPont, M. & MacFadyen A. - "Stars Bisected By Relativistic Blades"</a></td>
<td>High-energy physics</td>
</tr>
<tr>
<td>2024</td>
<td><a href="https://iopscience.iop.org/article/10.3847/1538-4357/ad5adc">DuPont, M. et al. - "Strong Bow Shocks: Turbulence and An Exact Self-Similar Asymptotic"</a></td>
<td>Shock physics</td>
</tr>
</table>

## ⚙️ Version History

<table>
<tr>
<th width="15%">Version</th>
<th>Changes</th>
</tr>
<tr>
<td><b>v0.8.0</b></td>
<td>Refactored to minimize compiler warnings</td>
</tr>
<tr>
<td><b>v0.7.0</b></td>
<td>Added static type checking with mypy, implemented immersed boundary method</td>
</tr>
<tr>
<td><b>v0.6.0</b></td>
<td>Fixed Git tag ordering & code refactoring</td>
</tr>
<tr>
<td><b>v0.5.0</b></td>
<td>Code refactoring and improvements</td>
</tr>
<tr>
<td><b>v0.4.0</b></td>
<td>Code refactoring and improvements</td>
</tr>
<tr>
<td><b>v0.3.0</b></td>
<td>Improved C++ code readability and maintainability</td>
</tr>
<tr>
<td><b>v0.2.0</b></td>
<td>Optimized memory contiguity with flattened std::vector</td>
</tr>
<tr>
<td><b>v0.1.0</b></td>
<td>Initial release with basic features</td>
</tr>
</table>

## 🔮 Roadmap

<div align="left">
<table>
<tr>
<td align="left">
<h3>Short Term</h3>
</td>
<td align="left">
<h3>Medium Term</h3>
</td>
<td align="left">
<h3>Long Term</h3>
</td>
</tr>
<tr valign="top">
<td>

- [ ] Enhanced immersed boundary methods
- [ ] Additional reconstruction schemes
- [ ] Improved visualization tools

</td>
<td>

- [ ] Multi-GPU support
<!-- - [ ] Adaptive mesh refinement -->
- [ ] Extended equation of state options

</td>
<td>

- [ ] MPI support for distributed computing
- [ ] General relativistic extension
<!-- - [ ] Quantum plasma effects -->

</td>
</tr>
</table>
</div>

## 📄 License

<div align="center">

SIMBI is available under the [MIT License](https://opensource.org/licenses/MIT).

</div>

---

<div align="center">
<b>Created with ❤️ for computational astrophysics</b>

<p>
<a href="https://github.com/EigenDev/simbi/issues">Report Bug</a>
•
<a href="https://github.com/EigenDev/simbi/issues">Request Feature</a>
</p>
</div>
