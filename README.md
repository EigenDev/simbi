# SIMBI: 3D Relativistic Gas Dynamics Code

# 2D Relativistic Jet Simulation

<https://user-images.githubusercontent.com/29236993/145315802-c8d7e8c5-7beb-488c-b496-b9edf404be2e.mp4>

# 2D Relativistic Shock Tube

<https://user-images.githubusercontent.com/29236993/212521070-0e2a7ced-cd5f-4006-9039-be67f174fb07.mp4>

# 2D Rayleigh-Taylor in Newtonian Fluid

https://github.com/EigenDev/simbi/assets/29236993/818d930d-d993-4e5d-8ed4-47a9bae11a7f


# 1D Moving mesh techniques

<https://user-images.githubusercontent.com/29236993/205418982-943af187-8ae3-4401-92d5-e09a4ea821e2.mp4>



# Requirements

1)  gcc >= gcc5 or clang >= clang5 (for c++17 support at the very least)
2)  [Cython](https://cython.org/)
3)  [meson](https://mesonbuild.com/Getting-meson.html), 
    `pip install meson` will usually suffice
4)  A build system like `make` or
    [ninja](https://github.com/ninja-build/ninja/wiki/Pre-built-Ninja-packages)
5)  HDF5 libraries
6)  [mypy](https://mypy-lang.org/), a static type checker
7)  Python >= 3.10

## For GPU capability

8)  HIP/ROCm if wanting to run on NVIDIA or AMD GPUs, or just CUDA if
    running purely NVIDIA
## Extras
a) [CMasher](https://cmasher.readthedocs.io/) for richer set of colormaps, blends into matplotlib

b) [rich-argparse](https://pypi.org/project/rich-argparse/) for pretty-print argparse

c) [rich](https://github.com/Textualize/rich) for pretty-printing console outputs
# Quick setup guide
## Installing
<strong>Easy Install</strong>
1) Run 
    ```bash
    $ CC=<your_c_compiler> CXX=<your_cpp_compiler> python dev.py install [options]
    ```

<strong>Manual Install</strong>
1)  Run

    ``` bash
    $ CC=<your_c_compiler> CXX=<your_cpp_compiler> meson setup <build_dir> -D<some_option>
    ```

    from project root. It is important that this directory is not named
    `build` because the `install` call with create `build` dir for the `pip`
    installation part. For the `-D<build_option>` part, check the
    `meson_options.txt` file for available build options.

2)  Run
    ``` bashbool
    $ ninja -v -C <build_dir> install
    ```

    or

    ``` bash
    $ meson install -C <build_dir>
    ```

3)  If `meson` detected `hip` or `cuda`, the install script will install
    both the cpu and gpu extensions into the `simbi/libs` directory.
## Important (!)
When compiling on a GPU, you must provide your GPU's respective architecture identifier.
That is to say, if I am compiling on an NVIDIA V100 device with compute capability 7.0, I would
build with:
```bash
# note the lack of a decimal
$ CC=<your_c_compiler> CXX=<your_cpp_compiler> python dev.py install --gpu-compilation --dev-arch 70 [options]
# or if manually installing
$ CC=<your_c_compiler> CXX=<your_cpp_compiler> meson setup <build_dir> -Dgpu_arch=70 -Dgpu_compilation=enabled [options]
```
Also, the meson.build script assumes c++20 by default. If your compiler
does not support this, one must do a clean build - i.e., delete the original build directory generated by meson(!) -  and run
```bash
$ CC=<your_c_compiler> CXX=<your_cpp_compiler> python dev.py install --cpp17
# or if manually installing
$ CC=<your_c_compiler> CXX=<your_cpp_compiler> meson setup <build_dir> -Dcpp_std=c++17 [options]
```

## Running
<strong>Running a Configuration</strong>

4)  If all is well, we can test. To test, try running the configuration
    scripts provided. For example:

    ``` bash
    $ simbi run simbi_configs/examples/marti_muller.py --mode gpu --nzones 100 --ad-gamma 1.4 
    # or one could do 
    $ simbi run marti_muller --mode gpu --nzones 100 --ad-gamma 1.4
    # or 
    $ simbi run marti-muller --mode gpu --nzones 100 --ad-gamma 1.4
    # since the entry point is built to recursively search the simbi_configs/ folder for valid .py scripts
    # and dash-cased searches for file matches with underscores
    ```

    where `--mode` is a global command line option available for every
    config script, and `--nzones` and `--ad-gamma` are problem-specific options
    that are dynamically parsed based on whatever `DynamicArg` variables
    exist in the config script you create. Check out how to create one of
    these configuration scripts in the `simbi_configs/examples/` folder! When creating
    your own configuration file, you must place it in a directory entitled `simbi_configs/` and run `simbi` from your `simbi_configs/` parent directory and `simbi` should auto detect your configuration and run the simulation.

    You can plot the above output by running 
    ``` bash
    $ simbi plot data/1000.chkpt.000_100.h5 "Marti \& Muller Problem 1" --field rho v p --tex
    ```

    The usual formula for plotting a checkpoint file is like so:
    ``` bash
    $ simbi plot <checkpoint_file> "<name_of_physics_setup>" --field <field_string> [options]
    ```
    One can also do `simbi clone --name <name_of_setup>`, and a new skeleton configuration script will appear in the `simbi_configs` directory named `<name_of_setup>` that you can build off of. 
5)  ???
6)  Profit

## Physics Features (so far)
1) Special Relativistic and Newtonian Hydro up to 3D (set the `regime` property to `classical` or `relativistic`)
2) Supports user-defined mesh expansion / contraction (`scale_factor` & `scale_factor_derivative` methods)
3) Supports user-defined density, momentum, and energy density terms outside of grid (Implementing the `dens_outer`, `mom_outer`, AND `edens_outer` methods sets this)
4) Supports source terms in the Euler equations (implementing the `sources` property sets this)
5) Support source terms at the boundaries (implementing the `boundary_sources` property sets this)
6) Boundary conditions given as array of strings like so `[bc_x1min, bc_x1max, bc_x2min, bc_x2max, bc_x3min, bc_x3max]` where the supported boundary conditions are `periodic, reflecting, outflow, inflow`. If an inflow boundary condition is set, but no inflow boundary source terms are given, the code will switch to outflow boundary conditions to prevent crashes. 
7) Can track a single passive scalar (implementing the `passive_scalars` property sets this)
8) Can insert an immersed boundary (Peskin 2002). It is impermeable by default. (Implementing the `object_cells` property sets this)
9) Gravity source terms (Implementing the `gravity_sources` property sets this)

TODO: 
  - [ ] Explore general IB in greater detail for sources and sinks!
  - [ ] multi-gpu support
  - [ ] MPI support





