# SIMBI: 3D Relativistic Hydro Code

# 2D Relativistic Jet Simulation

<https://user-images.githubusercontent.com/29236993/145315802-c8d7e8c5-7beb-488c-b496-b9edf404be2e.mp4>

# 1D Moving mesh techniques

<https://user-images.githubusercontent.com/29236993/205418982-943af187-8ae3-4401-92d5-e09a4ea821e2.mp4>



# Requirements

1)  GCC \>= GCC5 (for c++17 support)
2)  Cython
3)  [meson](https://mesonbuild.com/Getting-meson.html)
    `pip install meson` will usually suffice
4)  A build system like `make` or
    [ninja](https://github.com/ninja-build/ninja/wiki/Pre-built-Ninja-packages)
    -- pre-built packages depending on your system.
5)  HDF5 libraries

## For GPU capability

6)  HIP/ROCm if wanting to run on NVIDIA or AMD GPUs, or just CUDA if
    running purely NVIDIA

## Quick setup guide

<strong>Easy Install</strong>
1) Run 
```bash
$ CXX=<your_cpp_compiler> . ./install.sh [options]
```

<strong>Manual Install</strong>
1)  Run

``` bash
$ CXX=<your_cpp_compiler> meson setup <build_dir> -D<some_option>
```

from project root. It is important that this directory is not named
`build` because the `install` call with create `build` dir for the `pip`
installation part. For the `-D<build_option>` part, check the
`meson_options.txt` file for available build options.

2)  Run

``` bash
$ ninja -v -C <build_dir> install
```

or

``` bash
$ meson install -C <build_dir>
```

3)  If `meson` detected `hip` or `cuda`, the install script will install
    both the cpu and gpu extensions into your system site-packages or
    `--user` site-packages depending on privileges.
4)  If all is well, we can test. To test, try running the configuration
    scripts provided. For example:

    ``` bash
    $ simbi configs/marti_muller.py --mode gpu --nzones 100 --ad_gamma 1.4 
    # or one could do 
    $ simbi marti_muller --mode gpu --nzones 100 --ad_gamma 1.4
    # since the entry point is built to recursively search the configs/ folder for valid .py scripts now
    ```

    where `--mode` is a global command line option available for every
    config script, and `--nzones` and `--gamma` are problem-specific options
    that are dynamically parsed based on whatever `DynamicArg` variables
    exist in the config script you create. Check out how to create one of
    these configuration scripts in the `simbi/configs/` folder! When creating
    your own configuration file, you must place it in a directory entitled `configs/` and
    run `simbi` from your `configs/` parent directory and `simbi` should auto detect your configuration and run the simulation.

    You can plot the above output by running 
    ``` bash
    $ ./tools/plot1d.py data/1000.chkpt.000_100.h5 "Marti \& Muller Problem 1" --field rho v p --tex
    ```
5)  ???
6)  Profit


