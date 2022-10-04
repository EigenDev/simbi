# SIMBI: 3D Relativistic Hydro Code

## 2D Relativistic Jet Simulation
https://user-images.githubusercontent.com/29236993/145315802-c8d7e8c5-7beb-488c-b496-b9edf404be2e.mp4

## 1D Moving mesh techniques
https://user-images.githubusercontent.com/29236993/173423001-53ab2b60-4159-4ce5-a5a9-de0095b6870a.mp4


## Requirements 
1) GCC >= GCC5 (for c++17 support)
2) Cython 
3) [meson](https://mesonbuild.com/Getting-meson.html) `pip install meson` will usually suffice
4) A build system like `make` or [ninja](https://github.com/ninja-build/ninja/wiki/Pre-built-Ninja-packages) -- pre-built packages depending on your system.
5) HDF5 libraries
### For GPU capability
6) HIP/RocM if wanting to run on NVIDIA or AMD GPUs, or just CUDA if running purely NVIDIA


### Quick setup guide
1) Run `CXX=<your_cpp_compiler> meson setup <build_dir> -D<some_option>` from project root. It is important that this directory is not named `build` because the `install` call with create `build` dir for the `pip` installation part. 
For the `-D<build_option>` part, check the `meson_options.txt` file for available build options. 
2) Run `ninja -v -C <build_dir> install` or  `meson install -C <build_dir>`
3) If `meson` detected `hip` or `cuda`, the install script will install both the cpu and gpu extensions into your system site-packages or `--user` site-packages depending on privileges.  
4) If all is well, we can test. To test, try running the example scripts provided. For example<br>
 `./examples/sod_test.py --nzones 512 --mode cpu --cfl 0.1 --bc outflow --tend 0.2` 
5) ???
6) Profit