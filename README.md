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
1) Run 
```bash 
$ CXX=<your_cpp_compiler> meson setup <build_dir> -D<some_option>
``` 
from project root. It is important that this directory is not named `build` because the `install` call with create `build` dir for the `pip` installation part. 
For the `-D<build_option>` part, check the `meson_options.txt` file for available build options. 

2) Run 
```bash 
$ ninja -v -C <build_dir> install
``` 
or  
```bash 
$ meson install -C <build_dir>
```
3) If `meson` detected `hip` or `cuda`, the install script will install both the cpu and gpu extensions into your system site-packages or `--user` site-packages depending on privileges.  
4) If all is well, we can test. To test, try running the example scripts provided. For example
 ```bash
 $ ./examples/sod_test.py --nzones 512 --mode cpu --cfl 0.1 --bc outflow --tend 0.2
``` 
5) ???
6) Profit

### Bonus
Another way to run the code is to create some configuration script and invoke it using the entry point. You would then run it like so:
```bash
$ pysimbi config/marti_muller.py --mode gpu --nzones 100 --gamma 1.4 
```
where `--mode` is a global command line option available for every config script, and `--nzones` and `--gamma` are problem-specific options that are dynamically parsed based on whatever `DynamicArg` variables exist in the config script you create.
Check out how to create one of these configuration scripts in the `config/` folder!