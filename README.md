# SIMBI: 3D Relatavisitc Hydro Code

## 2D Relativisitc Jet Simulation
https://user-images.githubusercontent.com/29236993/145315802-c8d7e8c5-7beb-488c-b496-b9edf404be2e.mp4

## 1D Moving mesh techniques
https://user-images.githubusercontent.com/29236993/172739739-1714ab98-0a20-4334-a28f-c133d17e8e18.mp4


https://user-images.githubusercontent.com/29236993/172739751-54c8a1da-5663-4da5-8c47-98c79b2b37c6.mp4

## Requirements 
1) GCC < GCC-11 (Cython doesn't play nicely with gcc 11 for some reason)
2) Cython 
3) HDF5 libraries
### For GPU capability
4) HIP/RocM if wanting to run on GPUs


### Quick setup guide
1) Create a build directory for out-of-source compilation:<br>
 `mkdir build`
2) Change directories into that build directory, i.e, `cd build`, and do: `cmake ..`
3) Make sure the necessary files were added by running `ls`. Afterwards, run `make install`.<br>
If Cmake found HIP, `make install` builds the gpu extension. If it did not, it builds to cpu extension.<br>
You can check which extension was built by looking in the `build/lib/` directory for either<br>
`gpu_ext.so` or `cpu_ext.so`

4) If you want to have both extensions after installation, do `make gpu` if you do not have the `gpu_ext.so`
library <br> 
or do `make cpu` if you do not have the `cpu_ext.so` library. After those install, you should be up and
running. 
5) To test, try running the example scripts provided. For example<br>
 `./examples/sod_test.py --nzones 512 --mode cpu --cfl 0.1 --bc outflow --tend 0.2` 
6) ???
7) Profit
