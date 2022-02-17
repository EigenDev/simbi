# SIMBI: 2D Relatavisitc Hydro Code

https://user-images.githubusercontent.com/29236993/145315802-c8d7e8c5-7beb-488c-b496-b9edf404be2e.mp4


## Requirements 
1) GCC < GCC-11 (Cython doesn't play nicely with gcc 11 for some reason)
2) Cython 
3) HDF5 libraries
### For GPU capability
4) HIP/RocM if wanting to run on GPUs


### Quick setup guide
1) Create a build directory for out-of-source compilation: `mkdir build`
2) Change directories into that build directory, i.e, `cd build`, and do: `cmake ..`
3) Make sure the ncessary files were added by running `ls`. Afterwards, run `make install`. If Cmake found HIP, `make install` builds the gpu extension. If it did not, it builds to cpu extension. You can check which extensions was 
build by looking in the `build/lib/` directory for either `gpu_ext.so` or `cpu_ext.so`
4) If you want to have both extensions after installation, do `make gpu` if you do not have the `gpu_ext.so`
library or do `make cpu` if you do not have the `cpu_ext.so` library. After those install, you should be up and
running. 
5) To test, try running the example scripts provided
6) ???
7) Profit