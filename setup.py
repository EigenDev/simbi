# Cython Compile the Hydro Code

# Must run with python setup.py build_ext --inplace

import os
import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize 

with open("README.md", "r", encoding = "utf-8") as fh:
    description = fh.read()
    
headerfile = ['helper_functions.h']

sourcefiles = ['src/state.pyx', 
               'src/simbi_1d.cpp', 
               'src/relativistic1D.cpp', 
               'src/helper_functions.cpp', 
               'src/simbi_2d.cpp', 
               'src/relativistic2D.cpp',
               'src/clattice.cpp',
               'src/hydro_structs.cpp',
               'src/viscous_diff.cpp']

extensions = [Extension("state", sourcefiles, 
                        include_dirs=[numpy.get_include(), "helper_functions.h"],
                        libraries=['hdf5', 'hdf5_hl', 'hdf5_cpp'],
                        extra_compile_args = ['-std=c++11', '-march=native', '-fno-wrapv', '-O3'] )]

os.environ["CC"] = ("clang++")
setup(
    name="SIMBI 2D Hydro Code",
    version="0.0.1",
    author="M. DuPont",
    author_email="md4469@nyu.edu",
    description="Cython module to solve hydrodynamic systems using a c++ backend",
    ext_modules=cythonize(extensions),
    packages=['simbi_py'],
    # install_requires=['numpy', 'matplotlib', 'cython'],
    # python_requires='>=3.6',
)