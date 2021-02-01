# Cython Compile the Hydro Code

# Must run with python setup.py build_ext --inplace

import os
import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize 

with open("README.md", "r", encoding = "utf-8") as fh:
    description = fh.read()
    
sourcefiles = ['state.pyx', 'simbi_1d.cpp', 'relativistic1D.cpp', 'helper_functions.cpp', 'simbi_2d.cpp', 'relativistic2D.cpp']

extensions = [Extension("state", sourcefiles, 
                        include_dirs=[numpy.get_include()],
                        libraries=['hdf5', 'hdf5_hl'],
                        library_dirs=['/usr/local/lib/'],
                        extra_compile_args = ['-std=c++11', '-march=native', '-fno-wrapv', '-O3'] )]

os.environ["CC"] = ("g++ -o -DNDEBUG -g -O2 -Wall -Wstrict-prototypes " +
                    "-fno-strict-aliasing -Wdate-time -D_FORTIFY_SOURCE=2 -g -fstack-protector-strong " +
                    "-Wformat -Werror=format-security -fPIC -ftrapv")
setup(
    name="SIMBI 2D Hydro Code",
    version="0.0.1",
    author="M. DuPont",
    author_email="md4469@nyu.edu",
    description="Cython module to solve hydrodynamic systems using a c++ backend",
    ext_modules=cythonize(extensions),
    #packages=['simbi_py'],
    #install_requires=['numpy', 'matplotlib', 'cython'],
    #python_requires='>=3.6',
)