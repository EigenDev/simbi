# Cython Compile the Hydro Code

# Must run with python setup.py build_ext --inplace

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize 

sourcefiles = ['state.pyx', 'simbi_1d.cpp', 'helper_functions.cpp', 'simbi_2d.cpp']

extensions = [Extension("state", sourcefiles)]

setup(
    ext_modules=cythonize(extensions)
)