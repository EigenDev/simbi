# Cython Compile the Hydro Code

# Must run with python setup.py build_ext --inplace

from distutils.core import setup
from Cython.Build import cythonize 

setup(
    name = "hydrocode",
    ext_modules= cythonize('state.pyx'),
)