# Cython Compile the Hydro Code

# Must run with python setup.py build_ext --inplace

import os
import numpy
from distutils.core import setup
from distutils.extension import Extension
from distutils import sysconfig

from Cython.Build import cythonize 

with open("README.md", "r", encoding = "utf-8") as fh:
    description = fh.read()

compiler_args = ['-std=c++17', '-march=native', '-fno-wrapv', '-O3']
linker_args   = ['-lhdf5', '-lhdf5_cpp']
libraries     = ['hdf5', 'hdf5_cpp']
library_dirs  = []
language = "c++"
sources  = ["src/state.pyx"]
source_path = "src/"
headers  = []
for file in os.listdir("src"):
    if file.endswith(".cpp") and file != "state.cpp":
        sources.append(source_path + file)
    if file.endswith(".hpp") or file.endswith(".h"):
        headers.append(source_path + file)



def extensions():
    '''
    Handle generation of extensions (a.k.a "managing cython compilery").
    '''
    try:
        from Cython.Build import cythonize
    except ImportError:
        def cythonize(*args, **kwargs):
            print("Hint: Wrapping import of cythonize in extensions()")
            from Cython.Build import cythonize
            return cythonize(*args, **kwargs)

    try:
        import numpy
        lstIncludes = [numpy.get_include()]
    except ImportError:
        lstIncludes = []

    extensionArguments = {
        'include_dirs':
        lstIncludes,
        'library_dirs': library_dirs,
        'extra_compile_args': compiler_args,
        'extra_link_args': linker_args,
        'libraries': libraries,
        'language': 'c++'
    }

    # me make damn sure, that disutils does not mess with our
    # build process

    sysconfig.get_config_vars()['CFLAGS'] = ''
    sysconfig.get_config_vars()['OPT'] = ''
    sysconfig.get_config_vars()['PY_CFLAGS'] = ''
    sysconfig.get_config_vars()['PY_CORE_CFLAGS'] = ''
    sysconfig.get_config_vars()['CC'] = 'clang'
    sysconfig.get_config_vars()['CXX'] = 'clang++'
    sysconfig.get_config_vars()['BASECFLAGS'] = ''
    sysconfig.get_config_vars()['CCSHARED'] = '-fPIC'
    sysconfig.get_config_vars()['LDSHARED'] = 'clang -shared'
    sysconfig.get_config_vars()['CPP'] = 'clang++'
    sysconfig.get_config_vars()['CPPFLAGS'] = ''
    sysconfig.get_config_vars()['BLDSHARED'] = ''
    sysconfig.get_config_vars()['CONFIGURE_LDFLAGS'] = ''
    sysconfig.get_config_vars()['LDFLAGS'] = ''
    sysconfig.get_config_vars()['PY_LDFLAGS'] = ''
    
    return cythonize(
        [Extension("state", sources, **extensionArguments)]
    )
    
# set the compiler
os.environ["CC"]  = "clang++"
os.environ["CXX"] = "clang++"
setup(
    name="SIMBI 2D Hydro Code",
    version="0.0.1",
    author="M. DuPont",
    author_email="md4469@nyu.edu",
    description="Cython module to solve hydrodynamic systems using a c++ backend",
    ext_modules=extensions(),
    packages=['simbi_py'],
    # install_requires=['numpy', 'cython'],
    # python_requires='>=3.6',
)

# Below is the old way of how I setup the compiler
# It was not portable initially

# headerfile = ['helper_functions.h']

# sourcefiles = ['src/state.pyx', 
#                'src/simbi_1d.cpp', 
#                'src/relativistic1D.cpp', 
#                'src/helper_functions.cpp', 
#                'src/simbi_2d.cpp', 
#                'src/relativistic2D.cpp',
#                'src/clattice.cpp',
#                'src/hydro_structs.cpp',
#                'src/viscous_diff.cpp',
#                'src/clattice_1d.cpp']
# lextensions = [Extension("state", sourcefiles, 
#                         include_dirs=[numpy.get_include(), "helper_functions.h"],
#                         libraries=['hdf5', 'hdf5_hl', 'hdf5_cpp'],
#                         extra_compile_args = ['-std=c++11', '-march=native', '-fno-wrapv', '-O3'] )]