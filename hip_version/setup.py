# Cython Compile the Hydro Code

# Must run with python setup.py build_ext --inplace

import os
import numpy
import subprocess

from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from distutils import sysconfig
from Cython.Build import cythonize, build_ext


with open("README.md", "r", encoding = "utf-8") as fh:
    description = fh.read()
    
def find_in_path(name, path):
    """Find a file in a search path"""

    # Adapted fom http://code.activestate.com/recipes/52224
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    """

    # First check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # Otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, '
                'or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be '
                                   'located in %s' % (k, v))

    return cudaconfig

def locate_hip():
    
    # First check if the CUDAHOME env variable is in use
    if 'HIP_PATH' in os.environ:
        home     = os.environ['HIP_PATH']
        hipcc    = pjoin(home, 'bin', 'hipcc')
        platform = os.system("{}/bin/hipconfig --platform".format(home))
    else:
        # Otherwise, search the PATH for NVCC
        hipcc    = find_in_path('hipcc', os.environ['PATH'])
        if hipcc is None:
            raise EnvironmentError('The hipcc binary could not be '
                'located in your $PATH. Either add it to your path, '
                'or set $HIP_PATH')
        
        home = "/opt/rocm/hip"
        platform = subprocess.check_output("{}/bin/hipconfig --platform".format(home), shell = True).decode('utf-8')
        
    if platform == "nvidia":
        CUDA = locate_cuda()
            
    
    hipconfig = {'home':     home, 
                 'hipcc':    hipcc,
                 'include':  pjoin(home, 'include') if platform == "amd" else CUDA["include"],
                 'lib':      pjoin(home, 'lib') if platform == "amd" else CUDA["lib64"],
                 'platform': platform}
    
    print("HOME:",     home)
    print("CC:",       hipcc)
    print("PLATFORM:", platform)
    for k, v in iter(hipconfig.items()):
        if k == "platform":
            pass
        elif not os.path.exists(v):
            raise EnvironmentError('The HIP %s path could not be '
                                   'located in %s' % (k, v))
    return hipconfig

# CUDA = locate_cuda()
HIP  = locate_hip()




compilerArguments = {
            'g++': ['-std=c++17','-march=native', '-fno-wrapv', '-O3'],
            'hipcc': [
                ]
            }
if HIP["platform"] == "nvidia":
    compilerArguments["hipcc"] += ['-arch=sm_50', '-c', '--ptxas-options=-v', '--compiler-options', "-fPIC", "-O3"]
else:
    compilerArguments["hipcc"] += compilerArguments["g++"]
    
compiler_args = ['-march=native', '-fno-wrapv', '-O3']
linker_args   = ['-lhdf5', '-lhdf5_cpp']
libraries     = ['hdf5', 'hdf5_cpp'] 
library_dirs  = [HIP["lib"]]
extraIncludes = []
language = "c++"
if HIP["platform"] == "nvidia":
    defineMacros      = [("__HIP_PLATFORM_NVIDIA__", "1")]
    libraries         += ['cudart']
else:
    defineMacros      = [("__HIP_PLATFORM_AMD__", "1")]
    

sources  = ["src/gpu_state.pyx"]
source_path = "src/"
headers  = []
for file in os.listdir("src"):
    if file.endswith(".cpp") and file != "gpu_state.cpp":
        sources.append(source_path + file)
    if file.endswith(".hpp"):
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
        lstIncludes + [HIP['include']] + extraIncludes,
        'library_dirs': library_dirs,
        'extra_compile_args': compilerArguments,
        'extra_link_args': linker_args,
        'libraries': libraries,
        'runtime_library_dirs': [HIP["lib"]],
        'language': 'c++',
        'define_macros': defineMacros
    }

    # Ensure disutils does not mess with the build process
    sysconfig.get_config_vars()['CFLAGS'] = ''
    sysconfig.get_config_vars()['OPT'] = ''
    sysconfig.get_config_vars()['PY_CFLAGS'] = ''
    sysconfig.get_config_vars()['PY_CORE_CFLAGS'] = ''
    sysconfig.get_config_vars()['CC'] = 'gcc'
    sysconfig.get_config_vars()['CXX'] = 'g++'
    sysconfig.get_config_vars()['BASECFLAGS'] = ''
    sysconfig.get_config_vars()['CCSHARED'] = '-fPIC'
    sysconfig.get_config_vars()['LDSHARED'] = 'g++ -shared'
    sysconfig.get_config_vars()['CPP'] = 'g++'
    sysconfig.get_config_vars()['CPPFLAGS'] = ''
    sysconfig.get_config_vars()['BLDSHARED'] = ''
    sysconfig.get_config_vars()['CONFIGURE_LDFLAGS'] = ''
    sysconfig.get_config_vars()['LDFLAGS'] = ''
    sysconfig.get_config_vars()['PY_LDFLAGS'] = ''
    
    return [Extension("gpu_state", sources, **extensionArguments)]

ext = extensions()

def customize_compiler_for_hipcc(self):
    """Inject deep into distutils to customize how the dispatch
    to gcc/hipcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """

    # Tell the compiler it can processes .cu
    # self.src_extensions.append('.cu')

    # Save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # Now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if src != "src/gpu_state.cpp":
            # use the cuda for .cu files
            self.set_executable('compiler_so', HIP['hipcc'])
            # use only a subset of the extra_postargs, which are 1-1
            # translated from the extra_compile_args in the Extension class
            postargs = extra_postargs['hipcc']
        else:
            postargs = extra_postargs['g++']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # Reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # Inject our redefined _compile method into the class
    self._compile = _compile



# Run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_hipcc(self.compiler)
        build_ext.build_extensions(self)


    
setup(
    name = 'PySIMBI GPU',
    author = 'Marcus DuPont',
    author_email="md4469@nyu.edu",
    description="Cython module to solve hydrodynamic systems using a hip/c++ backend",
    version = '0.0.1',
    ext_modules = ext,
    # Inject our custom trigger
    cmdclass = {'build_ext': custom_build_ext},
    packages=['pysimbi_gpu'],
    # Since the package has c code, the egg cannot be zipped
    zip_safe = False)