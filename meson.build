project('simbi', 'c', 'cpp', 'cython',
    version : run_command('git', 'describe', '--tag', '--abbrev=0', check: true).stdout().strip(), 
    license : 'MIT',
    default_options : { 'cpp_std': 'c++20', 'cpp_args': ['-march=native']},
)

# BUILD CONFIG
conf_data = configuration_data()
conf_data.set10('FLOAT_PRECISION', get_option('float_precision'))
conf_data.set10('FOUR_VELOCITY',   get_option('four_velocity'))
conf_data.set10('COLUMN_MAJOR',    get_option('column_major'))
conf_data.set10('PROGRESS_BAR',    get_option('progress_bar'))
conf_data.set10('SHARED_MEMORY',   get_option('shared_memory'))
configure_file(
    input:  'build_options.hpp.in',
    output: 'build_options.hpp',
    configuration: conf_data
)

# DEPENDENCIES
fs      = import('fs')
pymod   = import('python')
py3     = pymod.find_installation('python3', required: true)
py3_dep = py3.dependency()
omp     = dependency('openmp', language: 'cpp')
hdf5    = dependency('hdf5', language: 'cpp', required: true)
# ensure the install directory recognizes conda environment
py3_purelib   = py3.get_path('purelib')
py3env_prefix = py3.get_variable('prefix', 'empty')
message('python path -- ' + py3.path())
message('python install_dir -- ' + py3env_prefix)
if py3_purelib.endswith('site-packages')
    py3_install_dir = py3_purelib
elif py3env_prefix.endswith('site-packages')
    py3_install_dir = py3env_prefix
else
    py3_install_dir = py3.get_install_dir()
endif

# Check if on a Mac
host_system = host_machine.system()
if host_system == 'darwin'
    homebrew = find_program('brew')
    if homebrew.found()
        brew_prefix = run_command(['brew', '--prefix'], check: true).stdout().strip()
        brew_inc = f'-I@brew_prefix@/include'
        # brew_lib = f'@brew_prefix@/lib'
        add_project_arguments([brew_inc], language: 'cpp')
    endif
endif

install_dir_permission = run_command(py3,
  ['-c', f'import os; print(os.access("@py3_install_dir@", os.W_OK))'],
  check : true
).stdout().strip()

pip_args = []
if install_dir_permission == 'False'
    py3_install_dir = run_command('python', '-m', 'site', '--user-site', check: true).stdout().strip()
    pip_args='--user'
endif
#################################################################
# SOME BOOK KEEPING FOR ALL INSTALLS
##################################################################
incdir_numpy = run_command(py3,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()
all_inc       = include_directories(incdir_numpy, 'src')
depends       = [py3_dep, hdf5, omp]
gpu_depends   = []
gpu_linkers   = []
gpu_comp_args = []
host_compiler  = meson.get_compiler('cpp')
host_compiler_name = host_compiler.get_id()
if host_compiler_name == 'g++' or host_compiler_name == 'gcc'
max_errors = '-fmax-errors=1'
add_project_arguments([max_errors], language: 'cpp')
elif host_compiler_name == 'clang++' or host_compiler_name == 'clang'
max_errors = '-ferror-limit=1'
add_project_arguments([max_errors], language: 'cpp')
endif

# if CC, CXX environment variables explicitly set, get their values
# this is useful because cuda lags behind on tested gcc versions
cc_var = run_command(py3, '-c', 'import os; print(os.environ["CC"])', check: false).stdout().strip()
if cc_var == ''
  cc_var = host_compiler_name
endif

##################################################################
# GPU CHECK
##################################################################
if get_option('gpu_compilation').enabled()
    hip  = dependency('HIP', cmake_module_path : '/opt/rocm', modules: ['hip::device', 'hip::host'], required: false)
    cuda = dependency('CUDA', required: false)
    host_compiler_version = host_compiler.version().split('.')[0]
    if host_compiler_name == 'g++' or host_compiler_name == 'gcc'
        if host_compiler_version.to_int() >= 13
            warning(f'The detected gcc version: gcc-@host_compiler_version@ might not be compatible with cuda version')
        endif
    endif

    if hip.found()
        env = environment()
        check_hip_config = run_command('hipconfig','--platform', check: true)
        hip_platform     = check_hip_config.stdout().strip()
        hip_platform_err = check_hip_config.stderr().strip()
        gpu_arch = get_option('gpu_arch')
        if hip_platform == 'nvidia'
            hip_runtime  = 'cuda'
            hip_compiler = 'nvcc'
            gpu_depends += [cuda]
            gpu_linkers += ['-lcudart']
            gpu_comp_args = [
                '-DGPU_PLATFORM_NVIDIA=1',
                '-std=c++17',
                f'-arch=sm_@gpu_arch@',
                f'-ccbin=@cc_var@',
                '--ptxas-options=-v', 
                '--extended-lambda',
                '--compiler-options', 
                f'-fPIC', '-O3', '-x=cu'
            ]
        elif hip_platform == 'amd' 
            hip_runtime  = 'rocm'
            hip_compiler = 'hipcc'
            gpu_linkers += ['-lamdhip64']
            gpu_comp_args = [
                '-DGPU_PLATFORM_AMD=1',
                '-std=c++17',
                '-fPIC',
                f'@max_errors@', 
                '-O3', 
                f'--offload-arch=gfx@gpu_arch@',
                '-march=native',
            ]
            rocm_path = run_command('hipconfig', '--rocmpath', check: true).stdout().strip()
            # Meson does not support hip yet, so we manually
            # declare the dependency
            hip_dep = declare_dependency(
                link_args: [f'-L@rocm_path@/lib', '-lamdhip64', '-O3', '-lgcc_s', '-lpthread', '-lm', '-lrt'],
                include_directories: [f'@rocm_path@/include']
            )
            gpu_depends += [hip_dep]
        endif
        env.set('HIP_PLATFORM', hip_platform)
        env.set('HIP_RUNTIME',  hip_runtime)
        env.set('HIP_COMPILER', hip_compiler)
        message(f'GPU Platform -- @hip_platform@')
        message(f'GPU Runtime  -- @hip_runtime@')
        message(f'GPU Compiler -- @hip_compiler@')
    elif cuda.found()
        gpu_depends += [cuda]
        gpu_linkers += ['-lcudart']
        gpu_arch = get_option('gpu_arch')
        gpu_comp_args = [
                '-std=c++17',
                '-DGPU_PLATFORM_NVIDIA=1',
                f'-arch=sm_@gpu_arch@', 
                f'-ccbin=@cc_var@',
                '--ptxas-options=-v', 
                '-lineinfo',
                '--extended-lambda',
                '--compiler-options', 
                f'-fPIC,@max_errors@', '-O3',
                '-x=cu',
        ]
        message(f'GPU Platform -- nvidia')
        message(f'GPU Runtime  -- cuda')
        message(f'GPU Compiler -- nvcc')
    endif
endif

sources = [
    #######################
    ### Common IMPLEMENTATIONS
    #######################
    'src/common/helpers.cpp',

    #######################
    #### GPU / CPU IMPLEMENTATIONS
    #######################
    'src/hydro/state.cpp',
    
    #######################
    #### UTIL IMPLEMENTATIONS
    #######################
    'src/util/device_api.cpp',
]

headers = [
    'src/common/helpers.hpp',
    'src/common/enums.hpp',
    'src/common/hydro_structs.hpp',
    'src/common/traits.hpp',
    
    'src/hydro/srhd.hpp',
    'src/hydro/newt.hpp',
    'src/hydro/rmhd.hpp',

    'src/util/device_api.hpp',
    'src/util/exec_policy.hpp',
    'src/util/kernel.hpp',
    'src/util/launch.hpp',
    'src/util/logger.hpp',
    'src/util/managed.hpp',
    'src/util/ndarray.hpp',
    'src/util/parallel_for.hpp',
    'src/util/printb.hpp',
    'src/util/range.hpp',
]


run_command('python', '-m', 'cython', '--cplus', 'src/call_obj.pyx', check: true)
cpu_ext = py3.extension_module(
    'cpu_ext',
    ['src/cpu_ext.pyx', 'src/call_obj.pyx'] + sources + headers,
    include_directories: all_inc,
    cpp_args: ['-DGPU_CODE=0'],
    dependencies: depends,
    override_options: ['cython_language=cpp'],
    install: true,
    install_dir: meson.current_source_dir() + '/simbi/libs',
)

if get_option('gpu_compilation').enabled()
    if hip.found() or cuda.found()
        gpu_cc = hip.found() ? 'hipcc' : 'nvcc'
        gpu_compiler = find_program(f'@gpu_cc@')
        gpu_objs = []
        gpu_includes = ['-I'+meson.current_source_dir()+'/src', '-I.', '-I'+meson.current_build_dir()]
        gpu_includes += ['-I' + get_option('gpu_include_dir'), '-I' + get_option('hdf5_include_dir')]
        gpu_link_trgs = []
        foreach source : sources
            gpu_trg = custom_target(
                fs.name(source) + '_target',
                command : [
                    gpu_compiler,
                    gpu_includes,
                    gpu_comp_args,
                    '-c',
                    '@INPUT@',
                    '-o',
                    '@OUTPUT@',
                    '-DGPU_CODE=1',
                    '-DMANAGED_MEMORY',
                ],
                input : source,
                output : '@BASENAME@.o',
                build_by_default: true,
                depend_files: headers,
            )
            gpu_objs += [gpu_trg]
        endforeach

        gpu_lib = custom_target(
            'gpu_library',
            command: [
                'ar',
                'rcs',
                '@OUTPUT@',
                '@INPUT@',
            ],
            input: gpu_objs,
            output: 'libsimbi_gpu.a',
            install: false, 
            build_by_default: true,
        )
        gpu_ext = py3.extension_module(
            'gpu_ext',
            ['src/gpu_ext.pyx', 'src/call_obj.pyx'],
            link_with: [gpu_lib],
            include_directories: all_inc,
            dependencies: depends + gpu_depends,
            cpp_args: ['-DMANAGED_MEMORY'],
            override_options: ['cython_language=cpp'],
            install: true,
            install_dir: meson.current_source_dir() + '/simbi/libs',
        )
    endif
endif

#-------------
# Radiation
#-------------
rad_src = [
    'simbi/afterglow/src/rad_units.cpp',
    'simbi/afterglow/src/rad_units.hpp',
    'simbi/afterglow/src/units/units.hpp'
]

py3.extension_module(
    'rad_hydro',
    ['simbi/afterglow/src/rad_hydro.pyx'] + rad_src,
    include_directories: include_directories('simbi/afterglow/src', incdir_numpy),
    dependencies: [py3_dep, omp],
    override_options: ['cython_language=cpp'],
    install: true,
    install_dir: meson.current_source_dir() + '/simbi/libs'
)