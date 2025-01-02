project('simbi', 'c', 'cpp', 'cython',
    version : run_command('git', 'describe', '--tag', '--abbrev=0', check: true).stdout().strip(), 
    license : 'MIT',
    default_options : { 
        'cpp_std': 'c++20', 
        'cpp_args': ['-march=native'],
    },
)

# Silence deprecated numpy API warnings
add_project_arguments(['-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION', '-DCYTHON_EXTERN_C=extern "C++"'], language : 'cpp')

# Build configuration
conf_data = configuration_data()
foreach opt : ['four_velocity', 'column_major', 'progress_bar', 'shared_memory']
    conf_data.set10(opt.to_upper(), get_option(opt))
endforeach
conf_data.set10('FLOAT_PRECISION', get_option('precision') == 'single')
configure_file(input: 'build_options.hpp.in', output: 'build_options.hpp', configuration: conf_data)

# Dependencies
fs = import('fs')
pymod = import('python')
py3 = pymod.find_installation('python3', required: true)
py3_dep = py3.dependency()
omp = dependency('openmp', language: 'cpp')
hdf5 = dependency('hdf5', language: 'cpp', required: true)
depends = [py3_dep, hdf5, omp]

# Python install directory
py3_purelib = py3.get_path('purelib')
py3env_prefix = py3.get_variable('prefix', 'empty')
if py3_purelib.endswith('site-packages')
    py3_install_dir = py3_purelib
elif py3env_prefix.endswith('site-packages')
    py3_install_dir = py3env_prefix
else
    py3_install_dir = py3.get_install_dir()
endif

# Check if on a Mac
if host_machine.system() == 'darwin'
    homebrew = find_program('brew')
    if homebrew.found()
        brew_prefix = run_command(['brew', '--prefix'], check: true).stdout().strip()
        add_project_arguments([f'-I{brew_prefix}/include'], language: 'cpp')
    endif
endif

# Install directory permission
install_dir_permission = run_command(py3, ['-c', f'import os; print(os.access("{py3_install_dir}", os.W_OK))'], check: true).stdout().strip()
pip_args = install_dir_permission == 'False' ? '--user' : []

# Include directories
incdir_numpy = run_command(py3, ['-c', 'import numpy; print(numpy.get_include())'], check: true).stdout().strip()
all_inc = include_directories(incdir_numpy, 'src')

# Compiler settings
host_compiler = meson.get_compiler('cpp')
host_compiler_name = host_compiler.get_id()
max_errors = host_compiler_name.startswith('g') ? '-fmax-errors=1' : '-ferror-limit=1'
add_project_arguments([max_errors], language: 'cpp')

# if CC, CXX environment variables explicitly set, get their values
# this is useful because cuda lags behind on tested gcc versions
cc_var = run_command(py3, '-c', 'import os; print(os.environ["CC"])', check: false).stdout().strip()
if cc_var == ''
  cc_var = host_compiler_name
endif

# GPU Compilation
gpu_dict = {
    'standard': 'c++20',
    'arch': 'sm_86',
    'cc_var': cc_var,

}
gpu_depends = []
gpu_linkers = []
gpu_comp_args = []
if get_option('gpu_compilation').enabled()
    hip = dependency('HIP', cmake_module_path: '/opt/rocm', modules: ['hip::device', 'hip::host'], required: false)
    cuda = dependency('CUDA', required: false)
    if hip.found() or cuda.found()
        gpu_cc = hip.found() ? 'hipcc' : 'nvcc'
        gpu_compiler = find_program(gpu_cc)
        gpu_arch = get_option('gpu_arch')
        if hip.found()
            hip_platform = run_command('hipconfig', '--platform', check: true).stdout().strip()
            gpu_comp_args = hip_platform == 'nvidia' ? [
                '-DGPU_PLATFORM_NVIDIA=1',
                '-std=c++20',
                f'-arch=sm_@gpu_arch@',
                f'-ccbin=@cc_var@',
                '--extended-lambda',
                '-dc',
                '-dlto',
                '--expt-relaxed-constexpr',
                '--compiler-options', 
                '-fPIC', '-O3', '-x=cu'
            ] : [
                '-DGPU_PLATFORM_AMD=1',
                '-std=c++20',
                '-fPIC',
                max_errors, 
                '-O3', 
                f'--offload-arch=gfx{gpu_arch}',
                '-march=native',
            ]
            gpu_linkers = hip_platform == 'nvidia' ? ['-lcudart', '-dlto', '-lcudadevrt', '-dlink'] : ['-lamdhip64']
            if hip_platform == 'amd'
                rocm_path = run_command('hipconfig', '--rocmpath', check: true).stdout().strip()
                hip_dep = declare_dependency(
                    link_args: [f'-L{rocm_path}/lib', '-lamdhip64', '-O3', '-lgcc_s', '-lpthread', '-lm', '-lrt'],
                    include_directories: [f'{rocm_path}/include']
                )
                gpu_depends += [hip_dep]
            endif
        elif cuda.found()
            gpu_comp_args = [
                '-std=c++20',
                '-DGPU_PLATFORM_NVIDIA=1',
                f'-arch=sm_@gpu_arch@', 
                f'-ccbin=@cc_var@',
                '-dc',
                '-dlto',
                '--extended-lambda',
                '--expt-relaxed-constexpr',
                '--compiler-options', 
                '-fPIC', max_errors, '-O3',
                '-x=cu',
            ]
            gpu_linkers = ['-lcudart', '-dlto']
        endif
    endif
endif

# Source and header files
sources = [
    'src/common/helpers.cpp',
    'src/common/exceptions.cpp',
    'src/hydro/rmhd.cpp',
    'src/hydro/srhd.cpp',
    'src/hydro/newt.cpp',
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

templates = [
    'src/common/helpers.ipp',
    'src/hydro/srhd.ipp',
    'src/hydro/newt.ipp',
    'src/hydro/rmhd.ipp',
    'src/util/ndarray.ipp',
    'src/util/printb.ipp',
]

# CPU Extension Module
cpu_ext = py3.extension_module(
    'cpu_ext',
    ['src/cpu_ext.pyx', 'src/call_obj.pyx'] + sources + headers + templates,
    include_directories: all_inc,
    cpp_args: ['-DGPU_CODE=0'],
    dependencies: depends,
    override_options: ['cython_language=cpp'],
    install: true,
    install_dir: meson.current_source_dir() + '/simbi/libs',
)

# GPU Extension Module
if get_option('gpu_compilation').enabled() and (hip.found() or cuda.found())
    gpu_objs = []
    gpu_includes = [
        '-I' + meson.current_source_dir() + '/src',
        '-I.',
        '-I' + meson.current_build_dir(),
        '-I' + get_option('gpu_include_dir'),
        '-I' + get_option('hdf5_include_dir')
    ]
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
            depend_files: headers + templates,
        )
        gpu_objs += [gpu_trg]
    endforeach

    # link the device objects with link time optimization
    gpu_lto_obj = custom_target(
        'gpu_lto_obj',
        command : [
            gpu_compiler,
            gpu_objs,
            [
                '-arch=sm_86',
                # '-fuse-linker-plugin',
                '-fPIC',
                # '-march=native',
            ],
            gpu_linkers + ['-lhdf5'],
            '-o',
            '@OUTPUT@',
            '-DGPU_CODE=1',
            '-DMANAGED_MEMORY',
        ],
        input : gpu_objs,
        output : 'gpu_lto.o',
        build_by_default: true,
    )

    # combine the device objects and lto objects into a static library
    # that the cpu will link against
    gpu_lib = static_library(
        'simbi_gpu',
        gpu_objs + [gpu_lto_obj],
        # include_directories: all_inc,
        link_args: gpu_linkers + ['-lhdf5'],
        build_by_default: true,
        cpp_args: ['-fPIC'],
        # install: true,
        # install_dir: meson.current_source_dir() + '/simbi/libs',
    )

    gpu_ext = py3.extension_module(
        'gpu_ext',
        ['src/gpu_ext.pyx', 'src/call_obj.pyx'],
        link_with: gpu_lib,
        include_directories: all_inc,
        dependencies: depends + gpu_depends,
        cpp_args: ['-DMANAGED_MEMORY'],
        override_options: ['cython_language=cpp'],
        install: true,
        install_dir: meson.current_source_dir() + '/simbi/libs',
    )
endif

# Radiation Module
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