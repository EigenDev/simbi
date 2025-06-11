# Main Meson Build File

project(
    'simbi',
    'c',
    'cpp',
    version: run_command('git', 'describe', '--tag', '--abbrev=0', check: true).stdout().strip(),
    license: 'MIT',
    default_options: {'cpp_std': 'c++20', 'cpp_args': ['-march=native'], 'warning_level': '3'},
)

# Add common project arguments
add_project_arguments(
    [
        '-Wno-nested-anon-types',
        '-Wno-gnu-anonymous-struct',
    ],
    language: 'cpp',
)
# Python installation directory
py3 = import('python').find_installation('python3', required: true)
py_install_dir = meson.current_source_dir() + '/simbi/libs'
pybind11_dep = dependency('pybind11')
pybind11_include = run_command(
    py3, ['-c', 'import pybind11; print(pybind11.get_include())'], check: true
).stdout().strip()


# show maximum of n errors before stopping the build
# cc = meson.get_compiler('cpp')
# if cc.get_id() == 'clang'
#     add_project_arguments('-ferror-limit=1', language : 'cpp')
# elif cc.get_id() == 'gcc'
#     add_project_arguments('-fmax-errors=1', language : 'cpp')
# endif
#Build configuration
conf_data = configuration_data()
foreach opt : ['four_velocity', 'column_major', 'progress_bar', 'shared_memory']
    conf_data.set10(opt.to_upper(), get_option(opt))
endforeach
conf_data.set10('FLOAT_PRECISION', get_option('precision') == 'single')
configure_file(input: 'build_options.hpp.in', output: 'build_options.hpp', configuration: conf_data)

openmp_dep = dependency('openmp', language: 'cpp', required: true)
hdf5_dep = dependency('hdf5', language: 'cpp', required: true)

#Include Subdirectories
subdir('src')
subdir('simbi/afterglow')


general_dependencies = [
    openmp_dep,
    hdf5_dep,
    py3.dependency(),
    pybind11_dep,
]
all_inc = include_directories('src')
#CPU Extension Module
py3.extension_module(
    'cpu_ext',
    ['src/core/pybind11/cpu_binding.cpp'] + sources,
    include_directories: all_inc,
    cpp_args: ['-DGPU_ENABLED=0'],
    dependencies: general_dependencies,
    install: true,
    install_dir: py_install_dir
)

#Check and configure GPU compilation
if get_option('gpu_compilation').enabled()
    fs = import('fs')
    hdf5_inc = get_option('hdf5_inc')
    python_include = run_command(
        py3, ['-c', 'import sysconfig; print(sysconfig.get_path("include"))'],
        check: true
    ).stdout().strip()

    # Build include directories list
    include_dirs = [
        meson.current_source_dir() / 'src',
        '.',
        meson.current_build_dir(),
        pybind11_include,
        python_include,
        hdf5_inc,
        numpy_include,
    ]

    if hdf5_inc == ''
        error('HDF5 include directory was not found for GPU compilation')
    endif
    subdir('gpu')
endif
