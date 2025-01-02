# Main Meson Build File

project(
    'simbi',
    'c',
    'cpp',
    'cython',
    version: run_command('git', 'describe', '--tag', '--abbrev=0', check: true).stdout().strip(),
    license: 'MIT',
    default_options: {'cpp_std': 'c++20', 'cpp_args': ['-march=native']},
)

# Add common project arguments
add_project_arguments(
    [
        '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
        '-DCYTHON_EXTERN_C=extern "C++"',
    ],
    language: 'cpp',
)
# Python installation directory
py3 = import('python').find_installation('python3', required: true)
py_install_dir = meson.current_source_dir() + '/simbi/libs'

# Build configuration
conf_data = configuration_data()
foreach opt : ['four_velocity', 'column_major', 'progress_bar', 'shared_memory']
    conf_data.set10(opt.to_upper(), get_option(opt))
endforeach
conf_data.set10('FLOAT_PRECISION', get_option('precision') == 'single')
configure_file(input: 'build_options.hpp.in', output: 'build_options.hpp', configuration: conf_data)

# get numpy headers
incdir_numpy = run_command(py3, ['-c', 'import numpy; print(numpy.get_include())'], check: true).stdout().strip()

# Include Subdirectories
subdir('src')
subdir('simbi/afterglow')

general_dependencies = [
    dependency('openmp', language: 'cpp'),
    dependency('hdf5', language: 'cpp', required: true),
    py3.dependency()
]
# CPU Extension Module
py3.extension_module(
    'cpu_ext',
    ['src/cpu_ext.pyx'] + sources,
    include_directories: include_directories('src', incdir_numpy),
    cpp_args: ['-DGPU_CODE=0'],
    dependencies: general_dependencies,
    override_options: ['cython_language=cpp'],
    install: true,
    install_dir: py_install_dir,
)

# Check and configure GPU compilation
if get_option('gpu_compilation').enabled()
    subdir('gpu')
    py3.extension_module(
        'gpu_ext',
        ['src/gpu_ext.pyx'],
        link_with: gpu_lib,
        include_directories: include_directories('src', incdir_numpy),
        dependencies: general_dependencies + settings['dependency'],
        cpp_args: ['-DMANAGED_MEMORY'],
        override_options: ['cython_language=cpp'],
        install: true,
        install_dir: meson.current_source_dir() + '/simbi/libs',
    )
endif