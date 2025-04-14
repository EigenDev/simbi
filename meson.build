#Main Meson Build File

project(
    'simbi',
    'c',
    'cpp',
    'cython',
    version: run_command('git', 'describe', '--tag', '--abbrev=0', check: true).stdout().strip(),
    license: 'MIT',
    default_options: {'cpp_std': 'c++20', 'cpp_args': ['-march=native']},
)

#Add common project arguments
add_project_arguments(
    [
        '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
        '-DCYTHON_EXTERN_C=extern "C++"',
    ],
    language: 'cpp',
)
#Python installation directory
py3 = import('python').find_installation('python3', required: true)
py_install_dir = meson.current_source_dir() + '/simbi/libs'

#show maximum of five errors before stopping the build
# cc = meson.get_compiler('cpp')
# if cc.get_id() == 'clang'
# add_project_arguments('-ferror-limit=1', language : 'cpp')
# endif
#Build configuration
conf_data = configuration_data()
foreach opt : ['four_velocity', 'column_major', 'progress_bar', 'shared_memory']
    conf_data.set10(opt.to_upper(), get_option(opt))
endforeach
conf_data.set10('FLOAT_PRECISION', get_option('precision') == 'single')
configure_file(input: 'build_options.hpp.in', output: 'build_options.hpp', configuration: conf_data)

#get numpy headers
numpy_include = run_command(py3, ['-c', 'import numpy; print(numpy.get_include())'], check: true).stdout().strip()
openmp_dep = dependency('openmp', language: 'cpp', required: true)
hdf5_dep = dependency('hdf5', language: 'cpp', required: true)

#Include Subdirectories
subdir('src')
subdir('simbi/afterglow')


general_dependencies = [
    openmp_dep,
    hdf5_dep,
    py3.dependency()
]
all_inc = include_directories(numpy_include, 'src')
#CPU Extension Module
py3.extension_module(
    'cpu_ext',
    ['src/cpu_ext.pyx'] + sources,
    include_directories: all_inc,
    cpp_args: ['-DGPU_CODE=0'],
    dependencies: general_dependencies,
    override_options: ['cython_language=cpp'],
    install: true,
    install_dir: py_install_dir
)

#Check and configure GPU compilation
if get_option('gpu_compilation').enabled()
    hdf5_inc = hdf5_dep.get_variable('includedir', cmake: 'HDF5_INCLUDE_DIRS')
    include_dirs = [
        meson.current_source_dir() + '/src',
        '.',
        meson.current_build_dir(),
        hdf5_inc,
    ]
    subdir('gpu')
endif
