#Main Meson Build File

project(
    'simbi',
    'c',
    'cpp',
    'cython',
    version: run_command('git', 'describe', '--tag', '--abbrev=0', check: true).stdout().strip(),
    license: 'MIT',
    default_options: {'cpp_std': 'c++20', 'cpp_args': ['-march=native'], 'warning_level': '3'},
)

#Add common project arguments
add_project_arguments(
    [
        '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
        '-DCYTHON_EXTERN_C=extern "C++"',
        '-Wno-nested-anon-types',
        '-Wno-gnu-anonymous-struct',
    ],
    language: 'cpp',
)
#Python installation directory
py3 = import('python').find_installation('python3', required: true)
py_install_dir = meson.current_source_dir() + '/simbi/libs'
pybind11_dep = dependency('pybind11')

#show maximum of n errors before stopping the build
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
    py3.dependency(),
    pybind11_dep,
]
all_inc = include_directories(numpy_include, 'src')
#CPU Extension Module
py3.extension_module(
    'cpu_ext',
    ['src/core/pybind11/binding.cpp'] + sources,
    include_directories: all_inc,
    cpp_args: ['-DGPU_CODE=0'],
    dependencies: general_dependencies,
    override_options: ['cython_language=cpp'],
    install: true,
    install_dir: py_install_dir
)

#Check and configure GPU compilation
if get_option('gpu_compilation').enabled()
    fs = import('fs')
    hdf5_inc = hdf5_dep.get_variable('includedir', cmake: 'HDF5_INCLUDE_DIRS', default_value: '')

    # Try pkg-config if dependency variable failed
    if hdf5_inc == ''
        pkg_config = find_program('pkg-config', required: false)
        if pkg_config.found()
            pkg_result = run_command(pkg_config, '--cflags-only-I', 'hdf5', check: false)
            if pkg_result.returncode() == 0
                hdf5_inc_raw = pkg_result.stdout().strip()
                # Extract include path from -I flags
                hdf5_inc_list = []
                foreach flag : hdf5_inc_raw.split()
                    if flag.startswith('-I')
                        hdf5_inc_list += flag.substring(2)
                    endif
                endforeach
                if hdf5_inc_list.length() > 0
                    hdf5_inc = hdf5_inc_list[0]  # Use first include path
                endif
            endif
        endif
    endif

    # Try common system locations
    if hdf5_inc == ''
        common_hdf5_paths = [
            '/usr/include/hdf5',
            '/usr/local/include/hdf5',
            '/opt/homebrew/include',  # macOS Homebrew
            '/usr/include/hdf5/serial',  # Ubuntu/Debian
            '/usr/include',
            '/usr/local/include',
        ]

        foreach path : common_hdf5_paths
            if hdf5_inc == '' and fs.exists(path / 'hdf5.h')
                hdf5_inc = path
                message('Found HDF5 headers at: ' + path)
            endif
        endforeach
    endif

    # Try to find hdf5.h using compiler
    if hdf5_inc == ''
        cc = meson.get_compiler('c')
        if cc.has_header('hdf5.h', dependencies: hdf5_dep)
            # If compiler can find it with the dependency, we can use empty string
            # and rely on the dependency's include paths
            hdf5_inc = ''
            message('HDF5 headers found via compiler dependency')
        endif
    endif

    # Final validation
    if hdf5_inc == ''
        warning('Could not determine HDF5 include directory. GPU compilation may fail.')
    else
        # Validate the path exists
        if not fs.exists(hdf5_inc / 'hdf5.h')
            warning('HDF5 include directory found but hdf5.h not present at: ' + hdf5_inc)
        else
            message('Using HDF5 include directory: ' + hdf5_inc)
        endif
    endif

    # Build include directories list
    include_dirs = [
        meson.current_source_dir() / 'src',
        '.',
        meson.current_build_dir(),
    ]

    # Only add hdf5_inc if it's not empty
    if hdf5_inc != ''
        include_dirs += hdf5_inc
    endif
    subdir('gpu')
endif
