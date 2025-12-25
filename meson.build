# =============================================================================
# meson.build
#
# main build configuration for simbi
# =============================================================================

project(
  'simbi',
  'c',
  'cpp',
  version: run_command('git', 'describe', '--tag', '--abbrev=0', check: false).stdout().strip(),
  license: 'MIT',
  default_options: {
    'cpp_std': 'c++20',
    'warning_level': '3',
  },
)

# common project arguments
add_project_arguments(
  [
    '-Wno-nested-anon-types',
    '-Wno-gnu-anonymous-struct',
  ],
  language: 'cpp',
)

# python and pybind11 setup
py3 = import('python').find_installation('python3', required: true)
py_install_dir = meson.current_source_dir() / 'simbi' / 'libs'
pybind11_dep = dependency('pybind11')

# limit error output during compilation
cc = meson.get_compiler('cpp')
if cc.get_id() == 'clang'
  add_project_arguments('-ferror-limit=1', language: 'cpp')
elif cc.get_id() == 'gcc'
  add_project_arguments('-fmax-errors=1', language: 'cpp')
endif

# build configuration
conf_data = configuration_data()
foreach opt : ['four_velocity', 'column_major']
  conf_data.set10(opt.to_upper(), get_option(opt))
endforeach
conf_data.set10('FLOAT_PRECISION', get_option('precision') == 'single')
conf_data.set(
  'BUILD_DATE',
  run_command('date', '+%Y-%m-%d %H:%M:%S', check: true).stdout().strip(),
)
configure_file(
  input: 'build_options.hpp.in',
  output: 'build_options.hpp',
  configuration: conf_data,
)

# dependencies
openmp_dep = dependency('openmp', language: 'cpp', required: true)
hdf5_dep = dependency('hdf5', language: 'cpp', required: true)

all_inc = include_directories('src')

# include subdirectories
subdir('src')
subdir('simbi/afterglow')

general_dependencies = [
  openmp_dep,
  hdf5_dep,
  py3.dependency(),
  pybind11_dep,
]

# compile once, link many
simbi_core = static_library(
  'simbi_core',
  sources + cpu_sources,
  include_directories: all_inc,
  dependencies: general_dependencies,
)

# cpu extension module
py3.extension_module(
  'cpu_ext',
  'src/bindings/cpu_binding.cpp',
  include_directories: all_inc,
  dependencies: general_dependencies,
  link_with: simbi_core,
  install: true,
  install_dir: py_install_dir,
)

# gpu compilation
if get_option('gpu_compilation').enabled()
  subdir('gpu')
endif

# tests
if get_option('build_tests')
  subdir('tests')
endif
